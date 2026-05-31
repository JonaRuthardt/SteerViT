import os
import time
import math
import argparse
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict

import wandb
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

from src.steervit.model import SteerViT
from src.steervit.train_data import ReferentialSegmentationDataset
import src.steervit.train_utils as utils

def setup_ddp(rank, world_size, MASTER_PORT=None):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(10000 + int(os.environ.get('SLURM_JOB_ID', 0)) % 50000) if MASTER_PORT is None else str(MASTER_PORT)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=36000))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    torch.distributed.destroy_process_group()

def is_main_process():
    return torch.distributed.get_rank() == 0

def sync_metric_totals(metric_sums, metric_counts):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return metric_sums, metric_counts

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (dict(metric_sums), dict(metric_counts)))

    synced_sums = defaultdict(float)
    synced_counts = defaultdict(int)
    for sums, counts in gathered:
        for key, value in sums.items():
            synced_sums[key] += value
        for key, value in counts.items():
            synced_counts[key] += value

    return synced_sums, synced_counts

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, targets, reduction='mean', bbox = None):
        if not bbox:
            log_probs = F.log_softmax(logits, dim = -1)
            loss = -(targets * log_probs).sum(dim=-1)
            if reduction is None:
                return loss
            return loss.mean()

@torch.no_grad()
def run_validation(model, loader, criterion, config, step, rank):
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    model.eval()

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    def update_metric(name, value, dataset=None):
        key = f"{name}/val/{dataset}" if dataset is not None else f"{name}/val/AVG"
        metric_sums[key] += float(value)
        metric_counts[key] += 1

    for sample in tqdm(loader, total=len(loader), desc="Running Validation", disable=not is_main_process()):

        images = sample['image'].to(device, non_blocking=True)
        targets = sample['target_patch_wise'].to(device, non_blocking=True)
        dataset_names = sample['dataset']
        texts = sample['prompt']

        segmentation_logits = model(images, texts, return_segmentation_logits=True)


        loss = criterion(segmentation_logits, targets.view(targets.size(0), -1), reduction=None)
        pmass = (F.softmax(segmentation_logits, dim=-1) * (targets > 0).float()).sum(dim=-1)

        for dataset, loss, mass in zip(dataset_names, loss, pmass):
            update_metric("loss", loss.item())
            update_metric("pmass", mass.item())
            update_metric("loss", loss.item(), dataset)
            update_metric("pmass", mass.item(), dataset)

    metric_sums, metric_counts = sync_metric_totals(metric_sums, metric_counts)
    val_stats = {
        key: metric_sums[key] / metric_counts[key]
        for key in sorted(metric_sums)
    }

    if is_main_process():
        if wandb.run:
            wandb.log(val_stats, step=step)
        val_stats['save_metric'] = val_stats[f'{config["eval"]["save_metric"]}/val/AVG']

    dist.barrier()
    torch.cuda.empty_cache()
    return val_stats

def train_epoch(model, criterion, optimizer, scheduler, loaders, config, current_step, epoch, train_sampler, rank):
    model.train()
    model.module.text_model.eval()
    device = torch.cuda.current_device()
    total_loss = 0

    if train_sampler is not None:
        # Set the epoch for the DistributedSampler to ensure different shuffling in each epoch
        train_sampler.set_epoch(epoch)
    pbar = tqdm(enumerate(loaders['train']), total=len(loaders['train']), desc="train", disable=not is_main_process())

    for batch_idx, sample in pbar:

        images = sample['image'].to(device, non_blocking=True)

        targets = sample['target_patch_wise'].to(device, non_blocking=True)

        optimizer.zero_grad()

        segmentation_logits = model(images, sample['prompt'], return_segmentation_logits=True)

        loss = criterion(segmentation_logits.view(segmentation_logits.size(0), -1), targets)
        total_loss += loss.item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config['train']['optim']['max_grad_norm'])

        optimizer.step()

        if scheduler:
            scheduler.step()

        # Logging
        if is_main_process() and wandb.run:
            train_log = {
            "trainer/lr": optimizer.param_groups[0]['lr'],
            "trainer/loss": loss.detach().item(),
            "trainer/grad_norm" : grad_norm.item(),
            }
            wandb.log(train_log, step=current_step)
        pbar.set_postfix({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Validation and checkpointing
        if current_step > 0 and ((current_step) % config['eval']["every_n_steps"]) == 0:
            val_stats = run_validation(model, loaders['val'], criterion, config, current_step, rank)
            model.train()
            model.module.text_model.eval()
            if is_main_process():
                if config['eval']['save_metric'] == 'pmass' and val_stats['save_metric'] > config['save_metric'] or \
                    config['eval']['save_metric'] == 'loss' and val_stats['save_metric'] < config['save_metric']:
                    config['save_metric'] = val_stats['save_metric']
                    utils.save_checkpoint(model, optimizer, scheduler, current_step, config['save_metric'], config, suffix='best')
                utils.save_checkpoint(model, optimizer, scheduler, current_step, val_stats['save_metric'], config, suffix=f"last")

        if is_main_process() and config.eval.checkpoint_every and ((current_step) % config.eval.checkpoint_every) == 0:
            utils.save_checkpoint(model, optimizer, scheduler, current_step, config['save_metric'], config, suffix=f"{current_step}")

        current_step += 1
        if current_step >= config['train']['train_iters']:
            break

    return {"loss": total_loss/len(loaders['train'])}, current_step

def trainer(rank, world_size, config, data_config, checkpoint_path=None):
    # Setup distributed training
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    config['save_metric'] = -1 if config['eval']['save_metric'] == 'pmass' else float('inf')
    wandb_run_id = None

    # Model Setup
    model = SteerViT(config)
    model.to(device)
    #TODO verify that everything is (un)frozen as intended
    for name, param in model.named_parameters():
        if "cross_attn" in name or "connector" in name or "lin_seg_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Data Setup
    datasets_dict = {
        'train': ReferentialSegmentationDataset(data_config, "train", transforms=model.get_transforms(), img_res=model.image_size[0], patch_size=model.patch_size),
        'val': ReferentialSegmentationDataset(data_config, "validation", transforms=model.get_transforms(), img_res=model.image_size[0], patch_size=model.patch_size),
    }
    loaders = utils.get_dataloaders(datasets_dict, rank, world_size, config)
    train_sampler = loaders["train_sampler"]

    # Training Setup
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    current_step = 0
    connector_params = list(model.module.connector.parameters())
    connector_ids = {id(p) for p in connector_params}
    gatedCA_params = [p for p in model.module.parameters() if p.requires_grad and id(p) not in connector_ids]

    optimizer = optim.AdamW(
        [
            {"params": connector_params, "weight_decay": 0.01},
            {"params": gatedCA_params,     "weight_decay": 0.1},
        ],
        lr=float(config['train']['optim']['final_lr']) * world_size,
    )

    lin_scaling_coeff = config['train']['optim']['lr_scaling_coeff_ddp']
    initial_lr = config['train']['optim']['initial_lr'] * lin_scaling_coeff
    min_lr  = float(config['train']['optim']['final_lr']) * lin_scaling_coeff
    ratio   = min_lr / initial_lr
    decay_steps = config['train']['optim']['cos_decay_steps']
    # Set the (scaled) starting LR explicitly
    for pg in optimizer.param_groups:
        pg['lr'] = initial_lr

    def cosine_then_const(step: int):
        if step >= decay_steps:
            return ratio
        cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
        return ratio + (1.0 - ratio) * cos

    scheduler = LambdaLR(optimizer, lr_lambda=cosine_then_const)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        current_step, wandb_run_id, save_metric = utils.load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        config['save_metric'] = save_metric
        if is_main_process():
            print(f"Loaded checkpoint saved at step {current_step}")


    if is_main_process():
        utils.init_wandb(config, resume_run_id=wandb_run_id)

    criterion = SoftCrossEntropyLoss()

    # Initial evaluation
    if config['eval']['init_eval']:
        val_stats = run_validation(model, loaders['val'], criterion, config, current_step, rank)
        if is_main_process():
            config['save_metric'] = val_stats['save_metric']

    # Training loop
    epoch = 0
    while True:

        if is_main_process() and wandb.run:
            wandb.log({"trainer/epoch": epoch}, step=current_step)

        s = time.time()
        _, current_step = train_epoch(model, criterion, optimizer, scheduler, loaders, config, current_step, epoch, train_sampler, rank)
        epoch_time = int((time.time()-s)/60)

        if is_main_process() and wandb.run:
            wandb.log({"trainer/epoch_time": epoch_time}, step=current_step)

        train_iters = config['train']['train_iters']
        if current_step >= train_iters:
            if is_main_process():
                print(f"| Reached train_iters={train_iters} at step {current_step}. Stopping training.")
            break

        epoch += 1

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SteerViT Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the model and training config file")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Path to the data config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (.pth)")
    args, unknown_args = parser.parse_known_args()

    config, data_config = utils.load_config(args.config, args.data_config, args.checkpoint, unknown_args)

    world_size = torch.cuda.device_count()
    utils.seed_everything(config.seed)
    if world_size == 0:
        print("No GPUs found. Please run on a machine with at least one GPU.")
    elif world_size > 1:
        print(f"| Starting distributed training on {world_size} GPUs")
        torch.multiprocessing.spawn(trainer, args=(world_size, config, data_config, args.checkpoint), nprocs=world_size, join=True)
    else:
        trainer(0, world_size, config, data_config, checkpoint_path=args.checkpoint)