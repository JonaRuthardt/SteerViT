import os
import re
import random
import numpy as np
from PIL import Image

import wandb
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torch.nn.parallel import DistributedDataParallel

def seed_everything(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_config(config_path, data_config_path, checkpoint_path=None, unknown_args=None):
    config = OmegaConf.load(config_path)
    data_config = OmegaConf.load(data_config_path)

    if unknown_args:
        config = OmegaConf.merge(config, OmegaConf.from_cli(unknown_args))

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = OmegaConf.merge(config, checkpoint['config'])

    return config, data_config

def save_checkpoint(model, optimizer, scheduler, training_step, save_metric, config, suffix):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    checkpoint = {
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items() if any([kw in k for kw in ['cross_attn', 'connector', 'lin_seg_head']])}, #only save trainable params
        'optim_state_dict': optimizer.state_dict(),
        'training_step': training_step + 1,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'save_metric': save_metric,
        'wandb_run_id': wandb.run.id if wandb.run else None,
        'config': config,
        }
    print(f"Saving checkpoint at step {training_step} to {config['eval']['checkpoint_dir']}")
    os.makedirs(config['eval']['checkpoint_dir'], exist_ok = True)
    torch.save(checkpoint, os.path.join(config['eval']['checkpoint_dir'], config['wandb']['run_name'].lower() + "_" + str(checkpoint['wandb_run_id']) + "_" + str(suffix) + ".pth"))

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(model, DistributedDataParallel):
        model = model.module
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer is not None and 'optim_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('training_step', 0), checkpoint.get('wandb_run_id', None), checkpoint.get('save_metric', None)

def init_wandb(config, resume_run_id=None):
    if config['wandb']['logging']:
        if (resume_run_id is not None):
            wandb.init(id=resume_run_id, resume='must', entity=config["wandb"]["entity"], project=config["wandb"]['project'], config=OmegaConf.to_container(config, resolve=True))
        else:
            wandb.init(entity=config["wandb"]["entity"], project=config["wandb"]['project'], config=OmegaConf.to_container(config, resolve=True))
        wandb.run.name = config['wandb']['run_name']

def get_dataloaders(datasets, rank, world_size, config):
    if world_size > 1:
        train_sampler = DistributedSampler(
            datasets['train'],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['seed']
        )
        # shard val across GPUs as well (no shuffle for deterministic eval)
        val_sampler = DistributedSampler(
            datasets['val'],
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=config['seed'],
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        datasets['train'],
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # shuffle only in single GPU training
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "train_sampler": train_sampler}

DIRECTIONAL_RE = re.compile(
    r"\b(left|right|leftmost|rightmost|"
    r"clockwise|counterclockwise|east|west|"
    r"above|below|upper|lower|top|bottom)\b",
    re.IGNORECASE,
)

class JointImageMaskAugment:
    def __init__(
        self,
        split: str,
        img_res: int,
        crop_scale=(0.65, 1.0),
        crop_ratio=(0.85, 1.18),
        hflip_p=0.5,
        affine_p=0.3,
        max_translate=0.06,
        max_rotate=8,
        min_mask_pixels=16,
        crop_tries=10,
    ):
        self.split = split
        self.is_train = split == "train"
        self.img_res = img_res
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.hflip_p = hflip_p
        self.affine_p = affine_p
        self.max_translate = max_translate
        self.max_rotate = max_rotate
        self.min_mask_pixels = min_mask_pixels
        self.crop_tries = crop_tries
        self.image_only_augment = (
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.1,
                    )
                ],
                p=0.7,
            )
            if self.is_train
            else None
        )

    def _has_directional_language(self, prompt: str) -> bool:
        return bool(DIRECTIONAL_RE.search(prompt or ""))

    def _mask_pixels(self, mask: Image.Image) -> int:
        return int(np.count_nonzero(np.array(mask, dtype=np.uint8)))

    def __call__(self, image: Image.Image, mask: Image.Image, prompt: str):
        # mask should be PIL "L" with values 0/255 or 0/1
        if mask.mode != "L":
            mask = mask.convert("L")

        # 1. Random resized crop, but avoid cropping away the target completely.
        i, j, h, w = None, None, None, None
        if self.is_train:
            for _ in range(self.crop_tries):
                params = torch.empty(1).uniform_(0, 1)  # just to keep torch imported; not needed
                i_, j_, h_, w_ = self._get_random_resized_crop_params(
                    image, self.crop_scale, self.crop_ratio
                )

                cropped_mask = TF.crop(mask, i_, j_, h_, w_)
                if self._mask_pixels(cropped_mask) >= self.min_mask_pixels \
                        or self._mask_pixels(mask) == 0: # also break if negative (-> only zero pixels in mask) to avoid infinite loop in edge cases
                    i, j, h, w = i_, j_, h_, w_
                    break

        if i is not None:
            image = TF.resized_crop(
                image, i, j, h, w,
                size=[self.img_res, self.img_res],
                interpolation=InterpolationMode.BICUBIC,
            )
            mask = TF.resized_crop(
                mask, i, j, h, w,
                size=[self.img_res, self.img_res],
                interpolation=InterpolationMode.NEAREST,
            )
        else:
            image = TF.resize(
                image,
                [self.img_res, self.img_res],
                interpolation=InterpolationMode.BICUBIC,
            )
            mask = TF.resize(
                mask,
                [self.img_res, self.img_res],
                interpolation=InterpolationMode.NEAREST,
            )

        # 2. Horizontal flip only when prompt has no directional language.
        if (
            random.random() < self.hflip_p
            and not self._has_directional_language(prompt)
            and self.is_train
        ):
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 3. Mild affine transform.
        if self.is_train and random.random() < self.affine_p:
            angle = random.uniform(-self.max_rotate, self.max_rotate)

            max_dx = int(self.max_translate * self.img_res)
            max_dy = int(self.max_translate * self.img_res)
            translate = (
                random.randint(-max_dx, max_dx),
                random.randint(-max_dy, max_dy),
            )

            scale = random.uniform(0.9, 1.1)
            shear = [random.uniform(-3, 3), random.uniform(-3, 3)]

            aug_image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
                center=None,
            )
            aug_mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
                center=None,
            )

            if self._mask_pixels(aug_mask) >= self.min_mask_pixels:
                image = aug_image
                mask = aug_mask


        target = torch.from_numpy(np.array(mask) > 0)

        # 4. Image-only color jitter.
        if self.image_only_augment is not None:
            image = self.image_only_augment(image)

        return image, target

    @staticmethod
    def _get_random_resized_crop_params(img, scale, ratio):
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback: center crop
        in_ratio = width / height
        if in_ratio < ratio[0]:
            w = width
            h = int(round(w / ratio[0]))
        elif in_ratio > ratio[1]:
            h = height
            w = int(round(h * ratio[1]))
        else:
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
