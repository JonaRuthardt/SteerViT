import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import datasets

from .train_utils import JointImageMaskAugment

def get_coco_path(coco_path, img_id):
    if not img_id.endswith('.jpg'):
        img_id += '.jpg'
    image_path = os.path.join(coco_path, img_id.split("_")[1], img_id)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"COCO image not found: {image_path}")
    return image_path

def get_visual_genome_path(vg_path, img_id):
    if not img_id.endswith('.jpg'):
        img_id += '.jpg'
    image_path_1 = os.path.join(vg_path, "VG_100K", img_id)
    image_path_2 = os.path.join(vg_path, "VG_100K_2", img_id)
    if os.path.isfile(image_path_1):
        return image_path_1
    elif os.path.isfile(image_path_2):
        return image_path_2
    else:
        raise FileNotFoundError(f"Visual Genome image not found: {image_path_1} or {image_path_2}")

def get_mapillary_path(mapillary_path, img_id):
    if not img_id.endswith('.jpg'):
        img_id += '.jpg'
    image_path = os.path.join(mapillary_path, img_id)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Mapillary image not found: {image_path}")
    return image_path

class ReferentialSegmentationDataset(Dataset):
    def __init__(self, config, split, transforms=None, img_res=None, patch_size=None):
        self.config = config
        self.split = split
        self.dataset = datasets.load_dataset(self.config.dataset, split=split)
        self.transforms = transforms

        self.patch_size = patch_size
        self.img_res = img_res
        self.joint_augment = JointImageMaskAugment(split=split, img_res=img_res) if img_res else None

    def setup_augmentation(self, patch_size, img_res):
        self.patch_size = patch_size
        self.img_res = img_res
        self.joint_augment = JointImageMaskAugment(split=self.split, img_res=img_res)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load image depending on the dataset
        if item["image_id"].startswith("COCO"):
            image_path = get_coco_path(os.path.expandvars(self.config.mscoco), item["image_id"])
        elif "visual genome" in item["dataset"]:
            image_path = get_visual_genome_path(os.path.expandvars(self.config.visual_genome), item["image_id"])
        elif "mapillary" in item["dataset"]:
            image_path = get_mapillary_path(os.path.expandvars(self.config.mapillary), item["image_id"])
        else:
            raise NotImplementedError(f"Unknown dataset: {item['dataset']}")

        image = Image.open(image_path).convert("RGB")

        target = item["mask"].convert("1")

        if self.joint_augment:
            image, target = self.joint_augment(image, target, item["referential_expression"])
        if self.transforms:
            image = self.transforms(image)

        if self.patch_size is not None:
            target_patch = F.avg_pool2d(
                target.float()[None, None],  # (1, 1, H, W)
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ).flatten().float()
            denom = target_patch.sum()
            if denom > 0:
                target_patch = target_patch / denom

        else:
            target_patch = None

        return {
            "image": image,
            "prompt": item["referential_expression"],
            "target_pixel_wise": target,
            "target_patch_wise": target_patch,
            "dataset": item["dataset"]
        }