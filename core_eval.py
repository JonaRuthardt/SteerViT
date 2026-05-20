#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from steervit import SteerViT

class ImageCollator:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        return torch.stack([
            self.transform(row["image"].convert("RGB"))
            for row in batch
        ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="JonaRuthardt/CORE")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def encode_dataset(dataset, model, transform, prompt, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=ImageCollator(transform),
    )

    feats = []

    for images in tqdm(loader, desc=f"Encoding: {prompt[:60]}"):
        images = images.to(device, non_blocking=True)
        texts = [prompt] * images.size(0)

        x = model.get_global_features(images=images, texts=texts)
        x = F.normalize(x.float(), dim=-1)

        if torch.isnan(x).any():
            raise RuntimeError(f"NaN features for prompt: {prompt}")

        feats.append(x)

    return torch.cat(feats, dim=0)


def precision_at_k_for_object(features, object_indices, image_ids, image_id_to_indices, k):
    scores = []
    positives = set(object_indices)

    for i in object_indices:
        similarities = features[i] @ features.T

        # Exclude self and same-source-image variants.
        for j in image_id_to_indices[image_ids[i]]:
            similarities[j] = -math.inf

        valid_positives = positives - set(image_id_to_indices[image_ids[i]])
        if not valid_positives:
            continue

        top_k = torch.topk(similarities, k=k).indices.tolist()
        precision = sum(j in valid_positives for j in top_k) / k
        scores.append(precision)

    return scores


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset, split="test")

    model = SteerViT.from_pretrained(args.checkpoint, device=args.device)
    model.eval()
    transform = model.get_transforms()

    results = {}
    all_scores = []

    scenes = sorted(set(dataset["scene"]))

    for scene in scenes:
        scene_indices = [
            i for i, s in enumerate(dataset["scene"])
            if s == scene
        ]
        scene_dataset = dataset.select(scene_indices)

        objects = scene_dataset["object"]
        image_ids = scene_dataset["image_id"]
        prompts = scene_dataset["prompt"]

        object_to_indices = defaultdict(list)
        image_id_to_indices = defaultdict(list)
        object_to_prompt = {}

        for i, (obj, image_id, prompt) in enumerate(zip(objects, image_ids, prompts)):
            object_to_indices[obj].append(i)
            image_id_to_indices[image_id].append(i)

            if obj in object_to_prompt and object_to_prompt[obj] != prompt:
                raise ValueError(
                    f"Conflicting prompts for scene={scene}, object={obj}: "
                    f"{object_to_prompt[obj]!r} vs {prompt!r}"
                )

            object_to_prompt[obj] = prompt

        results[scene] = {}

        for obj in sorted(object_to_indices):
            prompt = object_to_prompt[obj]

            features = encode_dataset(
                dataset=scene_dataset,
                model=model,
                transform=transform,
                prompt=prompt,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )

            scores = precision_at_k_for_object(
                features=features,
                object_indices=object_to_indices[obj],
                image_ids=image_ids,
                image_id_to_indices=image_id_to_indices,
                k=args.k,
            )

            if not scores:
                continue

            precision = sum(scores) / len(scores)

            results[scene][obj] = {
                f"precision@{args.k}": precision,
                "n_queries": len(scores),
                "prompt": prompt,
            }

            all_scores.extend(scores)

            print(f"{scene}/{obj}: P@{args.k} = {precision:.4f} ({len(scores)} queries)")

    global_precision = sum(all_scores) / len(all_scores)

    output = {
        "global": {
            f"precision@{args.k}": global_precision,
            "n_queries": len(all_scores),
        },
        "per_scene": results,
    }

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=4)

    print(f"\nGlobal Precision@{args.k}: {global_precision:.4f}")


if __name__ == "__main__":
    main()