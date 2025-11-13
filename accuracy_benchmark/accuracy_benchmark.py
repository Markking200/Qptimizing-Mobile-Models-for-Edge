#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from time import time
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# -------- Config --------
MODEL_PATH = "../opt/models/mobilenet_v2_static/image200.pt"   # dein TorchScript-INT8-Modell
BATCH_SIZE = 64                              # CPU-geeignet; an deine Maschine anpassen
NUM_WORKERS = 4                              # DataLoader-Worker (0 auf Windows)
PIN_MEMORY = False                           # CPU-Only -> False
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SHORTER = 256
CROP_SIZE = 224
SPLIT = "validation"                         # 50k val
SUBSET = None                                # z.B. "validation[:5000]" für schnellen Test
MAX_SAMPLES = 5000                        # z.B. 1000 für schnellen Test (überschreibt SUBSET)


class HFImageNetDataset(IterableDataset):
    """
    Wrappt ein HuggingFace-ImageNet-Split und appliziert torchvision-Preprocessing.
    Erwartet 'image' (PIL) und 'label' (int).
    """
    def __init__(self, hf_ds, tfm, max_samples: int = None):
        self.ds = hf_ds
        self.tfm = tfm
        self.max_samples = max_samples

    # def __len__(self):
    #     return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        img: Image.Image = item["image"].convert("RGB")
        x = self.tfm(img)          # (3,H,W) float tensor
        y = int(item["label"])     # integer class id
        return x, y
    
    def __iter__(self):
        count = 0
        for item in self.ds:
            img = item["image"].convert("RGB")
            x = self.tfm(img)
            y = int(item["label"])
            yield x, y
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break


def build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(RESIZE_SHORTER),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int = 5) -> int:
    """
    Zählt, wie viele Targets in den Top-k Vorhersagen liegen.
    logits: (B, C), target: (B,)
    """
    # topk returns (values, indices)
    _, pred = torch.topk(logits, k, dim=1)
    # Vergleiche jede Zeile: target[b] in pred[b, :]
    correct = (pred == target.view(-1, 1)).any(dim=1).sum().item()
    return correct


def main():
    # 1) Modell laden (TorchScript INT8 -> CPU)
    print(f"[INFO] Loading TorchScript model from: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    print("[INFO] Model loaded. Running on CPU (INT8).")

    # 2) Dataset laden (HuggingFace)
    split_str = SUBSET if SUBSET is not None else SPLIT
    print(f"[INFO] Loading ImageNet-1k split from Hugging Face: {split_str}")
    hf_ds = load_dataset("imagenet-1k", split=split_str,streaming=True)#cache_dir="/home/marceldavis/University/BA/FirstZoo/data/huggingfaceval")
    #n_samples = len(hf_ds)
    #print(f"[INFO] Dataset size: {n_samples} images")

    # 3) Preprocessing + DataLoader
    preprocess = build_preprocess()
    ds = HFImageNetDataset(hf_ds, preprocess,max_samples=MAX_SAMPLES)

    # Collate: default stack reicht (x: (3,H,W), y: int)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    # 4) Eval loop
    total = 0
    top1_correct = 0
    top5_correct = 0

    t0 = time()
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Evaluating"):#, total=math.ceil(n_samples / BATCH_SIZE)):
            # Modell ist CPU/INT8; keine .to('cuda') hier!
            logits = model(x)               # (B, 1000)
            # Top-1
            pred1 = logits.argmax(dim=1)    # (B,)
            top1_correct += (pred1 == y).sum().item()
            # Top-5
            top5_correct += topk_correct(logits, y, k=5)
            total += y.size(0)
            print(f"\r[INFO] Processed {total} samples...", end="", flush=True)

    dt = time() - t0
    top1 = top1_correct / total
    top5 = top5_correct / total

    print("\n================= Results =================")
    print(f"Samples evaluated : {total}")
    print(f"Top-1 Accuracy    : {top1:.4%}")
    print(f"Top-5 Accuracy    : {top5:.4%}")
    print(f"Total time (s)    : {dt:.1f}")
    print(f"Throughput (img/s): {total / dt:.1f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
