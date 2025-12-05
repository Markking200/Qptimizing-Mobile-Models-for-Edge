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
import sys
from pathlib import Path


from third_prot.context import Context

# -------- Config --------
MODEL_PATH = "/home/marceldavis/University/BA/FirstZoo/opt/models/efficientnet_b0/efficientnet_b0_ts.pt"   # dein TorchScript-INT8-Modell
BATCH_SIZE = 64                              # CPU-geeignet; an deine Maschine anpassen
NUM_WORKERS = 12                              # DataLoader-Worker (0 auf Windows)
PIN_MEMORY = False                           # CPU-Only -> False
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_SHORTER = 256
CROP_SIZE = 224
SPLIT = "validation"                         # 50k val
SUBSET = None                                # z.B. "validation[:5000]" für schnellen Test
MAX_SAMPLES = 50                        # number of samples to eval (total will be MAX_SAMPLES * NUM_WORKERS if NUM_WORKERS>0)
MODEL_PATH = "../modelzoo/"


# currently for streaming = True => IterableDataset, if dataset is downloaded HFImageNetDataset can inherit from Dataset
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

def main(ctx: Context = None):
    if ctx is None:
        return
    args = ctx.args
    path = args.modeldir
    samplesize = args.samplesize
    all = args.all
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if all == False:
        if path is None:
            logger.error("Please provide model directory path using --modeldir")
            return
        dir = Path(path).expanduser().resolve()
        pt_path = next(
                (p for p in dir.rglob("*.pt") if "dict" not in p.name.lower()),
                None
            )
        if pt_path is None:
            raise FileNotFoundError(f"Keine .pt Datei in {dir} gefunden")
        if samplesize is None:
            samplesize = MAX_SAMPLES
        benchmark(pt_path, samplesize,logger)
    
    if all == True:
        for model_name in mod_reg:
            dir = Path(MODEL_PATH).expanduser().resolve() / model_name
            pt_path = next(
                            (p for p in dir.rglob("*.pt") if "dict" not in p.name.lower()),
                            None
                        )
            if pt_path is None:
                logger.error(f"Keine .pt Datei in {dir} gefunden, überspringe...")
                return
            if samplesize is None:
                samplesize = MAX_SAMPLES
            logger.info(f"Benchmarking model: {model_name}")
            benchmark(pt_path, samplesize,logger)


def benchmark(path, size,logger):
    


    path_str = str(path.resolve())

    # 1) Modell laden (TorchScript INT8 -> CPU)
    logger.info(f"[INFO] Loading TorchScript model from: {path}")
    model = torch.jit.load(path, map_location="cpu") #"cuda" if torch.cuda.is_available() else 
    #activates inference mode, forecast gets deterministic, 
    model.eval()
    logger.info("[INFO] Model loaded. Running on CPU (INT8).")

    # 2) Dataset laden (HuggingFace)
    split_str = SUBSET if SUBSET is not None else SPLIT
    logger.info(f"[INFO] Loading ImageNet-1k split from Hugging Face: {split_str}")
    hf_ds = load_dataset("imagenet-1k", split=split_str,streaming=True)#cache_dir="/home/marceldavis/University/BA/FirstZoo/data/huggingfaceval")
    #n_samples = len(hf_ds)
    #print(f"[INFO] Dataset size: {n_samples} images")

    # 3) Preprocessing + DataLoader
    preprocess = build_preprocess()
    ds = HFImageNetDataset(hf_ds, preprocess,max_samples=size)

    # Collate: default stack reicht (x: (3,H,W), y: int)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,#NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    # 4) Eval loop
    total = 0
    top1_correct = 0
    top5_correct = 0

    #zw1 = None
    #zw2 = None
    #zw3 = None

    t0 = time()
    with torch.inference_mode():
        #tqdm shows progress bar with iterations per second
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
            
            #zw1 = logits
            #zw2 = x
            #zw3 = y
            #break
    #print(zw1)
    #print(zw2)
    #print(zw3)

    dt = time() - t0
    top1 = top1_correct / total
    top5 = top5_correct / total

    logger.info("\n================= Results =================")
    logger.info(f"Samples evaluated : {total}")
    logger.info(f"Top-1 Accuracy    : {top1:.4%}")
    logger.info(f"Top-5 Accuracy    : {top5:.4%}")
    logger.info(f"Total time (s)    : {dt:.1f}")
    logger.info(f"Throughput (img/s): {total / dt:.1f}")
    logger.info("===========================================\n")

    log_path = "accuracy_benchmark_results.txt"

    line = (
        "\n================= Results =================\n"
        f"Model: {path}\n"
        f"Samples: {total}, \n"
        f"Top-1: {top1:.4%}, \n"
        f"Top-5: {top5:.4%}, \n"
        f"Time: {dt:.1f}s, \n"
        f"Throughput: {total/dt:.1f} img/s\n"
        "===========================================\n"
    )

    # 'a' = append (anhängen) → erstellt die Datei, falls sie noch nicht existiert
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

    logger.info(f"[INFO] Results appended to {log_path}")


if __name__ == "__main__":
    main()
