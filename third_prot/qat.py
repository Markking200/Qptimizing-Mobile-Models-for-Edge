#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantization-Aware Training (QAT) for MobileNetV2

This script:
1. Loads a pretrained MobileNetV2 from torchvision
2. Prepares it for QAT using PyTorch FX API
3. Performs short fine-tuning (few batches) to simulate QAT
4. Converts and saves the quantized model
"""

import platform
from pathlib import Path
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.ao.quantization import (
    get_default_qat_qconfig
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from datasets import load_dataset
from PIL import Image
import random
import os
import pathlib
import json
from datasets import load_dataset, Image as HFImage

# ---------------------------
# CONFIGURATION
# ---------------------------
backend = "fbgemm"    # for x86 CPUs
epochs = 2            # short fine-tuning
lr = 1e-4
batch_size = 32
max_batches_per_epoch = 100  # limit batches for quick demo

#save_dir = "../opt/models/mobilenet_v2_static"

def infer_default_engine() -> str:
    # ARM/Raspberry Pi → qnnpack, x86 → fbgemm
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return "qnnpack"
    return "fbgemm"

def resolve_weights(weights_enum_path: str):
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)

def qat_download(ctx:object)-> None:

    args = ctx.args
    model_name = args.model
    path = args.modeldir
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"

    out_dir = Path(path).expanduser().resolve()
    #out_dir = Path("../opt/models/mobilenet_v2_static").expanduser().resolve()
    out_dir = out_dir / f"{model_name}_qat"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in mod_reg:
        raise ValueError(f"Unknown model '{model_name}'. Supporting only: {', '.join(mod_reg)}")
    
    #os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    ctor, weights_path,_,dataset_name = mod_reg[model_name]
    weights = resolve_weights(weights_path)
    categories=weights.meta.get("categories", [])
    float_model = ctor(weights=weights)
    float_model = ctor(weights=weights).to(device)
    float_model.eval()

      # ---------------------------
    # TRANSFORMS & DATASET
    # ---------------------------
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Dataset aus Registry verwenden
    DATASET_MAP = {
        "ImageNet-1K": ("imagenet-1k", "validation"),
        # hier kannst du später mehr Aliases hinzufügen:
        # "Food101": ("food101", "validation"),
        # "CIFAR-10": ("cifar10", "test"),  # HF split heißt je nach dataset anders
    }

    hf_id, hf_split = DATASET_MAP.get(dataset_name, (dataset_name, "validation"))

     # ---------------------------
    # DATASET (local parquet only; no HF hub)
    # ---------------------------
    DATASET_MAP = {
        "ImageNet-1K": ("imagenet-1k", "validation"),
    }
    hf_id, hf_split = DATASET_MAP.get(dataset_name, (dataset_name, "validation"))

    LOCAL_HF_ROOT = Path(os.path.abspath("../data/hf_try1"))  # <- Ordner, wo "hf download" hinlädt
    # erwartet: ../data/imagenet1k_hf/data/validation-00000-of-00014.parquet ...
    parquet_dir = LOCAL_HF_ROOT / "data"

    if hf_id == "imagenet-1k" and hf_split == "validation":
        parquet_files = sorted(str(p) for p in parquet_dir.glob("validation-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"Keine validation-*.parquet gefunden in {parquet_dir}.\n"
                f"Bitte einmalig ausführen:\n"
                f"  hf download imagenet-1k --repo-type dataset "
                f"--include \"data/validation-*.parquet\" --local-dir {LOCAL_HF_ROOT}"
            )

        # lокales Dataset aus Parquet laden
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        # sorgt dafür, dass dataset[i]['image'] als PIL Image decodiert wird
        dataset = dataset.cast_column("image", HFImage(decode=True))
    else:
        raise RuntimeError(f"Kein lokaler Parquet-Loader konfiguriert für: {hf_id} / {hf_split}")

    # echtes Shuffle (kein buffer mehr nötig)
    dataset = dataset.shuffle(seed=0)

    logger.info("\n📦 Preparing DataLoader...")

    def collate(batch):
        imgs = []
        labels = []
        for b in batch:
            img = b["image"]
            # Sicherstellen, dass wir ein PIL-Image haben
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # Wichtig: immer auf 3 Kanäle bringen
            img = img.convert("RGB")
            imgs.append(transform(img))
            labels.append(b["label"])
        return torch.stack(imgs), torch.tensor(labels)

    logger.info("✅ DataLoader ready.")


    #, shuffle=True
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    #print(f"\n🚀 Loaded MobileNetV2 and prepared for QAT on {len(dataset)} samples.")
    logger.info("="*80)
    # ---------------------------
    # PREPARE FOR QAT
    # ---------------------------
    qconfig_dict = {"": get_default_qat_qconfig(backend)}
    example_inputs = (torch.randn(1, 3, 224, 224,device=device),)
    model_prepared = prepare_qat_fx(float_model.train(), qconfig_dict,example_inputs=example_inputs)
    model_prepared.to(device) 

    optimizer = optim.Adam(model_prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("\n🚀 Starting Quantization-Aware Training...")

    #---------------------------
    # TRAINING LOOP
    #---------------------------

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(loader):
            if i >= max_batches_per_epoch:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                logger.info(f"[Epoch {epoch+1}] Batch {i} | Loss: {loss.item():.4f}")

    logger.info("\n✅ QAT fine-tuning complete.")

    # ---------------------------
    # CONVERT & SAVE
    # ---------------------------
    model_prepared.eval()
    model_prepared.to("cpu") 
    model_quantized = convert_fx(model_prepared)

    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model_quantized, example)

    meta = {
        "architecture": model_name+"_qat_int8",           
        "source": "torchvision",
        "weights": weights_path,
        "image_size": 224,
        "resize_shorter_side": 256,
        "center_crop": 224,
        "normalize_mean": [
                                0.485,
                                0.456,
                                0.406
                            ],
        "normalize_std": [
                            0.229,
                            0.224,
                            0.225
                        ],
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
        "quantization": {
            "type": "qat",
            "dtype": "int8",
            "layers": "all",
            "engine": torch.backends.quantized.engine,
            "artifact": "torchscript",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


    torch.jit.save(traced, os.path.join(out_dir, f"{model_name}_qat_int8_ts.pt"))

    torch.save(model_quantized.state_dict(), os.path.join(out_dir, f"{model_name}_qat_int8_state.pt"))
    logger.info(f"💾 Saved quantized model to {out_dir}")