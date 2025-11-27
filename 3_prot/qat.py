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

# ---------------------------
# CONFIGURATION
# ---------------------------
backend = "fbgemm"    # for x86 CPUs
epochs = 2            # short fine-tuning
lr = 1e-4
batch_size = 32

save_dir = "../opt/models/mobilenet_v2_static"

def qat_download(ctx:object)-> None:
    
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------
    # LOAD MODEL
    # ---------------------------
    float_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
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

    # use a small subset from ImageNet-1k validation (via HuggingFace)
    dataset = load_dataset("imagenet-1k", split="validation",cache_dir="../data/huggingface1proz", streaming=True)

    print("\n📦 Preparing DataLoader...")

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

    print("✅ DataLoader ready.")


    #, shuffle=True
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    #print(f"\n🚀 Loaded MobileNetV2 and prepared for QAT on {len(dataset)} samples.")
    print("="*80)
    # ---------------------------
    # PREPARE FOR QAT
    # ---------------------------
    qconfig_dict = {"": get_default_qat_qconfig(backend)}
    example_inputs = (torch.randn(1, 3, 224, 224,device=device),)
    model_prepared = prepare_qat_fx(float_model.train(), qconfig_dict,example_inputs=example_inputs)

    optimizer = optim.Adam(model_prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Starting Quantization-Aware Training...")

    max_batches_per_epoch = 100  # limit batches for quick demo

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
                print(f"[Epoch {epoch+1}] Batch {i} | Loss: {loss.item():.4f}")

    print("\n✅ QAT fine-tuning complete.")

    # ---------------------------
    # CONVERT & SAVE
    # ---------------------------
    model_prepared.eval()
    model_quantized = convert_fx(model_prepared)

    example = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model_quantized, example)
    torch.jit.save(traced, os.path.join(save_dir, "mobilenet_v2_qat_int8_ts.pt"))

    torch.save(model_quantized.state_dict(), os.path.join(save_dir, "mobilenet_v2_qat_int8_state.pt"))
    print(f"💾 Saved quantized model to {save_dir}")