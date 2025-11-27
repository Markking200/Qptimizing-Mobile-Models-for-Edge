from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import platform
import random
from typing import Tuple, List
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Callable, Optional
import torch.nn.functional as F
from torch.ao.quantization import get_default_qconfig
#from torch.ao.quantization.fx import prepare_fx, convert_fx
from torch.quantization import quantize_fx
import os, platform
from datasets import load_dataset
from context import Context
from download_model import resolve_weights
from torch.ao.quantization import QConfigMapping


def static_download(ctx:Context) -> None:

    args = ctx.args
    path = args.modeldir
    model_name = args.model
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    #Prepare for download
    out_dir = Path(path).expanduser().resolve()
    #out_dir = Path("../opt/models/mobilenet_v2_static").expanduser().resolve()
    out_dir = out_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in mod_reg:
        logger.error(f"Modell {model_name} nicht in der Registry gefunden!")
        return

    logger.info(f"Starte Static Quantization (FX) für: {model_name}")
    logger.info(f"Zielordner: {out_dir}")

    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"

    ctor, weights_path = mod_reg[model_name]
    weights = resolve_weights(weights_path)

    model_fp32 = ctor(weights=weights).eval()
    categories=weights.meta.get("categories", [])

    preprocess = weights.transforms()
    image_size = 224 
    if hasattr(preprocess, 'crop_size'):
         image_size = preprocess.crop_size[0]
    elif hasattr(preprocess, 'resize_size'):
         image_size = preprocess.resize_size[0]

    logger.info(f"Verwende Image Size: {image_size}")
    
    
    qconfig = get_default_qconfig(torch.backends.quantized.engine)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    #old qconfig_dict 
    #qconfig_dict = {"": qconfig}
    example_inputs = torch.randn(1, 3, image_size, image_size)
    prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_inputs=example_inputs)

    def load_and_preprocess(path: Path):
        try:
            img = Image.open(path).convert("RGB")
            return preprocess(img).unsqueeze(0)
        except Exception as e:
            logger.warning(f"Konnte Bild {path} nicht laden: {e}")
            return None
    
    #calib_dir = Path("/home/University/BA/archive\(1\)//allimages")
    calib_dir = Path(os.path.abspath("../data/calibration2"))  # lege dort 50–200 Bilder ab (beliebige natürliche Fotos)
    #print(calib_dir)
    #print(list(calib_dir.glob("*.jpg")))
    #dataset = load_dataset("imagenet-1k",split="validation")
    calibration_dataset = list(calib_dir.glob("*.jpg"))
    try:
        with torch.inference_mode():
                #for i in range(200):
                imageindex = 0
                #for fname in list(calib_dir.glob("*.jpg"))[:6500]:
                for i in range(100):
                    #y = random.randint(0,49999)
                    #torch.rand(1,3,224, 224)
                    #print(fname)
                    x = load_and_preprocess(calibration_dataset[i])
                    #x = dataset[y]['image']
                    prepared(x)  # nur forward, keine Labels nötig
    except Exception as e:
        print(e)
    
    model_int8 = quantize_fx.convert_fx(prepared).eval()

    example = torch.randn(1, 3, image_size, image_size)
    ts_int8 = torch.jit.trace(model_int8, example)
    ts_int8.save(str(out_dir/f"{model_name}_static_int8_ts.pt"))
    print("Gespeichert:", f"{model_name}_static_int8_ts.pt")
    print("Checkpoint 1")
    # with torch.inference_mode():
    #     y = ts_int8(torch.randn(1,3,image_size,image_size))
    # print("OK, Output-Shape:", tuple(y.shape))

    meta = {
        "architecture": model_name+"_static_int8",            # wichtig für infer()
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
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Download abgeschlossen. Modell ist offline nutzbar.")
    print("Checkpoint 2")