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
from datasets import load_dataset, load_from_disk
from context import Context
from download_model import resolve_weights
from torch.ao.quantization import QConfigMapping
from datasets import load_dataset
from datasets import Image as HFImage

IMAGE_POOL = 1000
SEED = 42

HF_CACHE_DIR = os.path.abspath("../data/huggingface_val")        # fürs erstmalige Download/Cache
HF_SAVED_DIR = os.path.abspath("../data/hf_saved_datasets") # für lokal gespeicherte Datasets


def static_download(ctx:Context) -> None:

    args = ctx.args
    path = args.modeldir
    model_name = args.model
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger


    if path is None:
        path = "../modelzoo"
    #Prepare for download
    out_dir = Path(path).expanduser().resolve()
    #out_dir = Path("../opt/models/mobilenet_v2_static").expanduser().resolve()
    #logger.info(out_dir)
    #logger.info(model_name)
    out_dir = out_dir / f"{model_name}_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name not in mod_reg:
        logger.error(f"Modell {model_name} not found in registry!")
        return

    logger.info(f"Start Static Quantization (FX) for: {model_name}")
    logger.info(f"Saving Directory: {out_dir}")

    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"

    ctor, weights_path,_,dataset_name = mod_reg[model_name]
    weights = resolve_weights(weights_path)

    model_fp32 = ctor(weights=weights).eval()
    categories=weights.meta.get("categories", [])

    preprocess = weights.transforms()
    image_size = 224 
    if hasattr(preprocess, 'crop_size'):
         image_size = preprocess.crop_size[0]
    elif hasattr(preprocess, 'resize_size'):
         image_size = preprocess.resize_size[0]

    logger.info(f"Using Image Size: {image_size}")
    
    
    qconfig = get_default_qconfig(torch.backends.quantized.engine)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    #old qconfig_dict 
    #qconfig_dict = {"": qconfig}o
    example_inputs = torch.randn(1, 3, image_size, image_size)
    prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_inputs=example_inputs)

    calib_images = IMAGE_POOL  # <- HIER festlegen
    calib_seed   = SEED

    # Dataset-Name aus MODEL_REGISTRY (4. Element im Tupel)
    # (falls du oben bereits sauber unpackst, nimm einfach dataset_name = dataset_name)
    #dataset_name = mod_reg[model_name][3]  # erwartet: (ctor, weights, task, dataset)

    # Map "Anzeige-Name" -> HuggingFace Dataset ID + Split
    # (erweitere das bei Bedarf um weitere Datasets)
    DATASET_MAP = {
        "ImageNet-1K": ("imagenet-1k", "validation"),
    }

    hf_id, hf_split = DATASET_MAP.get(dataset_name, (dataset_name, "validation"))

    logger.info(f"Calibration Dataset: {dataset_name} -> HF='{hf_id}' split='{hf_split}', samples={calib_images}, seed={calib_seed}")

    try:
        # >>> NEU: lokale saved version bevorzugen (offline-fähig)
        LOCAL_HF_DATASETS = Path(os.path.abspath("../data/hf_try1"))  # <- hier liegt hf download output
        parquet_dir = LOCAL_HF_DATASETS / "data"

        if hf_id == "imagenet-1k" and hf_split == "validation":
            files = sorted(str(p) for p in parquet_dir.glob("validation-*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"Keine validation-*.parquet gefunden in {parquet_dir}. "
                    f"Bitte vorher ausführen:\n"
                    f"  hf download imagenet-1k --repo-type dataset --include \"data/validation-*.parquet\" "
                    f"--local-dir {LOCAL_HF_DATASETS}"
                )

            calib_ds = load_dataset("parquet", data_files=files, split="train")
            # sorgt dafür, dass calib_ds[i]["image"] als PIL/Decoded Image kommt
            calib_ds = calib_ds.cast_column("image", HFImage(decode=True))
        else:
            raise RuntimeError(f"Kein lokaler Loader für hf_id={hf_id} split={hf_split} konfiguriert.")

        # reproducible shuffle und dann die ersten N nehmen
        calib_ds = calib_ds.shuffle(seed=calib_seed)

        def to_input(example):
            img = example.get("image", None)
            if img is None:
                return None
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            return preprocess(img).unsqueeze(0)

        n = min(calib_images, len(calib_ds))
        with torch.inference_mode():
            taken = 0
            for ex in calib_ds.select(range(n)):
                x = to_input(ex)
                if x is None:
                    continue
                prepared(x)
                taken += 1

        logger.info(f"Calibration done. Successfully used {taken} samples.")

    except Exception as e:
        logger.warning(f"HF calibration failed ({e}). Falling back to local images in ../data/calibration2")

        # Fallback: wie vorher lokale JPGs
        calib_dir = Path(os.path.abspath("../data/calibration2"))
        calibration_dataset = list(calib_dir.glob("*.jpg"))
        random.Random(calib_seed).shuffle(calibration_dataset)

        def load_and_preprocess(path: Path):
            try:
                img = Image.open(path).convert("RGB")
                return preprocess(img).unsqueeze(0)
            except Exception as ee:
                logger.warning(f"Konnte Bild {path} nicht laden: {ee}")
                return None

        with torch.inference_mode():
            for p in calibration_dataset[:calib_images]:
                x = load_and_preprocess(p)
                if x is None:
                    continue
                prepared(x)

    # nach der Calibration konvertieren
    model_int8 = quantize_fx.convert_fx(prepared).eval()

    example = torch.randn(1, 3, image_size, image_size)
    ts_int8 = torch.jit.trace(model_int8, example)
    ts_int8.save(str(out_dir/f"{model_name}_static_int8_ts.pt"))
    logger.info("Gespeichert:", f"{model_name}_static_int8_ts.pt")
    #print("Checkpoint 1")
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
        "quantization": {
            "type": "static",
            "dtype": "int8",
            "layers": "all",
            "engine": torch.backends.quantized.engine,
            "artifact": "torchscript",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Download abgeschlossen. Modell ist offline nutzbar.")
    #print("Checkpoint 2")