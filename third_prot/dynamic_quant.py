#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import annotations
import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Callable, List, Tuple, Dict
from context import Context

import torch
from torchvision import models, transforms
from download_model import resolve_weights

# ---------------- Registry ----------------
# Sichtbarer Modellname -> (ctor, weights_enum_path)
# MODEL_REGISTRY: Dict[str, Tuple[Callable[..., torch.nn.Module], str]] = {
#     "mobilenet_v2":        (models.mobilenet_v2,       "MobileNet_V2_Weights.IMAGENET1K_V1"),
#     "mobilenet_v3_small":  (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
#     "mobilenet_v3_large":  (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),
#     "shufflenet_v2_x0_5":  (models.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
#     "shufflenet_v2_x1_0":  (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),
#     "mnasnet0_5":          (models.mnasnet0_5,         "MNASNet0_5_Weights.IMAGENET1K_V1"),
#     "mnasnet1_0":          (models.mnasnet1_0,         "MNASNet1_0_Weights.IMAGENET1K_V1"),
#     "squeezenet1_0":       (models.squeezenet1_0,      "SqueezeNet1_0_Weights.IMAGENET1K_V1"),
#     "efficientnet_b0":     (models.efficientnet_b0,    "EfficientNet_B0_Weights.IMAGENET1K_V1"),
# }

# -------------- Helpers --------------
def resolve_weights(weights_enum_path: str):
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)

def safe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def infer_default_engine() -> str:
    # ARM/Raspberry Pi → qnnpack, x86 → fbgemm
    arch = platform.machine().lower()
    if "arm" in arch or "aarch64" in arch:
        return "qnnpack"
    return "fbgemm"

def extract_preprocess_from_weights(weights) -> dict:
    # Defaults (ImageNet)
    meta = {
        "resize_shorter_side": 256,
        "center_crop": 224,
        "normalize_mean": weights.meta.get("mean", [0.485, 0.456, 0.406]),
        "normalize_std":  weights.meta.get("std",  [0.229, 0.224, 0.225]),
        "categories": weights.meta.get("categories", []),
    }
    # Best effort: Transform-Pipeline inspizieren (v1/v2 unterschiedlich)
    tfm = weights.transforms()
    def maybe_update(obj):
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            meta["resize_shorter_side"] = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            meta["center_crop"] = int(getattr(obj, "size"))
        elif hasattr(obj, "mean") and hasattr(obj, "std"):
            try:
                meta["normalize_mean"] = [float(x) for x in obj.mean]
                meta["normalize_std"]  = [float(x) for x in obj.std]
            except Exception:
                pass
    if hasattr(tfm, "transforms") and isinstance(getattr(tfm, "transforms"), (list, tuple)):
        for t in tfm.transforms: maybe_update(t)
    else:
        inner = getattr(tfm, "_transforms", None)
        if isinstance(inner, (list, tuple)):
            for t in inner: maybe_update(t)
    return meta

def save_metadata(out_dir: Path, model_name: str, weights_path: str, preprocess: dict, quant: dict) -> None:
    name = model_name +"_dynamic_int8"
    meta = {
        "architecture": name,
        "source": "torchvision",
        "weights": weights_path,
        "image_size": preprocess.get("center_crop", 224),
        **preprocess,
        "quantization": quant,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def ensure_fp32_files(modelregistry, model_name: str, out_dir: Path, force_download: bool = False) -> Tuple[Path, Path]:
    """
    Stellt sicher, dass FP32 state_dict + metadata vorhanden sind.
    Lädt sie ggf. herunter.
    Returns: (state_dict_path, metadata_path)
    """
    sd_path   = out_dir / f"{model_name}_state_dict.pt"
    meta_path = out_dir / "metadata.json"

    if sd_path.exists() and meta_path.exists() and not force_download:
        logging.info("FP32 state_dict und metadata.json vorhanden – kein Download nötig.")
        return sd_path, meta_path

    logging.info("Loading FP-32 Weight from torchvision …")
    ctor, weights_path,_,_ = modelregistry[model_name]
    weights = resolve_weights(weights_path)
    model = ctor(weights=weights)
    model.eval()

    # state_dict speichern
    torch.save(model.state_dict(), sd_path)
    logging.info(f"Saved: {sd_path}")

    # Preprocess/Labels extrahieren und Metadata schreiben (ohne quantization)
    preprocess = extract_preprocess_from_weights(weights)
    save_metadata(out_dir, model_name, weights_path, preprocess, quant={"type": "none"})
    logging.info(f"Saved: {meta_path}")

    return sd_path, meta_path

# -------------- Core: Dynamic Quantization --------------
def dynamic_quantize(ctx: Context) -> Path:
    """
    Lädt FP32 state_dict, quantisiert dynamisch {nn.Linear} → int8,
    traced TorchScript speichern. Gibt Pfad zum quantisierten TS zurück.
    """
    # Engine setzen

    args = ctx.args
    model_name = args.model
    path = args.modeldir
    mod_reg = ctx.MODEL_REGISTRY
    logger = ctx.logger

    if path is None:
        path = "../modelzoo"

    out_dir = Path(path).expanduser().resolve()
    #out_dir = Path("../opt/models/mobilenet_v2_static").expanduser().resolve()
    out_dir = out_dir / f"{model_name}_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"
    engine= torch.backends.quantized.engine

    logging.info(f"Quantization engine: {engine}")

    # 1) FP32 state_dict sicherstellen/laden
    sd_path, meta_path = ensure_fp32_files(mod_reg, model_name, out_dir, force_download=False)

    # 2) Architektur + Gewichte laden
    ctor, weights_path,_,_  = mod_reg[model_name]
    weights = resolve_weights(weights_path)
    categories=weights.meta.get("categories", [])
    model = ctor(weights=None)
    model.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
    model.eval()

    # 3) Dynamische Quantisierung (nur Linear-Schichten)
    from torch.ao.quantization import quantize_dynamic
    #LSTM wurde raus genommen
    q_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()


    preprocess = weights.transforms()
    trace_size = 224 
    if hasattr(preprocess, 'crop_size'):
         trace_size = preprocess.crop_size[0]
    elif hasattr(preprocess, 'resize_size'):
         trace_size = preprocess.resize_size[0]

    logger.info(f"Using Image Size: {trace_size}")

    # 4) TorchScript erzeugen (Inputgröße: trace_size)
    example = torch.randn(1, 3, trace_size, trace_size)
    ts = torch.jit.trace(q_model, example)
    ts_path = out_dir / f"{model_name}_dynamic_int8_ts.pt"
    ts.save(str(ts_path))
    logging.info(f"Quantized TorchScript saved: {ts_path}")

    # 5) Metadata aktualisieren (quantization block)

    meta = {
        "architecture": model_name+"_dynamic_int8",           
        "source": "torchvision",
        "weights": weights_path,
        "image_size": trace_size,
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
            "type": "dynamic",
            "dtype": "int8",
            "layers": "all",
            "engine": torch.backends.quantized.engine,
            "artifact": "torchscript",
        }
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # meta["quantization"] = {
    #     "type": "dynamic",
    #     "dtype": "qint8",
    #     "layers": "Linear",
    #     "engine": engine,
    #     "artifact": ts_path.name,
    # }
    # meta["image_size"] = trace_size  # falls trace_size != center_crop
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info(f"Metadata aktualisiert: {meta_path}")

    return ts_path

