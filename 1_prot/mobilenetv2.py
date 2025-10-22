#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MobileNet V2 CLI:
- download: Lädt offizielle torchvision-Gewichte, speichert state_dict, TorchScript und Metadata.
- infer:    Lädt TorchScript oder state_dict + Architektur und führt Offline-Inferenz auf Bilddateien aus.

Beispiel:
    Downlaod:
    python mobilenet_v2_cli.py download \
    --model mobilenet_v3_small \
    --out-dir ./opt/models/mobilenet_v3_small

    python ./src/mobilenetv2.py download --model alexnet --out-dir opt/models/alexnet

    Inference:
    python mobilenet_v2_cli.py infer \
    --model-dir ./opt/models/mobilenet_v3_small \
    --image ./example.jpg \
    --topk 5

    python src/mobilenetv2.py infer     --model-dir opt/models/alexnet  --image data/apple.jpg --topk 5

"""

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

# Jeder Eintrag: sichtbarer Name -> (ctor, weights_enum_attr)
# Die weights-Enums heißen in neueren torchvision-Versionen z.B. MobileNet_V3_Small_Weights
MODEL_REGISTRY: dict[str, tuple[Callable[..., torch.nn.Module], str]] = {
    "mobilenet_v2":        (models.mobilenet_v2,        "MobileNet_V2_Weights.IMAGENET1K_V1"),
    "mobilenet_v3_small":  (models.mobilenet_v3_small,  "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
    "mobilenet_v3_large":  (models.mobilenet_v3_large,  "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),
    "shufflenet_v2_x0_5":  (models.shufflenet_v2_x0_5,  "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
    "shufflenet_v2_x1_0":  (models.shufflenet_v2_x1_0,  "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),
    "mnasnet0_5":          (models.mnasnet0_5,          "MNASNet0_5_Weights.IMAGENET1K_V1"),
    "mnasnet1_0":          (models.mnasnet1_0,          "MNASNet1_0_Weights.IMAGENET1K_V1"),
    "squeezenet1_0":       (models.squeezenet1_0,       "SqueezeNet1_0_Weights.IMAGENET1K_V1"),
    "efficientnet_b0":     (models.efficientnet_b0,     "EfficientNet_B0_Weights.IMAGENET1K_V1"),
    "alexnet":              (models.alexnet,"AlexNet_Weights.IMAGENET1K_V1"),
}


# ---------- Logging ----------
def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level,
    )


# ---------- Download normal----------
def cmd_download(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name: str = getattr(args,"model", "mobilenet_v2")  # default
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unbekanntes Modell '{model_name}'. Unterstützt: {', '.join(MODEL_REGISTRY)}")

    ctor, weights_path = MODEL_REGISTRY[model_name]
    weights = resolve_weights(weights_path)

    logging.info(f"Lade {model_name} mit {weights_path} …")
    model = ctor(weights=weights)
    model.eval()

    categories = weights.meta.get("categories", [])
    tfm = weights.transforms()

    # Defaults
    resize_side = 256
    crop_size = 224
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std  = weights.meta.get("std",  [0.229, 0.224, 0.225])

    # introspektiere Transforms (wie in deiner Version)
    def _maybe_update_from_transform(obj):
        nonlocal resize_side, crop_size, mean, std
        name = obj.__class__.__name__.lower()
        if "resize" in name and hasattr(obj, "size"):
            s = getattr(obj, "size")
            resize_side = int(s[0] if isinstance(s, (list, tuple)) else s)
        elif "centercrop" in name and hasattr(obj, "size"):
            crop_size = int(getattr(obj, "size"))
        elif hasattr(obj, "mean") and hasattr(obj, "std"):
            try:
                mean = [float(x) for x in obj.mean]
                std  = [float(x) for x in obj.std]
            except Exception:
                pass

    if hasattr(tfm, "transforms") and isinstance(getattr(tfm, "transforms"), (list, tuple)):
        for t in tfm.transforms:
            _maybe_update_from_transform(t)
    else:
        inner = getattr(tfm, "_transforms", None)
        if isinstance(inner, (list, tuple)):
            for t in inner:
                _maybe_update_from_transform(t)

    # (A) state_dict speichern
    sd_path = out_dir / f"{model_name}_state_dict.pt"
    torch.save(model.state_dict(), sd_path)
    logging.info(f"state_dict gespeichert: {sd_path}")

    # (B) TorchScript speichern
    # Input-Größe meist 224; für EfficientNet_B0 auch ok; wenn ein Modell 299 o.ä. erfordert,
    # greifen oben die meta/transform-Werte. Wir tracen mit (1,3,crop_size,crop_size).
    example = torch.randn(1, 3, crop_size, crop_size)
    traced = torch.jit.trace(model, example)
    ts_path = out_dir / f"{model_name}_ts.pt"
    traced.save(str(ts_path))
    logging.info(f"TorchScript gespeichert: {ts_path}")

    meta = {
        "architecture": model_name,            # wichtig für infer()
        "source": "torchvision",
        "weights": weights_path,
        "image_size": crop_size,
        "resize_shorter_side": resize_side,
        "center_crop": crop_size,
        "normalize_mean": mean,
        "normalize_std": std,
        "categories": categories,
        "framework": {
            "torch": torch.__version__,
            "torchvision": getattr(models, "__version__", "unknown"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Download abgeschlossen. Modell ist offline nutzbar.")



def resolve_weights(weights_enum_path: str):
    """
    Nimmt z.B. 'MobileNet_V2_Weights.IMAGENET1K_V1' und gibt das Enum-Objekt zurück.
    """
    enum_name, member = weights_enum_path.split(".", 1)
    enum_obj = getattr(models, enum_name, None)
    if enum_obj is None:
        raise ValueError(f"Weights enum '{enum_name}' not found in torchvision.models")
    return getattr(enum_obj, member)

# ------------ Download dynamic ---------------------------

# ------------- Download static ---------------------------

def static_download(args: argparse.Namespace) -> None:
    #Prepare for download
    out_dir = Path(args.out_dir).expanduser().resolve()
    #out_dir = Path("../opt/models/mobilenet_v2_static").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name= "mobilenet_v2"

    backend = platform.machine().lower()
    torch.backends.quantized.engine = "qnnpack" if ("arm" in backend or "aarch64" in backend) else "fbgemm"

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model_fp32 = models.mobilenet_v2(weights=weights).eval()
    categories=weights.meta.get("categories", [])

    preprocess = weights.transforms()
    image_size = 224 

    qconfig = get_default_qconfig(torch.backends.quantized.engine)

    qconfig_dict = {"": qconfig}
    example_inputs = torch.randn(1, 3, image_size, image_size)
    prepared = quantize_fx.prepare_fx(model_fp32, qconfig_dict, example_inputs=example_inputs)

    def load_and_preprocess(path: Path):
        img = Image.open(path).convert("RGB")
        return preprocess(img).unsqueeze(0)  # (1,3,224,224)
    
    #calib_dir = Path("/home/University/BA/archive\(1\)//allimages")
    calib_dir = Path(os.path.abspath("../data/allimages"))  # lege dort 50–200 Bilder ab (beliebige natürliche Fotos)
    print(calib_dir)
    #print(list(calib_dir.glob("*.jpg")))
    try:
        with torch.inference_mode():
                #for i in range(200):
                imageindex = 0
                for fname in list(calib_dir.glob("*.jpg"))[:6500]:
                    #torch.rand(1,3,224, 224)
                    #print(fname)
                    x = load_and_preprocess(fname)
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
        "weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
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




# ---------- Preprocessing ----------
def build_preprocess_from_meta(meta: dict) -> transforms.Compose:
    # Rekonstruiert die Pipeline offline aus der metadata.json
    resize_side = int(meta.get("resize_shorter_side", 256))
    crop_size = int(meta.get("center_crop", 224))
    mean = meta.get("normalize_mean", [0.485, 0.456, 0.406])
    std = meta.get("normalize_std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(resize_side),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image(path: Path, preprocess: transforms.Compose) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # (1,3,H,W)
    return tensor


# ---------- Inference ----------
def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def cmd_infer(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser().resolve()
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    model_name = meta.get("architecture", "mobilenet_v2")
    # if model_name not in MODEL_REGISTRY:
    #     raise ValueError(f"Modell '{model_name}' aus metadata.json unbekannt. Unterstützt: {', '.join(MODEL_REGISTRY)}")

    preprocess = build_preprocess_from_meta(meta)
    categories = meta.get("categories", [])

    isquantized = False
    try:
            q=meta.get("quantization")
            if q != None:
                isquantized= True
    except Exception as e:
        pass 

    isquantized = True

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu and not isquantized) else "cpu")
    iscudaavailable= torch.cuda.is_available()
    #logging.info(f"{iscudaavailable}")
    logging.info(f"Gerät: {device}")
    if isquantized:
        torch.backends.quantized.engine = "fbgemm"

    #print("PyTorch:", torch.__version__)
    #print("CUDA built into PyTorch:", torch.version.cuda)
    #print("cuDNN:", torch.backends.cudnn.version())
    #print("CUDA available:", torch.cuda.is_available())

    ts_path = model_dir / f"{model_name}_ts.pt"
    sd_path = model_dir / f"{model_name}_state_dict.pt"

    if ts_path.exists() and not args.force_state_dict:
        logging.info(f"Lade TorchScript: {ts_path}")
        model = torch.jit.load(str(ts_path), map_location=device)
        print("Checkpoint 1")
        model.eval()
    else:
        logging.info(f"Lade state_dict + Architektur für {model_name}")
        ctor, _ = MODEL_REGISTRY[model_name]
        model = ctor(weights=None)
        model.load_state_dict(torch.load(sd_path, map_location="cpu"))
        model.eval()
        model.to(device)

    # Bild laden
    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")

    x = load_image(img_path, preprocess).to(device)

    print("Checkpoint")

    with torch.no_grad():
        logits = model(x)
        #print("Logits:\n")
        #print(logits)
        #print("\n-----------------------------------------------")
        probs = softmax_np(logits.cpu().numpy().squeeze())

    #print(logits.cpu().numpy().squeeze())
    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
    #print("Probabilites: pytorch function")
    #print(probabilities)
    #print("Probabilites: Self implemented")
    #print(probs)

    topk = int(args.topk)
    idxs = np.argsort(probs)[::-1][:topk]
    logging.info(f"Top-{topk} Ergebnisse:")

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Ausgabe
    # m= meta.get("architecture")
    # print(f"\nBild: {img_path}")
    # print(f"Modell: {m}")
    # print(f"Top-{topk} Klassen:")
    # for rank, i in enumerate(idxs, start=1):
    #     label = categories[i] if i < len(categories) else f"class_{i}"
    #     print(f"{rank:>2d}. {label:<30s} prob={probs[i]:.4f}")
        
    for i in range(top5_prob.size(0)):    
        print(f"{i+1:>2d}. {categories[top5_catid[i]]:<30s} prob: {top5_prob[i].item():.4f}")


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download & Inference (torchvision, ImageNet-1k)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v=INFO, -vv=DEBUG")

    sub = p.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Lädt Gewichte und speichert state_dict, TorchScript und Metadata")
    p_dl.add_argument("--model",type=str,required=True,help="Welches Model")
    p_dl.add_argument("--out-dir", type=str, required=True, help="Zielverzeichnis (wird erstellt, falls nicht vorhanden)")
    p_dl.set_defaults(func=cmd_download)

    p_inf = sub.add_parser("infer", help="Führt Inferenz auf einem Bild aus")
    p_inf.add_argument("--model-dir", type=str, required=True, help="Verzeichnis mit mobilenet_v2_{ts,state_dict}.pt und metadata.json")
    p_inf.add_argument("--image", type=str, required=True, help="Pfad zu einem Bild (jpg/png)")
    p_inf.add_argument("--topk", type=int, default=5, help="Top-K Ergebnisse (default: 5)")
    p_inf.add_argument("--cpu", action="store_true", help="Erzwinge CPU auch wenn CUDA verfügbar ist")
    p_inf.add_argument("--force-state-dict", action="store_true", help="Ignoriere TorchScript und lade state_dict + Architektur")
    p_inf.set_defaults(func=cmd_infer)

    p_dyn = sub.add_parser("dynamic")

    p_static= sub.add_parser("static")
    p_static.add_argument("--out-dir", type=str, required=True, help="Zielverzeichnis (wird erstellt, falls nicht vorhanden)")
    p_static.set_defaults(func=static_download)
    

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()