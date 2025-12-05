import argparse
import logging
from pathlib import Path
import json

from download_model import cmd_download
from dynamic_quant import dynamic_quantize
from static_quant import static_download
from qat import qat_download
from context import Context
from download_model import cmd_infer
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # FirstZoo/
sys.path.insert(0, str(ROOT))
from accuracy_benchmark.accuracy_benchmark import main as accracy_benchmark    

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Callable, List, Tuple, Dict
import textwrap

import torch
from torchvision import models, transforms



"""
    Recommendation System for Edge Devices.
    This system filters models based on user preferences and hardware specifications
    and provides the best model recommendation based on given priorities.

    Example Usage for recommendation:
    python main.py recom --input input.json

    Example Usage for dowload of mobilenetv2 static quantization:

    python main.py download --model mobilenet_v2 --modeldir ../opt/models --static True

    ###
    # conda activate mobilemlzoo
    ###

"""


TASK_CLASSIFICATION = "classification"
DS_IMAGENET1K = "ImageNet-1K"
DS_IMAGENET1K_SWAG = "ImageNet-1K (SWAG pretrain)"

MODEL_REGISTRY: Dict[str, Tuple[Callable[..., torch.nn.Module], str, str, str]] = {
    # AlexNet
    "alexnet": (models.alexnet, "AlexNet_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ConvNeXt
    "convnext_tiny":  (models.convnext_tiny,  "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_small": (models.convnext_small, "ConvNeXt_Small_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_base":  (models.convnext_base,  "ConvNeXt_Base_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "convnext_large": (models.convnext_large, "ConvNeXt_Large_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # DenseNet
    "densenet121": (models.densenet121, "DenseNet121_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet161": (models.densenet161, "DenseNet161_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet169": (models.densenet169, "DenseNet169_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "densenet201": (models.densenet201, "DenseNet201_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # EfficientNet
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b1": (models.efficientnet_b1, "EfficientNet_B1_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b2": (models.efficientnet_b2, "EfficientNet_B2_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b3": (models.efficientnet_b3, "EfficientNet_B3_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b4": (models.efficientnet_b4, "EfficientNet_B4_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b5": (models.efficientnet_b5, "EfficientNet_B5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b6": (models.efficientnet_b6, "EfficientNet_B6_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_b7": (models.efficientnet_b7, "EfficientNet_B7_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # EfficientNetV2
    "efficientnet_v2_s": (models.efficientnet_v2_s, "EfficientNet_V2_S_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_v2_m": (models.efficientnet_v2_m, "EfficientNet_V2_M_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "efficientnet_v2_l": (models.efficientnet_v2_l, "EfficientNet_V2_L_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # GoogLeNet / InceptionV3
    "googlenet":   (models.googlenet,   "GoogLeNet_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "inception_v3": (models.inception_v3, "Inception_V3_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MaxVit
    "maxvit_t": (models.maxvit_t, "MaxVit_T_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MNASNet
    "mnasnet0_5":  (models.mnasnet0_5,  "MNASNet0_5_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet0_75": (models.mnasnet0_75, "MNASNet0_75_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet1_0":  (models.mnasnet1_0,  "MNASNet1_0_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mnasnet1_3":  (models.mnasnet1_3,  "MNASNet1_3_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    # MobileNetV2 / V3
    "mobilenet_v2":       (models.mobilenet_v2,       "MobileNet_V2_Weights.IMAGENET1K_V1",       TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mobilenet_v3_small": (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # RegNet
    "regnet_x_400mf": (models.regnet_x_400mf, "RegNet_X_400MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_800mf": (models.regnet_x_800mf, "RegNet_X_800MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_1_6gf": (models.regnet_x_1_6gf, "RegNet_X_1_6GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_3_2gf": (models.regnet_x_3_2gf, "RegNet_X_3_2GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_8gf":   (models.regnet_x_8gf,   "RegNet_X_8GF_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_16gf":  (models.regnet_x_16gf,  "RegNet_X_16GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_x_32gf":  (models.regnet_x_32gf,  "RegNet_X_32GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    "regnet_y_400mf": (models.regnet_y_400mf, "RegNet_Y_400MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_800mf": (models.regnet_y_800mf, "RegNet_Y_800MF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_1_6gf": (models.regnet_y_1_6gf, "RegNet_Y_1_6GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_3_2gf": (models.regnet_y_3_2gf, "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_8gf":   (models.regnet_y_8gf,   "RegNet_Y_8GF_Weights.IMAGENET1K_V1",   TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_16gf":  (models.regnet_y_16gf,  "RegNet_Y_16GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "regnet_y_32gf":  (models.regnet_y_32gf,  "RegNet_Y_32GF_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),

    # Special case: SWAG -> ImageNet-1K fine-tune
    "regnet_y_128gf": (models.regnet_y_128gf, "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1", TASK_CLASSIFICATION, DS_IMAGENET1K_SWAG),

    # ResNet
    "resnet18":  (models.resnet18,  "ResNet18_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet34":  (models.resnet34,  "ResNet34_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet50":  (models.resnet50,  "ResNet50_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet101": (models.resnet101, "ResNet101_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnet152": (models.resnet152, "ResNet152_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ResNeXt
    "resnext50_32x4d":  (models.resnext50_32x4d,  "ResNeXt50_32X4D_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnext101_32x8d": (models.resnext101_32x8d, "ResNeXt101_32X8D_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "resnext101_64x4d": (models.resnext101_64x4d, "ResNeXt101_64X4D_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # ShuffleNetV2
    "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x1_5": (models.shufflenet_v2_x1_5, "ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "shufflenet_v2_x2_0": (models.shufflenet_v2_x2_0, "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # SqueezeNet
    "squeezenet1_0": (models.squeezenet1_0, "SqueezeNet1_0_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "squeezenet1_1": (models.squeezenet1_1, "SqueezeNet1_1_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # Swin Transformer (v1 + v2)
    "swin_t":    (models.swin_t,    "Swin_T_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_s":    (models.swin_s,    "Swin_S_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_b":    (models.swin_b,    "Swin_B_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_t": (models.swin_v2_t, "Swin_V2_T_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_s": (models.swin_v2_s, "Swin_V2_S_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "swin_v2_b": (models.swin_v2_b, "Swin_V2_B_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # VGG
    "vgg11":    (models.vgg11,    "VGG11_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg11_bn": (models.vgg11_bn, "VGG11_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg13":    (models.vgg13,    "VGG13_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg13_bn": (models.vgg13_bn, "VGG13_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg16":    (models.vgg16,    "VGG16_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg16_bn": (models.vgg16_bn, "VGG16_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg19":    (models.vgg19,    "VGG19_Weights.IMAGENET1K_V1",    TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vgg19_bn": (models.vgg19_bn, "VGG19_BN_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),

    # VisionTransformer (ViT)
    "vit_b_16": (models.vit_b_16, "ViT_B_16_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_b_32": (models.vit_b_32, "ViT_B_32_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_l_16": (models.vit_l_16, "ViT_L_16_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_l_32": (models.vit_l_32, "ViT_L_32_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
    "vit_h_14": (models.vit_h_14, "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1", TASK_CLASSIFICATION, DS_IMAGENET1K_SWAG),

    # Wide ResNet
    "wide_resnet50_2":  (models.wide_resnet50_2,  "Wide_ResNet50_2_Weights.IMAGENET1K_V1",  TASK_CLASSIFICATION, DS_IMAGENET1K),
    "wide_resnet101_2": (models.wide_resnet101_2, "Wide_ResNet101_2_Weights.IMAGENET1K_V1", TASK_CLASSIFICATION, DS_IMAGENET1K),
}

# MODEL_REGISTRY: Dict[str, Tuple[Callable[..., torch.nn.Module], str]] = {
#     "mobilenet_v2":        (models.mobilenet_v2,       "MobileNet_V2_Weights.IMAGENET1K_V1", "classification","ImageNet-1K"),
#     "mobilenet_v3_small":  (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
#     "mobilenet_v3_large":  (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),
#     "shufflenet_v2_x0_5":  (models.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
#     "shufflenet_v2_x1_0":  (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),
#     "mnasnet0_5":          (models.mnasnet0_5,         "MNASNet0_5_Weights.IMAGENET1K_V1"),
#     "mnasnet1_0":          (models.mnasnet1_0,         "MNASNet1_0_Weights.IMAGENET1K_V1"),
#     "squeezenet1_0":       (models.squeezenet1_0,      "SqueezeNet1_0_Weights.IMAGENET1K_V1"),
#     "efficientnet_b0":     (models.efficientnet_b0,    "EfficientNet_B0_Weights.IMAGENET1K_V1"),
# }


def recomStart(ctx: Context, args: argparse.Namespace):

    logger.info("Starting recommendation process based on user preferences.")
    logger.info("="*80)

    path = Path(args.inputfile).expanduser().resolve()

    # load json file
    logger.info(f"Loading input data from {path}...")
    with path.open("r", encoding="utf-8") as f:
        inputdata = json.load(f)

    logger.info("Input data loaded successfully.")
    logger.info("="*80)

    # print(type(inputdata))
    # for key,value in inputdata.items():
    #     print(f"{key} : {value}")
    modelPreference = inputdata["model"]["name"]
    quantizationPreference = inputdata["model"]["quantization"]
    device = inputdata["hardware"]["DEVICE_NAME"]
    cpu = inputdata["hardware"]["CPU"]["model"]
    cpuOnly = inputdata["hardware"]["CPU_ONLY"]
    architecture = inputdata["hardware"]["CPU"]["architecture"]
    precisionPref = inputdata["model"]["precision"]
    gpuPref = inputdata["hardware"]["GPU"]["model"]

    logger.info("Preferences extracted: ")
    logger.info(f"Model Preference: {modelPreference}")
    logger.info(f"Quantization Preference: {quantizationPreference}")
    logger.info(f"Device: {device}")
    logger.info(f"CPU Model: {cpu}, CPU Only: {cpuOnly}, Architecture: {architecture}")
    logger.info(f"Precision Preference: {precisionPref}, GPU Preference: {gpuPref}")
    logger.info("="*80)
  

    logger.info("Loading model benchmark database...")

    with open("utility_files/modelbenchmark_db.json","r", encoding="utf-8") as f:
        benchDB = json.load(f)

    logger.info(f"Loaded {len(benchDB)} models from benchmark database.")
    logger.info("="*80)

    def filter_by_precision(models, precision):
        logger.debug(f"Filtering models by precision: {precision}")
        return [
            model for model in models
            if (model["precision"].find(precision) != -1)]

    def filter_by_cpuOnly(models, cpuOnly):
        logger.debug(f"Filtering models by CPU only preference: {cpuOnly}")
        return [
            model for model in models 
            if model["hardware"]["cpu_only"] == cpuOnly]

    def filter_by_gpu(models, gpu_model):
        logger.debug(f"Filtering models by GPU preference: {gpu_model}")
        return [model for model in models if model["hardware"]["GPU"]["model"] == gpu_model or model["hardware"]["GPU"]["model"] == "None"]
    
    def filter_by_model(models, givenMod):
        logger.debug(f"Filtering models by model name: {givenMod}")
        return [
            model for model in models 
            if givenMod != "None" and model["model_name"] == givenMod]
    
    def filter_by_quantization (models,quant):
        logger.debug(f"Filtering models by quantization type: {quant}")
        return [
            model for model in models
            if model["quantization"]["type"]==quant
        ]

    filteredmodels = filter_by_cpuOnly(benchDB,cpuOnly)

    if precisionPref != "":
        filteredmodels = filter_by_precision(filteredmodels,precisionPref)

    if inputdata["model"]["filter"] != False:
        filteredmodels = filter_by_model(filteredmodels,modelPreference)

    if quantizationPreference != "":
        filteredmodels = filter_by_quantization(filteredmodels,quantizationPreference)

    if inputdata["hardware"]["GPU"]["filter"] != False:
        filteredmodels = filter_by_gpu(filteredmodels,gpuPref)

    accuracy_pref = inputdata["metrics"]["accuracy"]
    inference_speed_pref = inputdata["metrics"]["inference_speed"]
    stora_con_pref = inputdata["metrics"]["storage_consumption"]
    throughput_pref = inputdata["metrics"]["throughput"]

    logger.info(f"User preferences: Accuracy={accuracy_pref}, Inference Speed={inference_speed_pref}, Storage Consumption={stora_con_pref}, Throughput={throughput_pref}")
    logger.info("="*80)

    total = float(accuracy_pref) + float(inference_speed_pref) + float(stora_con_pref)
    acc_norm  = float(accuracy_pref) / total
    inf_norm = float(inference_speed_pref) / total
    stor_norm = float(stora_con_pref) / total
    throu_norm = float(throughput_pref) / total

    accuracies = [m["benchmark"]["accuracy_top1"]   for m in filteredmodels]
    latencies  = [m["benchmark"]["latency_ms_per_sample"]        for m in filteredmodels]
    sizes      = [m["artifact"]["disk_size_mb"]     for m in filteredmodels]
    throughputs = [m["benchmark"]["throughput_fps"]     for m in filteredmodels]

    acc_min, acc_max = min(accuracies), max(accuracies)
    lat_min, lat_max = min(latencies),  max(latencies)
    size_min, size_max = min(sizes),    max(sizes)
    throughput_min, throughput_max = min(throughputs), max(throughputs)

    logger.info("Normalizing model metrics...")
    logger.info("="*80)

    scores = []

    for mod in filteredmodels:

        model_throughput = mod["benchmark"]["throughput_fps"]
        model_latency = mod["benchmark"]["latency_ms_per_sample"]
        model_accuracy = mod["benchmark"]["accuracy_top1"]
        model_storage= mod["artifact"]["disk_size_mb"]

        if acc_max == acc_min:
            mod_norm_acc = 1.0
        else:
            mod_norm_acc = (model_accuracy - acc_min) / (acc_max - acc_min)

        if lat_max == lat_min:
            mod_norm_latency = 1.0
        else:
            mod_norm_latency = (lat_max - model_latency) / (lat_max - lat_min)

        if size_max == size_min:
            mod_norm_size = 1.0
        else:
            mod_norm_size = (size_max - model_storage) / (size_max - size_min)

        if acc_max == acc_min:
            mod_norm_throughput = 1.0
        else:
            mod_norm_throughput = (model_throughput - throughput_min) / (throughput_max - throughput_min)

        score = (
            mod_norm_acc     * acc_norm      +
            mod_norm_latency   * inf_norm  +
            mod_norm_size * stor_norm +
            mod_norm_throughput * throu_norm
        )

        scores.append({
            "model": mod["model_name"],
            "score": score,
        })

    scores_sorted = sorted(scores, key=lambda x: x['score'], reverse=True)

    logger.info(f"Top recommended models based on preferences:")

    counter = 1
    for entry in scores_sorted:
        logger.info(f"{counter}. | Model: {entry['model']} - Score: {entry['score']:.4f}")
        counter = counter +1

def inferece(ctx: Context, args: argparse.Namespace):
    logger.info("Starting inference process...")
    logger.info("="*80)
    cmd_infer(ctx) 

def download(ctx: Context, args: argparse.Namespace):
    #download for all models
    #ctx.logger.info(args.all)
    args = ctx.args
    if args.all:
        logger.info("Download requested for all models.")
        logger.info("="*80)
        for model in ctx.MODEL_REGISTRY.keys():
            logger.info(f"Download requested for model: {model}")
            ctx.args.model = model
            if args.dynamic == True:
                logger.info(f"Starting dynamic quantization download for model: {model}")
                logger.info("="*50)
                dynamic_quantize(ctx)
                logger.info("="*50)
            if args.static == True:
                logger.info(f"Starting static quantization download for model: {model}")
                logger.info("="*50)
                static_download(ctx)
                logger.info("="*50)
            if args.qat == True:
                logger.info(f"Starting QAT download for model: {model}")
                logger.info("="*50)
                qat_download(ctx)
                logger.info("="*50)
            if args.normal == True:
                logger.info(f"Starting normal download for model: {model}")
                logger.info("="*50)
                cmd_download(ctx)
                logger.info("="*50)

        return   
        
    ctx.logger.info(f"Download requested for model: {args.model}")
    #normal download for one model
    if args.dynamic == True:
        logger.info("="*50)
        logger.info(f"Starting dynamic quantization download for model: {args.model}")
        dynamic_quantize(ctx)
        logger.info("="*50)
    if args.static == True:
        logger.info("="*50)
        logger.info(f"Starting static quantization download for model: {args.model}")
        static_download(ctx)
        logger.info("="*50)
    if args.qat == True:
        logger.info("="*50)
        logger.info(f"Starting QAT download for model: {args.model}")
        qat_download(ctx)
        logger.info("="*50)
    if args.normal == True:
        logger.info("="*50)
        logger.info(f"Starting normal download for model: {args.model}")
        cmd_download(ctx)
        logger.info("="*50)

def get_models(ctx: Context, args: argparse.Namespace):
    #wenn --model und --all gesetzt ist dann abbrechen denn es darf nur eins gesetzt sein
    pass

def test(ctx: Context,args: argparse.Namespace):
    print(ctx.logger)
    print(ctx.MODEL_REGISTRY)
    print(ctx.args)

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            #logging.FileHandler('app.log') 
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def benchmark_accuracy(ctx: Context, args: argparse.Namespace):
    ctx.logger.info("Benchmarking accuracy...")
    ctx.logger.info("="*80)
    accracy_benchmark(ctx)


logger = setup_logging()

def build_parser(ctx:Context) -> argparse.ArgumentParser:
    epilog_text = textwrap.dedent(f"""{ctx.MODEL_REGISTRY}""")
    p = argparse.ArgumentParser(description="Description", epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v=INFO, -vv=DEBUG")

    sub = p.add_subparsers(dest="command", required=True)

    # -------------------- recommendation system --------------------
    p_dl = sub.add_parser("recom", help="recommend")
    p_dl.add_argument("--inputfile",type=str,required=True,help="")
    p_dl.set_defaults(func=recomStart)

    # -------------------- inference on image --------------------
    p_inf = sub.add_parser("infer", help="Führt Inferenz auf einem Bild aus")
    p_inf.add_argument("--model", type=str, required=True, help="")
    p_inf.add_argument("--dir", type=str, required=True, help="")
    #p_inf.add_argument("--image", type=str, required=True, help="Pfad zu einem Bild (jpg/png)")
    #p_inf.add_argument("--topk", type=int, default=5, help="Top-K Ergebnisse (default: 5)")
    #p_inf.add_argument("--cpu", action="store_true", help="Erzwinge CPU auch wenn CUDA verfügbar ist")
    #p_inf.add_argument("--force-state-dict", action="store_true", help="Ignoriere TorchScript und lade state_dict + Architektur")
    p_inf.set_defaults(func=inferece)


    # -------------------- download specified model --------------------
    p_inf = sub.add_parser("download", help="")
    # for now required = False because there is only one model
    p_inf.add_argument("--model", type=str, required=False, help="")
    p_inf.add_argument("--modeldir", type=str, required=False, help="")
    p_inf.add_argument("--all", type=str, required=False,default=False, help="")

    p_inf.add_argument("--normal", type=bool, default=False, help="")
    p_inf.add_argument("--static", type=bool, default=False, help="")
    p_inf.add_argument("--dynamic",type=bool,default=False, help="")
    p_inf.add_argument("--qat",type=bool,default=False, help="")
    p_inf.set_defaults(func=download)

    # -------------------- fetch all models from server (for edge device testing) --------------------
    p_inf = sub.add_parser("get", help="")
    # for now required = False because there is only one model
    p_inf.add_argument("--model", type=str, required=True, help="")
    p_inf.add_argument("--all", type=str, required=False,default=False, help="")
    p_inf.set_defaults(func=get_models)

    # -------------------- dev testing purposes (delete later) --------------------
    p_inf = sub.add_parser("test", help="")
    # for now required = False because there is only one model
    p_inf.add_argument("--model", type=str, required=False, help="")
    p_inf.add_argument("--all", type=str, required=False,default=False, help="")
    p_inf.set_defaults(func=test)

    # ---------------------------benchmark accuracy--------------------------------------
    p_inf = sub.add_parser("accbenchmark", help="")
    p_inf.add_argument("--modeldir", type=str, required=False, help="")
    p_inf.add_argument("--logpath", type=str, required=False, help="")
    p_inf.add_argument("--samplesize", type=int, required=False, help="")
    p_inf.add_argument("--all", type=bool, required=False,default=False, help="")
    p_inf.set_defaults(func=benchmark_accuracy)

    return p

def main():
    ctx = Context(logger= setup_logging(), registry= MODEL_REGISTRY)
    parser = build_parser(ctx)
    args = parser.parse_args()
    #setup_logging(args.verbose)
    ctx.args=args
    args.func(ctx, args)

if __name__ == "__main__":
    main()