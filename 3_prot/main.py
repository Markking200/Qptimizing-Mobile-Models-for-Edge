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

    python main.py download --modeldir ../opt/models/mobilenet_v2_static --static True

    ###
    # conda activate mobilemlzoo
    ###

"""

MODEL_REGISTRY: Dict[str, Tuple[Callable[..., torch.nn.Module], str]] = {
    "mobilenet_v2":        (models.mobilenet_v2,       "MobileNet_V2_Weights.IMAGENET1K_V1"),
    "mobilenet_v3_small":  (models.mobilenet_v3_small, "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),
    "mobilenet_v3_large":  (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),
    "shufflenet_v2_x0_5":  (models.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),
    "shufflenet_v2_x1_0":  (models.shufflenet_v2_x1_0, "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),
    "mnasnet0_5":          (models.mnasnet0_5,         "MNASNet0_5_Weights.IMAGENET1K_V1"),
    "mnasnet1_0":          (models.mnasnet1_0,         "MNASNet1_0_Weights.IMAGENET1K_V1"),
    "squeezenet1_0":       (models.squeezenet1_0,      "SqueezeNet1_0_Weights.IMAGENET1K_V1"),
    "efficientnet_b0":     (models.efficientnet_b0,    "EfficientNet_B0_Weights.IMAGENET1K_V1"),
}


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
    cmd_infer(ctx) 

def download(ctx: Context, args: argparse.Namespace):
    #download for all models
    if args.all == True:
        for model in ctx.MODEL_REGISTRY.keys():
            args.model = model
            print(f"Downloading model: {model}")
            if args.dynamic == True:
                dynamic_quantize(ctx)
            if args.static == True:
                static_download(ctx)
            if args.qat == True:
                qat_download(ctx)
            if args.normal == True:
                cmd_download(ctx)

    #normal download for one model
    if args.dynamic == True:
        dynamic_quantize(ctx)
    if args.static == True:
        static_download(ctx)
    if args.qat == True:
        qat_download(ctx)
    if args.normal == True:
        cmd_download(ctx)

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
    p_inf.add_argument("--model", type=str, required=True, help="")
    p_inf.add_argument("--modeldir", type=str, required=True, help="")
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