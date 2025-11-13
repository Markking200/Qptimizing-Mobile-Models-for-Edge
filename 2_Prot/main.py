import argparse
import logging
from pathlib import Path
import json
from mobilenetv2 import cmd_download
from mobilenetv2 import static_download



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

def recomStart(args: argparse.Namespace):

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

    with open("modelbenchmark_db.json","r", encoding="utf-8") as f:
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



def download(args: argparse.Namespace):
    if args.dynamic == True:
        cmd_download(args.modeldir)
    if args.static == True:
        static_download(args.modeldir)
    else:
        cmd_download(args.modeldir)

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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Description")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v=INFO, -vv=DEBUG")

    sub = p.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("recom", help="recommend")
    p_dl.add_argument("--inputfile",type=str,required=True,help="")
    p_dl.set_defaults(func=recomStart)

    p_inf = sub.add_parser("download", help="")
    # for now required = False because there is only one model
    p_inf.add_argument("--model", type=str, required=False, help="")
    p_inf.add_argument("--modeldir", type=str, required=True, help="")
    p_inf.add_argument("--static", type=bool, default=False, help="")
    p_inf.add_argument("--dynamic",type=bool,default=False, help="")
    p_inf.set_defaults(func=download)

    return p

def main():
    logger = setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    #setup_logging(args.verbose)
    args.func(args)

if __name__ == "__main__":
    main()