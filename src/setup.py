import argparse
from quantized import run

def main():
    parser = argparse.ArgumentParser(description="Descrption of the Script!")
    parser.add_argument("--setup",type=bool,default=False,help="Genrate the Model Zoo.")
    parser.add_argument("--run_inference",type=bool,default=False,help="Run inference on given image.")
    parser.add_argument("--benchmark",type=bool,default=False,help="Benchmark the Models on the current device.")

    args= parser.parse_args()
    if args.setup:
        #this list can be extended in the future
        models=["mobilenet_v2","mobilenet_v3_large","mobilenet_v3_small","efficientnet_b0"]
        #will quantize all the models and save them in #ToDo
        run(models,True)


    elif args.run_inference:
        print("inference")
    elif args.benchmark:
        print("benchmark")
    else:
        raise Exception("Please choose one mode!")



if __name__ == '__main__':
    main()