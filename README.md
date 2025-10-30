# Optimizing Mobilemodels for Edge environments

## Quantization

__Important Note: Working sourcecode can be found in 1_prot. The code is not clean, neither error free.__

### Usage
+ in __1_prot__ can be found the first prototype for quantizing the MobileNetV2
+ the file is named mobilenetv2.py
+ via args three different can be chosen between the static quantized model and the normal model
+ the file dynamicQuantization.py is responsible for the dynamic quanitzation
+ the complete .pt, the dictionary files and the meta.json are saved inside opt/models (if the folder doesnt exists yet, it will be automatically created by the code)
+ for the dynamic and espacially static quantization, a data folder with 200-1000 training and testing images should be available


## ToDo
+ finish the quantization of MobileNetV2 (with benchmarking)
+ implement QAT
+ expand to different mobilemodels
+ (find and implement other relevant Optimization techniques)