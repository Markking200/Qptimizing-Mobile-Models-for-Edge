import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch
import os 
import torchvision.models.quantization as quantization_models

def get_model(model_name: str, pretrained: bool = True, save_fp32:bool =False,quantized:bool=False):
    """
    Factory function to get the requested model.
    """
    model_name = model_name.lower()
    weights = "DEFAULT"
    try:
        if quantized:
            # Use quantized model architectures
            if model_name == "mobilenet_v2":
                returnmodel = quantization_models.mobilenet_v2(weights=weights, quantize=False)
            elif model_name == "mobilenet_v3_large":
                returnmodel = quantization_models.mobilenet_v3_large(weights=weights, quantize=False)
            elif model_name == "mobilenet_v3_small":
                returnmodel = quantization_models.mobilenet_v3_small(weights=weights, quantize=False)
            else:
                raise ValueError(f"Quantized version not supported for {model_name}")
        else:
            # Use normal model architectures
            if model_name == "mobilenet_v2":
                returnmodel = models.mobilenet_v2(weights=weights)
            elif model_name == "mobilenet_v3_large":
                returnmodel = models.mobilenet_v3_large(weights=weights)
            elif model_name == "mobilenet_v3_small":
                returnmodel = models.mobilenet_v3_small(weights=weights)
            elif model_name == "efficientnet_b0":
                if pretrained:
                    returnmodel = EfficientNet.from_pretrained('efficientnet-b0')
                else:
                    returnmodel = EfficientNet.from_name('efficientnet-b0')
            else:
                raise ValueError(f"Model {model_name} not supported.")
    except :
        raise Exception("Error")
    
    if pretrained and save_fp32:
        try:
            # Create the FP32 directory
            save_dir = f"models/{model_name}/fp32"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the state_dict
            save_path = os.path.join(save_dir, f"{model_name}_fp32.pth")
            torch.save(returnmodel.state_dict(), save_path)
            print(f"FP32 model state dict saved to: {save_path}")
        except:
            raise Exception
    
    return returnmodel

#Save the model V this will be needed when benchmarking on edge devices
#torch.backends.quantized.engine = 'qnnpack'
model = get_model("mobilenet_v2", pretrained=True)
torch.save(model.state_dict(), "models/mobilenet_v2/fp32/mobilenet_v2_fp32.pth")