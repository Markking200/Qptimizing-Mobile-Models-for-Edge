import torch
from torchvision import transforms
from PIL import Image
from modelloader import get_model  # Your function to get the model architecture
import torchvision.models as models
import torchvision.models.quantization as quantization_models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image",type=str,default="GoldenRet")

args = parser.parse_args()


# --- Configuration ---
model_path = 'models/mobilenet_v3_large/fp32/mobilenet_v3_large_fp32.pth'
image_path = f"../data/{args.image}.jpg"
label_path = '../data/imagenet_classes.txt'  # File with ImageNet class names
device = 'cpu'  # We're targeting CPU for edge deployment
# ---

# 1. Load the model architecture (remember to set the engine for quantized models!)
if 'int8' in model_path:
    torch.backends.quantized.engine = 'qnnpack'  # Crucial for quantized models!

model = get_model("mobilenet_v3_large", pretrained=False,quantized=False) 

model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))

model.eval()  # Set to evaluation mode!
model.to(device)

# 2. Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_image = Image.open(image_path)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch of size 1
input_batch = input_batch.to(device)

# 3. Run inference
with torch.no_grad():
    output = model(input_batch)

# 4. Process the output
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# 5. Read the labels and print the top prediction
with open(label_path, 'r') as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(f"{categories[top5_catid[i]]}: {top5_prob[i].item() * 100:.2f}%")