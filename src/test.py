# quantize_mobilenetv3.py
import torch
import torchvision.models.quantization as quant_models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os

def create_calibration_data_loader(num_samples=100):
    """
    Create a simple calibration data loader using CIFAR-10 as an example.
    In practice, you should use ImageNet data for better results.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Use CIFAR-10 as example calibration data
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    
    # Create a subset for calibration
    indices = torch.randperm(len(dataset))[:num_samples]
    calib_subset = Subset(dataset, indices)
    
    calib_loader = DataLoader(calib_subset, batch_size=16, shuffle=False)
    print(f"Created calibration loader with {len(calib_subset)} images")
    return calib_loader

def quantize_mobilenetv3_large():
    """Quantize MobileNetV3 Large and save it"""
    print("Loading pre-trained MobileNetV3 Large...")
    
    # Load the quantizable version of the model
    model = quant_models.mobilenet_v3_large(weights='DEFAULT', quantize=False)
    model.eval()
    
    print("Creating calibration data...")
    calib_loader = create_calibration_data_loader(num_samples=100)
    
    print("Setting up quantization...")
    torch.backends.quantized.engine = 'qnnpack'
    
    # Fuse model (important for quantization)
    model.fuse_model()
    
    # Set quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare for calibration
    torch.quantization.prepare(model, inplace=True)
    
    print("Calibrating model...")
    with torch.no_grad():
        for data, _ in calib_loader:
            model(data)
    
    print("Converting to quantized model...")
    torch.quantization.convert(model, inplace=True)
    
    # Create directory if it doesn't exist
    os.makedirs('quantized_models', exist_ok=True)
    
    # Save the quantized model
    model_path = 'quantized_models/mobilenet_v3_large_quantized.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Quantized model saved to: {model_path}")
    
    # Also save as scripted model for easier deployment
    scripted_model = torch.jit.script(model)
    scripted_path = 'quantized_models/mobilenet_v3_large_quantized_scripted.pth'
    scripted_model.save(scripted_path)
    print(f"Scripted quantized model saved to: {scripted_path}")
    
    return model, scripted_path

def load_quantized_model(model_path):
    """Load a quantized model"""
    print(f"Loading quantized model from {model_path}...")
    
    # Set the quantization engine
    torch.backends.quantized.engine = 'qnnpack'
    
    if model_path.endswith('_scripted.pth'):
        # Load scripted model
        model = torch.jit.load(model_path)
    else:
        # Load state dict and create model architecture
        model = quant_models.mobilenet_v3_large(weights=None, quantize=False)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    
    print("Model loaded successfully!")
    return model

def run_inference(model, image_path):
    """Run inference on an image"""
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_batch)
    
    # Get predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Simple class names for demonstration
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    print("\nTop 5 Predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{i+1}. {class_names[top5_catid[i]]}: {top5_prob[i].item() * 100:.2f}%")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize and use MobileNetV3 Large')
    parser.add_argument('--quantize', action='store_true', help='Quantize the model')
    parser.add_argument('--inference', action='store_true', help='Run inference')
    parser.add_argument('--image', type=str,default="../data/cat.jpg", help='Path to image for inference')
    parser.add_argument('--model', type=str, default='quantized_models/mobilenet_v3_large_quantized_scripted.pth', 
                       help='Path to quantized model')
    
    args = parser.parse_args()
    
    if args.quantize:
        # Quantize the model
        quantize_mobilenetv3_large()
        print("Quantization complete!")
    
    if args.inference:
        if not args.image:
            print("Please provide an image path with --image")
            return
        
        # Load quantized model
        model = load_quantized_model(args.model)
        
        # Run inference
        run_inference(model, args.image)

if __name__ == "__main__":
    main()