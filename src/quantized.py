# src/quantize.py
import torch
import torch.quantization
import os
from modelloader import get_model  # Correct import name
from dataloader import get_calibration_loader  # Correct import name

def quantize_and_save_model_static(model_name, model, calibration_data_loader):
    """
    Applies static quantization to a model and saves it to the correct folder in the model zoo.
    """
    # 0. Inform the user which model is being processed
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")

    # 1. Set the engine for mobile (ARM CPU)
    torch.backends.quantized.engine = 'qnnpack'
    print("Quantization backend set to 'qnnpack' for ARM CPUs.")

    # 2. Set the model to eval mode and fuse
    model.eval()
    print("Model set to evaluation mode.")
    
    # Check if the model has fusion support before trying to fuse
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
        print("Model layers fused.")
    else:
        print(f"Warning: Model {model_name} does not have a fuse_model method. Proceeding without fusion.")

    # 3. Specify quantization config
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print("Quantization configuration set.")

    # 4. Prepare for calibration
    torch.quantization.prepare(model, inplace=True)
    print("Model prepared for calibration.")

    # 5. Calibrate
    print(f"Calibrating {model_name} with {len(calibration_data_loader.dataset)} images...")
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_data_loader):
            if i % 10 == 0:  # Print progress every 10 batches
                print(f"  Calibration batch {i}/{len(calibration_data_loader)}")
            model(data)
    print("Calibration complete.")

    # 6. Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    print("Model converted to quantized INT8.")

    # 7. Save the model's state dictionary
    save_dir = f"models/{model_name}/int8"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}_int8_static.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model state dict saved to: {save_path}")

    # # 8. Save a scripted version for deployment
    # print("Scripting model for deployment...")
    # try:
    #     scripted_model = torch.jit.script(model)
    #     scripted_save_path = os.path.join(save_dir, f"{model_name}_int8_scripted.pth")
    #     scripted_model.save(scripted_save_path)
    #     print(f"Scripted model saved to: {scripted_save_path}")
    # except Exception as e:
    #     print(f"Warning: Could not script model {model_name}. Error: {e}")
    #     print("Proceeding with non-scripted model.")

    # print(f"Finished processing {model_name}!")
    # return model

def run(modelnames,first:bool):
    """Main function to run the quantization pipeline for multiple models."""
    # List of models to quantize
    if first:
        models_to_quantize = modelnames[0]
    else:
        models_to_quantize = modelnames

    # Start with one, add more later
    
    # Configuration
    batch_size = 32
    num_calibration_samples = 1000  # Use a subset for calibration

    for model_name in models_to_quantize:
        try:
            print(f"\nLoading pre-trained {model_name}...")
            # Load the FP32 model - don't save FP32 here, that should be in model_loader.py
            model = get_model(model_name, pretrained=True,quantized=False)
            
            print("Preparing calibration data...")
            # Get the calibration data loader
            calib_loader = get_calibration_loader(
                batch_size=batch_size, 
                num_samples=num_calibration_samples
            )
            
            # Quantize AND SAVE the model
            quantized_model = quantize_and_save_model_static(model_name, model, calib_loader)
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run(["mobilenet_v2"], first=True)