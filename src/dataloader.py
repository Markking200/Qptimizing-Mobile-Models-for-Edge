import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import kagglehub

# Download latest version


def get_calibration_loader(batch_size=32,
                           num_samples=1000):
    """
    Creates a DataLoader for calibration.
    Args:
        dataset_path: Path to the training data.
        batch_size: Batch size for calibration.
        num_samples: The total number of images to use for calibration.
    """
    #path = kagglehub.dataset_download("imagenetmini-1000")
    path="/home/marceldavis/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1"

    # Define the preprocessing (must match what the model was trained with)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset from the folder
    dataset = datasets.ImageFolder(root=path, transform=preprocess)

    # This creates random indices to select 'num_samples' images.
    indices = torch.randperm(len(dataset))[:num_samples]
    calib_subset = Subset(dataset, indices)

    # Create the DataLoader - no shuffle
    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for calibration
        num_workers=2,  # Parallel data loading (speed up) #ToDo for later OP
        pin_memory=True # Faster data transfer to GPU if you were using one
    )

    print(f"Created calibration loader with {len(calib_subset)} images.")
    return calib_loader