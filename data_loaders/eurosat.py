import os
import ssl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

def load_eurosat(batch_size=64, data_dir='./data/eurosat', test_split=0.1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 as ResNet requires this input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Use ImageNet mean and std
    ])

    # Create the data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download and load the entire dataset
    full_dataset = datasets.EuroSAT(root=data_dir, download=True, transform=transform)

    # Calculate sizes for train and test splits
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size

    # Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
