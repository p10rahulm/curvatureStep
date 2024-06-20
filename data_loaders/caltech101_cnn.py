import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_caltech101(batch_size=64):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Ensure all images have 3 channels
        transforms.Resize((96, 96)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Use ImageNet mean and std
    ])

    train_dataset = torchvision.datasets.Caltech101(root='./data', download=True, transform=transform)
    test_dataset = torchvision.datasets.Caltech101(root='./data', download=True, transform=transform)  # No official split, use the same for simplicity
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
