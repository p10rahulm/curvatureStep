import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_flowers102(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 as ResNet requires this input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Use ImageNet mean and std
    ])

    train_dataset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform)
    test_dataset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
