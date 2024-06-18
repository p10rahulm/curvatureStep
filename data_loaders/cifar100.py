import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_cifar100(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224 as ResNet requires this input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Use ImageNet mean and std
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
