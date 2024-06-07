import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from optimizer import CustomOptimizer
from optim_adam import CustomAdam
from optim_simplesgd import SimpleSGD
from optim_heavyball import SGDWithMomentum
from tqdm import tqdm
import numpy as np
import random


# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set your desired seed here

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True) # loss moved from 81.92 to 40.97

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the models, loss function, and optimizers
model_custom = SimpleNN().to(device)
model_adam = SimpleNN().to(device)
model_sgd = SimpleNN().to(device)
model_momentum = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer_custom = CustomOptimizer(model_custom.parameters(), lr=1e-3, momentum_mult=0.9, epsilon=0.01)
optimizer_adam = CustomAdam(model_adam.parameters(), lr=1e-3)
optimizer_sgd = SimpleSGD(model_sgd.parameters(), lr=1e-3)
optimizer_momentum = SGDWithMomentum(model_momentum.parameters(), lr=1e-3, momentum=0.9)

# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {running_loss/len(train_loader):.4f}")

# Testing loop
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")


# Train and test the models
print("Training with Custom Optimizer")
train(model_custom, train_loader, criterion, optimizer_custom, num_epochs=2)
test(model_custom, test_loader, criterion)

print("\nTraining with Custom Adam Optimizer")
train(model_adam, train_loader, criterion, optimizer_adam, num_epochs=2)
test(model_adam, test_loader, criterion)

print("\nTraining with Simple SGD Optimizer")
train(model_sgd, train_loader, criterion, optimizer_sgd, num_epochs=2)
test(model_sgd, test_loader, criterion)

print("\nTraining with SGD with Heavy Ball Momentum Optimizer")
train(model_momentum, train_loader, criterion, optimizer_momentum, num_epochs=2)
test(model_momentum, test_loader, criterion)
