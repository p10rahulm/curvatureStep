import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.heavyball import HeavyBall

# Set random seeds for reproducibility
set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, test_loader = load_mnist()

# Initialize the models, loss function, and optimizers
model_heavyball_curvature = SimpleNN().to(device)

criterion = torch.nn.CrossEntropyLoss()

# Vary momentum from 0.0 to 1.0 in steps of 0.05
momentum_values = [round(x * 0.05, 2) for x in range(21)]

for momentum in momentum_values:
    print(f"\nTraining with Heavy Ball Curvature Optimizer with momentum={momentum}")
    # Reinitialize the model for each momentum value
    model_heavyball = SimpleNN().to(device)
    optimizer_heavyball = HeavyBall(model_heavyball.parameters(), lr=1e-3, momentum=momentum)



    # Train and test the model
    train(model_heavyball, train_loader, criterion, optimizer_heavyball, device, num_epochs=2)
    test(model_heavyball, test_loader, criterion, device)


print("\nFinished varying momentum values")
