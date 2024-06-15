import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.heavyball_curvature import HeavyBallCurvature  # Assuming this is your custom optimizer

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
    model_heavyball_curvature = SimpleNN().to(device)
    optimizer_heavyball_curvature = HeavyBallCurvature(model_heavyball_curvature.parameters(), lr=1e-3,
                                                       momentum=momentum,
                                                       epsilon=0.01)



    # Train and test the model
    train(model_heavyball_curvature, train_loader, criterion, optimizer_heavyball_curvature, device, num_epochs=2)
    test(model_heavyball_curvature, test_loader, criterion, device)


print("\nFinished varying momentum values")
