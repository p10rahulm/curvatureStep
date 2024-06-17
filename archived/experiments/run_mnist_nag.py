import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed

from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature

# Set random seeds for reproducibility
set_seed(42)
# set_seed(8)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, test_loader = load_mnist()

# Initialize the models, loss function, and optimizers
criterion = torch.nn.CrossEntropyLoss()

# Vary momentum from 0.0 to 1.0 in steps of 0.05
momentum_values = [round(x * 0.05, 2) for x in range(21)]

for momentum in momentum_values:
    print(f"\nTraining with Nesterov Accelerated Gradient (NAG) with Curvature Optimizer with momentum={momentum}")
    model_nag_curvature = SimpleNN().to(device)
    optimizer_nag_curvature = NAGCurvature(model_nag_curvature.parameters(), lr=1e-3, momentum=momentum, epsilon=0.01)
    train(model_nag_curvature, train_loader, criterion, optimizer_nag_curvature, device, num_epochs=2)
    test(model_nag_curvature, test_loader, criterion, device)

set_seed(42)

for momentum in momentum_values:
    print(f"\nTraining with Nesterov Accelerated Gradient (NAG) with momentum={momentum}")
    model_nag = SimpleNN().to(device)
    optimizer_nag = NAG(model_nag.parameters(), lr=1e-3, momentum=momentum)
    train(model_nag, train_loader, criterion, optimizer_nag, device, num_epochs=2)
    test(model_nag, test_loader, criterion, device)