import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.adam import Adam
from optimizers.adam_curvature import AdamCurvature

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
momentum_values = [round(x * 0.05, 2) for x in range(20)]

for beta1 in momentum_values:
    print(f"\nTraining with Adam with Curvature with beta1={beta1}")
    model_adam_curvature = SimpleNN().to(device)
    optimizer_adam_curvature = AdamCurvature(model_adam_curvature.parameters(), lr=1e-3, epsilon=0.01,
                                             betas=(beta1, 0.999))
    train(model_adam_curvature, train_loader, criterion, optimizer_adam_curvature, device, num_epochs=2)
    test(model_adam_curvature, test_loader, criterion, device)

set_seed(42)

for beta1 in momentum_values:
    print(f"\nTraining with Adam with beta1={beta1}")
    model_adam = SimpleNN().to(device)
    optimizer_adam = Adam(model_adam.parameters(), lr=1e-3,
                                             betas=(beta1, 0.999))
    train(model_adam, train_loader, criterion, optimizer_adam, device, num_epochs=2)
    test(model_adam, test_loader, criterion, device)