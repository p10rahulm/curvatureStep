import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.adadelta_curvature import AdadeltaCurvature

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

for rho in momentum_values:
    print(f"\nTraining with Adadelta Curvature Optimizer with rho={rho}")
    model_adadelta_curvature = SimpleNN().to(device)
    optimizer_adadelta_curvature = AdadeltaCurvature(model_adadelta_curvature.parameters(), lr=1.0, rho=rho, eps=1e-6, epsilon=0.01)
    train(model_adadelta_curvature, train_loader, criterion, optimizer_adadelta_curvature, device, num_epochs=2)
    test(model_adadelta_curvature, test_loader, criterion, device)

