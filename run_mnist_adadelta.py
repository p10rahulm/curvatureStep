import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.simplesgd_curvature import SimpleSGDCurvature
from optimizers.adam import Adam
from optimizers.simplesgd import SimpleSGD
from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature
from optimizers.nag import NAG
from optimizers.adagrad import Adagrad
from optimizers.adadelta import Adadelta
from optimizers.adadelta_curvature import AdadeltaCurvature

from optimizers.rmsprop_with_momentum import RMSPropMomentum
from optimizers.rmsprop import RMSProp
from optimizers.adamw import AdamW
from optimizers.nadamw import NAdamW
from optimizers.nadam import NAdam
from optimizers.amsgrad import AMSGrad
from optimizers.shampoo import Shampoo

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
    model_adadelta = SimpleNN().to(device)
    optimizer_adadelta = Adadelta(model_adadelta.parameters(), lr=1.0, rho=rho, eps=1e-6)
    train(model_adadelta, train_loader, criterion, optimizer_adadelta, device, num_epochs=2)
    test(model_adadelta, test_loader, criterion, device)

