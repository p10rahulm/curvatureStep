import torch
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed
from optimizers.adam import Adam
from optimizers.adam_curvature import AdamCurvature

from optimizers.simplesgd import SimpleSGD
from optimizers.simplesgd_curvature import SimpleSGDCurvature

from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature

from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature

from optimizers.adagrad import Adagrad
from optimizers.adagrad_curvature import AdagradCurvature

from optimizers.adadelta import Adadelta
from optimizers.adadelta_curvature import AdadeltaCurvature

from optimizers.rmsprop import RMSProp
from optimizers.rmsprop_curvature import RMSPropCurvature

from optimizers.rmsprop_with_momentum import RMSPropMomentum
from optimizers.rmsprop_with_momentum_curvature import RMSPropMomentumCurvature


from optimizers.adamw import AdamW
from optimizers.adamw_curvature import AdamWCurvature

from optimizers.nadamw import NAdamW
from optimizers.nadam_curvature import NAdamCurvature

from optimizers.nadam import NAdam
from optimizers.nadamw_curvature import NAdamWCurvature

from optimizers.amsgrad import AMSGrad
from optimizers.amsgrad_curvature import AMSGradCurvature

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