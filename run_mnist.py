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



# Train and test the models
# print("Training with Simple SGD with Curvature Optimizer")
# model_sgd_curvature = SimpleNN().to(device)
# optimizer_sgd_curvature = SimpleSGDCurvature(model_sgd_curvature.parameters(), lr=1e-3, momentum_mult=0.9, epsilon=0.01)
# train(model_sgd_curvature, train_loader, criterion, optimizer_sgd_curvature, device, num_epochs=2)
# test(model_sgd_curvature, test_loader, criterion, device)
#
# print("\nTraining with Simple SGD Optimizer")
# model_sgd = SimpleNN().to(device)
# optimizer_sgd = SimpleSGD(model_sgd.parameters(), lr=1e-3)
# train(model_sgd, train_loader, criterion, optimizer_sgd, device, num_epochs=2)
# test(model_sgd, test_loader, criterion, device)
#
# print("\nTraining with Adam Optimizer")
# model_adam = SimpleNN().to(device)
# optimizer_adam = Adam(model_adam.parameters(), lr=1e-3)
# train(model_adam, train_loader, criterion, optimizer_adam, device, num_epochs=2)
# test(model_adam, test_loader, criterion, device)
#
# print("\nTraining with SGD with Heavy Ball Momentum and Curvature Optimizer")
# model_heavyball_curvature = SimpleNN().to(device)
# optimizer_heavyball_curvature = HeavyBallCurvature(model_heavyball_curvature.parameters(), lr=1e-3, momentum=0.1, epsilon=0.01)
# train(model_heavyball_curvature, train_loader, criterion, optimizer_heavyball_curvature, device, num_epochs=2)
# test(model_heavyball_curvature, test_loader, criterion, device)
#
# print("\nTraining with SGD with Heavy Ball Momentum Optimizer")
# model_heavyball = SimpleNN().to(device)
# optimizer_heavyball = HeavyBall(model_heavyball.parameters(), lr=1e-3, momentum=0.9)
# train(model_heavyball, train_loader, criterion, optimizer_heavyball, device, num_epochs=2)
# test(model_heavyball, test_loader, criterion, device)
#
# print("Training with Nesterov Accelerated Gradient (NAG) Optimizer")
# model_nag = SimpleNN().to(device)
# optimizer_nag = NAG(model_nag.parameters(), lr=1e-3, momentum=0.9)
# train(model_nag, train_loader, criterion, optimizer_nag, device, num_epochs=2)
# test(model_nag, test_loader, criterion, device)
#
# print("\nTraining with Adagrad Optimizer")
# model_adagrad = SimpleNN().to(device)
# optimizer_adagrad = Adagrad(model_adagrad.parameters(), lr=1e-2, eps=1e-10)
# train(model_adagrad, train_loader, criterion, optimizer_adagrad, device, num_epochs=2)
# test(model_adagrad, test_loader, criterion, device)
#
# print("\nTraining with Adadelta Optimizer")
# model_adadelta = SimpleNN().to(device)
# optimizer_adadelta = Adadelta(model_adadelta.parameters(), lr=1.0, rho=0.9, eps=1e-6)
# train(model_adadelta, train_loader, criterion, optimizer_adadelta, device, num_epochs=2)
# test(model_adadelta, test_loader, criterion, device)
#
# print("\nTraining with RMSProp Optimizer")
# model_rmsprop = SimpleNN().to(device)
# optimizer_rmsprop = RMSProp(model_rmsprop.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0)
# train(model_rmsprop, train_loader, criterion, optimizer_rmsprop, device, num_epochs=2)
# test(model_rmsprop, test_loader, criterion, device)

# print("\nTraining with RMSProp Optimizer With Momentum")
# model_rmsprop_momentum = SimpleNN().to(device)
# optimizer_rmsprop_momentum = RMSPropMomentum(model_rmsprop_momentum.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.1)
# train(model_rmsprop_momentum, train_loader, criterion, optimizer_rmsprop_momentum, device, num_epochs=2)
# test(model_rmsprop_momentum, test_loader, criterion, device)

# print("\nTraining with AdamW Optimizer")
# model_adamw = SimpleNN().to(device)
# optimizer_adamw = AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.01)
# train(model_adamw, train_loader, criterion, optimizer_adamw, device, num_epochs=2)
# test(model_adamw, test_loader, criterion, device)

# print("\nTraining with NAdamW Optimizer")
# model_nadamw = SimpleNN().to(device)
# optimizer_nadamw = NAdamW(model_nadamw.parameters(), lr=1e-3, weight_decay=0.01)
# train(model_nadamw, train_loader, criterion, optimizer_nadamw, device, num_epochs=2)
# test(model_nadamw, test_loader, criterion, device)

# print("\nTraining with NAdam Optimizer")
# model_nadam = SimpleNN().to(device)
# optimizer_nadam = NAdam(model_nadam.parameters(), lr=1e-3)
# train(model_nadam, train_loader, criterion, optimizer_nadam, device, num_epochs=2)
# test(model_nadam, test_loader, criterion, device)