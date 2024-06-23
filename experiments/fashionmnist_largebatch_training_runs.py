# fashionmnist_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys
import torch
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from data_loaders.fashion_mnist import load_fashion_mnist
from models.simpleCNN_template import SimpleCNN
import torch.nn as nn
from train import train
from test import test

results = []

total_epochs = 50
total_runs = 2

print("#", "-" * 100)
print(f"# Running {total_epochs} epochs of training - {total_runs} runs")
print("#", "-" * 100)


for optimizer_class, default_params in optimizers:
    print(f"\nRunning FashionMNIST training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()
    
    # Set device to GPU 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # dataset_loader = load_fashion_mnist
    # model_class = SimpleCNNFashionMNIST
    def load_fmnist(batch_size=512):
        return load_fashion_mnist(batch_size=batch_size)

    dataset_loader = load_fmnist
    model_class = SimpleCNN

    # Hyperparameters
    model_hyperparams = {
        'num_classes': 10,
        'image_width': 28, 
        'num_channels': 1
    }
    
    loss_criterion = nn.CrossEntropyLoss
    trainer_function = train
    test_function = test

    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=dataset_loader,
        model_class=model_class,
        num_runs=total_runs,
        num_epochs=total_epochs,
        debug_logs=True,
        model_hyperparams=model_hyperparams,
        loss_criterion=loss_criterion,
        device=device,
        trainer_fn=trainer_function,
        tester_fn=test_function,
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

# write_to_file('outputs/fashionmnist_training_logs.csv', results)
