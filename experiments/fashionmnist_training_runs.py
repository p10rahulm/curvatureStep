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
from models.simpleCNN_fashion import SimpleCNNFashionMNIST

results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)

dataset_loader = load_fashion_mnist
model_class = SimpleCNNFashionMNIST

for optimizer_class, default_params in optimizers:
    print(f"\nRunning FashionMNIST Training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()
    
    # Set device to GPU 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=dataset_loader,
        model_class=model_class,
        num_runs=10,
        num_epochs=10,
        debug_logs=True,
        device=device
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/fashionmnist_training_logs.csv', results)