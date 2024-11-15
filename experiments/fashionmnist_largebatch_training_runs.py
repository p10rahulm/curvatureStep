# experiments/fashionmnist_largebatch_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import torch
import torch.nn as nn
from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleCNN_template import SimpleCNN
from data_loaders.fashion_mnist import load_fashion_mnist
from train import train
from test import test

def load_fmnist_large_batch(batch_size=512):
    """Wrapper function to load Fashion MNIST with large batch size"""
    return load_fashion_mnist(batch_size=batch_size)

def main():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

    print("#", "-"*100)
    print("# Running Fashion MNIST Large Batch Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'num_classes': 10,
        'image_width': 28,
        'num_channels': 1
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'fashionmnist_largebatch',
        'train_fn': train,
        'test_fn': test,
        'dataset_loader': load_fmnist_large_batch,  # Using the wrapper function
        'model_class': SimpleCNN,
        'num_runs': 2,           # 2 runs as per your original script
        'num_epochs': 50,        # 50 epochs as per your original script
        'model_hyperparams': model_hyperparams,
        'loss_criterion': nn.CrossEntropyLoss
    }

    # Run experiments for all optimizers
    train_df, test_df = run_all_experiments(
        optimizers=optimizers,
        **dataset_config
    )

    print("\nExperiment completed!")
    print(f"Results saved in outputs/{dataset_config['dataset_name']}/")
    print(f"- {dataset_config['dataset_name']}_train_full_logs.csv")
    print(f"- {dataset_config['dataset_name']}_test_full_logs.csv")

if __name__ == "__main__":
    main()