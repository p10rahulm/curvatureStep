# experiments/flowers102_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2

import torch
import torch.nn as nn
from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.resnet import SimpleResNet
from data_loaders.flowers102 import load_flowers102
from train import train
from test import test

def main():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

    print("#", "-"*100)
    print("# Running Flowers102 Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'num_classes': 102  # Flowers102 has 102 classes
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'flowers102',
        'train_fn': train,
        'test_fn': test,
        'dataset_loader': load_flowers102,
        'model_class': SimpleResNet,
        'num_runs': 3,           # 3 runs
        'num_epochs': 10,        # 10 epochs
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