# experiments/stl10_nn_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

import torch
import torch.nn as nn
from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleNN import SimpleNN  # Changed to SimpleNN
from data_loaders.stl10 import load_stl10
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
    print("# Running STL10 Neural Network Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'input_size': 96 * 96 * 3,  # STL10 images are 96x96x3
        'hidden_size': 512,         # Larger hidden layer for complex images
        'num_classes': 10           # STL10 has 10 classes
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'stl10_nn',  # Different name to distinguish from ResNet version
        'train_fn': train,
        'test_fn': test,
        'dataset_loader': load_stl10,
        'model_class': SimpleNN,
        'num_runs': 3,           # 3 runs as per your original script
        'num_epochs': 10,        # 10 epochs as per your original script
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