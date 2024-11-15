# experiments/reuters_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2

import torch
import torch.nn as nn
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleRNN_multiclass import SimpleRNN
from data_loaders.reuters import load_dataset, vocab_size, pad_idx
from train import train_lm
from test import test_lm_multiclass

def main():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

    print("#", "-"*100)
    print("# Running Reuters Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters with larger architecture
    model_hyperparams = {
        'vocab_size': vocab_size,
        'embed_dim': 300,        # Increased from 100
        'hidden_dim': 750,       # Increased from 256
        'output_dim': 46,        # Reuters has 46 classes
        'pad_idx': pad_idx,
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'reuters',
        'train_fn': train_lm,
        'test_fn': test_lm_multiclass,
        'dataset_loader': load_dataset,
        'model_class': SimpleRNN,
        'num_runs': 1,           # 1 run as per your original script
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