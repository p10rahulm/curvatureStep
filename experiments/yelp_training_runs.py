# experiments/yelp_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import torch
import torch.nn as nn
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleRNN_multiclass import SimpleRNN
from data_loaders.yelp import load_yelp, vocab_size, pad_idx
from train import train_lm
from test import test_lm_multiclass

def modify_optimizer_params(optimizers):
    """Modify optimizer parameters for specific optimizers"""
    modified_optimizers = []
    for optimizer_class, params in optimizers:
        new_params = params.copy()
        if optimizer_class.__name__ in ["SimpleSGDCurvature", "HeavyBallCurvature"]:
            new_params['r_max'] = 10
        modified_optimizers.append((optimizer_class, new_params))
    return modified_optimizers

def main():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

    print("#", "-"*100)
    print("# Running Yelp Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'vocab_size': vocab_size,
        'embed_dim': 100,
        'hidden_dim': 256,
        'output_dim': 2,
        'pad_idx': pad_idx,
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'yelp',
        'train_fn': train_lm,
        'test_fn': test_lm_multiclass,
        'dataset_loader': load_yelp,
        'model_class': SimpleRNN,
        'num_runs': 2,          # 2 runs as per your original script
        'num_epochs': 5,        # 5 epochs as per your original script
        'model_hyperparams': model_hyperparams,
        'loss_criterion': nn.CrossEntropyLoss
    }

    # Modify optimizer parameters for specific optimizers
    modified_optimizers = modify_optimizer_params(optimizers)

    # Run experiments for all optimizers
    train_df, test_df = run_all_experiments(
        optimizers=modified_optimizers,
        **dataset_config
    )

    print("\nExperiment completed!")
    print(f"Results saved in outputs/{dataset_config['dataset_name']}/")
    print(f"- {dataset_config['dataset_name']}_train_full_logs.csv")
    print(f"- {dataset_config['dataset_name']}_test_full_logs.csv")

if __name__ == "__main__":
    main()