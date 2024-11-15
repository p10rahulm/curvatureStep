# experiments/imdb_training_runs.py

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
from models.simpleRNN import SimpleRNN
from data_loaders.imdb import load_imdb_reviews, vocab
from train import train_lm
from test import test_lm

def main():
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available, using CPU")

    print("#", "-"*100)
    print("# Running IMDB Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'vocab_size': len(vocab),
        'embed_dim': 100,
        'hidden_dim': 256,
        'output_dim': 1,
        'pad_idx': vocab["<pad>"]
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'imdb',
        'train_fn': train_lm,
        'test_fn': test_lm,
        'dataset_loader': load_imdb_reviews,
        'model_class': SimpleRNN,
        'num_runs': 3,          # 10 runs as per your original script
        'num_epochs': 2,         # 2 epochs as per your original script
        'model_hyperparams': model_hyperparams,
        'loss_criterion': nn.BCELoss  # Binary Cross Entropy Loss for binary classification
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