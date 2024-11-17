# experiments/ag_news_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use GPU 0

import torch
import torch.nn as nn
from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleRNN_multiclass import SimpleRNN
from data_loaders.ag_news import load_ag_news, vocab  # Import vocab directly from the module
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
    print("# Running AG News Training Experiment")
    print("#", "-"*100)
    
    # Get vocabulary information from the imported vocab object
    vocab_size = len(vocab)
    pad_idx = vocab["<pad>"]
    
    # Model hyperparameters
    model_hyperparams = {
        'vocab_size': vocab_size,
        'embed_dim': 100,      # Embedding dimension
        'hidden_dim': 256,     # RNN hidden dimension
        'output_dim': 4,       # AG News has 4 classes
        'pad_idx': pad_idx,    # Padding token index
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'agnews',
        'train_fn': train_lm,
        'test_fn': test_lm_multiclass,
        'dataset_loader': load_ag_news,
        'model_class': SimpleRNN,
        'num_runs': 2,          # 2 runs
        'num_epochs': 5,        # 5 epochs
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