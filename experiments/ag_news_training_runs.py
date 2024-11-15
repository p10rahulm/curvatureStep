# experiments/ag_news_training_runs.py

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
from models.bert_model import TinyBERTClassifier  # Using TinyBERT as per your script
from data_loaders.ag_news_bert import load_ag_news
from train import train_bert
from test import test_bert

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

    # Model hyperparameters
    model_hyperparams = {
        'num_classes': 4,
        'freeze_bert': False
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'agnews',
        'train_fn': train_bert,
        'test_fn': test_bert,
        'dataset_loader': load_ag_news,
        'model_class': TinyBERTClassifier,  # Using TinyBERT for efficiency
        'num_runs': 2,          # 2 runs as per your original script
        'num_epochs': 5,        # 5 epochs as per your original script
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