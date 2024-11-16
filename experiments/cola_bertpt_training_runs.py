# experiments/cola_bert_pretrained_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 3

import torch
import torch.nn as nn
from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.bert_model import PretrainedBERTClassifier
from data_loaders.cola_bert import load_cola
from train import train_bert
from test import test_bert

def modify_optimizer_params(optimizers):
    """Modify optimizer parameters for all optimizers with specific settings for BERT"""
    modified_optimizers = []
    for optimizer_class, params in optimizers:
        new_params = params.copy()
        # Set learning rate for all optimizers
        new_params['lr'] = 1e-3
        
        # Set specific parameters if they exist
        if 'eps' in new_params:
            new_params['eps'] = 1e-6
        if 'epsilon' in new_params:
            new_params['epsilon'] = 1e-2
        if 'weight_decay' in new_params:
            new_params['weight_decay'] = 1e-2
            
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
    print("# Running CoLA BERT Pretrained Training Experiment")
    print("#", "-"*100)

    # Model hyperparameters
    model_hyperparams = {
        'num_classes': 2,        # Binary classification
        'freeze_bert': True      # Freeze BERT layers
    }

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'colabertpt',  # Name for pretrained BERT on CoLA
        'train_fn': train_bert,
        'test_fn': test_bert,
        'dataset_loader': load_cola,
        'model_class': PretrainedBERTClassifier,
        'num_runs': 2,           # 2 runs as per your original script
        'num_epochs': 4,         # 4 epochs as per your original script
        'model_hyperparams': model_hyperparams,
        'loss_criterion': nn.CrossEntropyLoss
    }

    # Modify optimizer parameters
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