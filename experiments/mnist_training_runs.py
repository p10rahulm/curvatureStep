# experiments/mnist_training_runs.py

import os
import sys
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_all_experiments
from optimizer_params import optimizers
from models.simpleNN import SimpleNN
from data_loaders.mnist import load_mnist
from train import train
from test import test

def main():
    print("#", "-"*100)
    print("# Running MNIST Training Experiment")
    print("#", "-"*100)

    # Dataset-specific configuration
    dataset_config = {
        'dataset_name': 'mnist',
        'train_fn': train,
        'test_fn': test,
        'dataset_loader': load_mnist,
        'model_class': SimpleNN,
        'num_runs': 2,            # You can adjust these parameters
        'num_epochs': 5,         # You can adjust these parameters
        'model_hyperparams': None,  # Add if your SimpleNN needs specific params
        'loss_criterion': None    # Will use default CrossEntropyLoss
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