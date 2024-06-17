# experiments/ag_news_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from models.simpleRNN_multiclass import SimpleRNN
from data_loaders.ag_news import vocab
import torch
import torch.nn as nn
from data_loaders.ag_news import load_ag_news
from train import train_lm
from test import test_lm_multiclass

results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)

for optimizer_class, default_params in optimizers:
    print(f"\nRunning AG News training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    # Set device to GPU
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    dataset_loader = load_ag_news

    # Hyperparameters
    model_hyperparams = {
        'vocab_size': len(vocab),
        'embed_dim': 100,
        'hidden_dim': 256,
        'output_dim': 4,
        'pad_idx': vocab["<pad>"],
    }
    model = SimpleRNN
    trainer_function = train_lm
    test_function = test_lm_multiclass
    loss_criterion = nn.CrossEntropyLoss
    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=dataset_loader,
        model_class=model,
        num_runs=10,
        num_epochs=10,
        debug_logs=True,
        model_hyperparams=model_hyperparams,
        loss_criterion=loss_criterion,
        device=device,
        trainer_fn=trainer_function,
        tester_fn=test_function,
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/ag_news_training_logs.csv', results)
