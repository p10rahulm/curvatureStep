# experiments/cola_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from models.bert_model import TinyBERTClassifierND
from data_loaders.cola_bert import load_cola
import torch
import torch.nn as nn
from train import train_bert
from test import test_bert

results = []

total_epochs = 4
total_runs = 2

print("#", "-" * 100)
print(f"# Running {total_epochs} epochs of training - {total_runs} runs")
print("#", "-" * 100)

for optimizer_class, default_params in optimizers:
    print(f"\nRunning CoLA training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()
    params['lr'] = 1e-3
    if 'eps' in params.keys():
        params['eps'] = 1e-6
    if 'epsilon' in params.keys():
        params['epsilon'] = 1e-2
    if 'weight_decay' in params.keys():
        params['weight_decay'] = 1e-2
    
    # Set device to GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    dataset_loader = load_cola

    # Hyperparameters
    model_hyperparams = {
        'num_classes': 2,
        'freeze_bert': False
    }
    model = TinyBERTClassifierND
    trainer_function = train_bert
    test_function = test_bert
    loss_criterion = nn.CrossEntropyLoss
    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=dataset_loader,
        model_class=model,
        num_runs=total_runs,
        num_epochs=total_epochs,
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

write_to_file('outputs/cola_training_logs.csv', results)

