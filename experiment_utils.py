# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root =os.getcwd()
sys.path.insert(0, project_root)

import torch
import numpy as np
from train import train
from test import test

from utilities import set_seed
from models.simpleNN import SimpleNN
from data_loaders.mnist import load_mnist


def run_experiment(optimizer_class, optimizer_params, dataset_loader=None, 
                   model_class=None, num_runs=10, num_epochs=2, debug_logs=False,
                   device=None, model_hyperparams=None,
                   loss_criterion=None, trainer_fn=None, tester_fn=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_loader is None:
        dataset_loader = load_mnist
    if model_class is None:
        model_class = SimpleNN
    if loss_criterion is None:
        loss_criterion = torch.nn.CrossEntropyLoss
    if trainer_fn is None:
        trainer_fn = train
    if tester_fn is None:
        tester_fn = test
    
    set_seed(42)
    print("params=", optimizer_params)
    train_loader, test_loader = dataset_loader()

    criterion = loss_criterion()
    accuracies = []
    for run_number in range(num_runs):
        if debug_logs:
            print(f"Running Loop: {run_number + 1}/{num_runs}")

        if model_hyperparams is None:
            model = model_class().to(device)
        else:
            model = model_class(**model_hyperparams).to(device)

        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        trainer_fn(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
        accuracy = tester_fn(model, test_loader, criterion, device)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy
