import torch
import numpy as np
from data_loaders.mnist import load_mnist
from models.simpleNN import SimpleNN
from train import train
from test import test
from utilities import set_seed

def run_experiment(optimizer_class, optimizer_params, num_runs=10, num_epochs=2, debug_logs=False):
    set_seed(42)
    train_loader, test_loader = load_mnist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    accuracies = []
    for run_number in range(num_runs):
        if debug_logs:
            print(f"Running Loop: {run_number+1}/{num_runs}")

        model = SimpleNN().to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
        accuracy = test(model, test_loader, criterion, device)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return mean_accuracy, std_accuracy
