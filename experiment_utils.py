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
import pandas as pd
from tqdm import tqdm


def run_experiment_old(optimizer_class, optimizer_params, dataset_loader=None, 
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


def run_experiment_with_logging(
    optimizer_class, 
    optimizer_params, 
    dataset_name,
    train_fn,
    test_fn,
    dataset_loader=None, 
    model_class=None, 
    num_runs=10, 
    num_epochs=2, 
    device=None,
    model_hyperparams=None, 
    loss_criterion=None
):
    """
    Runs experiments with specified training and testing functions, logging results
    
    Args:
        optimizer_class: The optimizer class to use
        optimizer_params: Parameters for the optimizer
        dataset_name: Name of the dataset (for logging)
        train_fn: Training function to use (must return list of epoch losses)
        test_fn: Testing function to use (must return tuple of (loss, accuracy))
        dataset_loader: Function to load the dataset
        model_class: Model class to use
        num_runs: Number of runs to perform
        num_epochs: Number of epochs per run
        device: Device to run on
        model_hyperparams: Hyperparameters for the model
        loss_criterion: Loss function to use
    """
    # Setup defaults
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_criterion is None:
        loss_criterion = torch.nn.CrossEntropyLoss
    
    # Create output directory
    output_dir = os.path.join('outputs', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logging DataFrames
    train_logs = []
    test_logs = []
    
    # Load data
    train_loader, test_loader = dataset_loader()
    criterion = loss_criterion()
    
    # Run experiments
    for run_number in range(num_runs):
        print(f"Running {optimizer_class.__name__} - Run {run_number + 1}/{num_runs}")
        
        # Initialize model
        if model_hyperparams is None:
            model = model_class().to(device)
        else:
            model = model_class(**model_hyperparams).to(device)
            
        # Initialize optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        # Train and log results
        train_losses = train_fn(model, train_loader, criterion, optimizer, device, num_epochs)
        
        # Log training results
        for epoch, loss in enumerate(train_losses, 1):
            train_logs.append({
                'Optimizer Name': optimizer_class.__name__,
                'Running loop number': run_number + 1,
                'Epoch number': epoch,
                'Average Training Loss': loss
            })
        
        # Test and log results
        test_loss, test_accuracy = test_fn(model, test_loader, criterion, device)
        test_logs.append({
            'Optimizer Name': optimizer_class.__name__,
            'Running loop number': run_number + 1,
            'Average Test set Loss': test_loss,
            'Average Test set accuracy': test_accuracy * 100  # Convert to percentage
        })
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_logs)
    test_df = pd.DataFrame(test_logs)
    
    return train_df, test_df

def run_all_experiments(
    optimizers, 
    dataset_name, 
    train_fn, 
    test_fn, 
    dataset_loader, 
    model_class,
    num_runs=2, 
    num_epochs=5,
    model_hyperparams=None,
    loss_criterion=None
):
    """
    Runs experiments for all optimizers and saves results
    
    Args:
        optimizers: List of (optimizer_class, params) tuples
        dataset_name: Name of the dataset
        train_fn: Training function to use
        test_fn: Testing function to use
        dataset_loader: Function to load the dataset
        model_class: Model class to use
        num_runs: Number of runs per optimizer
        num_epochs: Number of epochs per run
        model_hyperparams: Hyperparameters for the model
        loss_criterion: Loss function to use
    """
    all_train_logs = []
    all_test_logs = []
    
    for optimizer_class, default_params in optimizers:
        print(f"\nRunning {dataset_name} training with Optimizer = {optimizer_class.__name__}")
        params = default_params.copy()
        
        train_df, test_df = run_experiment_with_logging(
            optimizer_class=optimizer_class,
            optimizer_params=params,
            dataset_name=dataset_name,
            train_fn=train_fn,
            test_fn=test_fn,
            dataset_loader=dataset_loader,
            model_class=model_class,
            num_runs=num_runs,
            num_epochs=num_epochs,
            model_hyperparams=model_hyperparams,
            loss_criterion=loss_criterion
        )
        
        all_train_logs.append(train_df)
        all_test_logs.append(test_df)
    
    # Combine all results
    final_train_df = pd.concat(all_train_logs, ignore_index=True)
    final_test_df = pd.concat(all_test_logs, ignore_index=True)
    
    # Save combined results
    output_dir = os.path.join('outputs', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    final_train_df.to_csv(os.path.join(output_dir, f'{dataset_name}_train_full_logs.csv'), index=True)
    final_test_df.to_csv(os.path.join(output_dir, f'{dataset_name}_test_full_logs.csv'), index=True)
    
    return final_train_df, final_test_df
