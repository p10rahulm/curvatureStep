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
    loss_criterion=None,
    previous_train_logs=None,  # Add parameter for previous logs
    previous_test_logs=None    # Add parameter for previous logs
):
    # Setup defaults
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss_criterion is None:
        loss_criterion = torch.nn.CrossEntropyLoss
    
    # Create output directory
    output_dir = os.path.join('outputs', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logging lists
    train_logs = [] if previous_train_logs is None else previous_train_logs
    test_logs = [] if previous_test_logs is None else previous_test_logs
    
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
            'Average Test set accuracy': test_accuracy * 100
        })
        
        # Save intermediate results after each run
        pd.DataFrame(train_logs).to_csv(
            os.path.join(output_dir, f'{dataset_name}_train_full_logs.csv'), 
            index=True
        )
        pd.DataFrame(test_logs).to_csv(
            os.path.join(output_dir, f'{dataset_name}_test_full_logs.csv'), 
            index=True
        )
    
    # Convert to DataFrames for return
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
    loss_criterion=None,
    resume_from_checkpoint=False  # Add option to resume from existing logs
):
    """
    Runs experiments for all optimizers and saves results frequently
    """
    output_dir = os.path.join('outputs', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize or load existing logs
    all_train_logs = []
    all_test_logs = []
    
    if resume_from_checkpoint:
        try:
            train_path = os.path.join(output_dir, f'{dataset_name}_train_full_logs.csv')
            test_path = os.path.join(output_dir, f'{dataset_name}_test_full_logs.csv')
            if os.path.exists(train_path) and os.path.exists(test_path):
                existing_train_df = pd.read_csv(train_path)
                existing_test_df = pd.read_csv(test_path)
                all_train_logs = existing_train_df.to_dict('records')
                all_test_logs = existing_test_df.to_dict('records')
                print(f"Resumed from existing logs in {output_dir}")
        except Exception as e:
            print(f"Could not load existing logs: {e}")
            all_train_logs = []
            all_test_logs = []
    
    for optimizer_class, default_params in optimizers:
        # Skip if this optimizer has already been completed
        if all_train_logs and optimizer_class.__name__ in [log['Optimizer Name'] for log in all_train_logs]:
            print(f"Skipping {optimizer_class.__name__} - already completed")
            continue
            
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
            loss_criterion=loss_criterion,
            previous_train_logs=all_train_logs,
            previous_test_logs=all_test_logs
        )
        
        # Results are already saved in run_experiment_with_logging
    
    # Load final results
    final_train_df = pd.read_csv(os.path.join(output_dir, f'{dataset_name}_train_full_logs.csv'))
    final_test_df = pd.read_csv(os.path.join(output_dir, f'{dataset_name}_test_full_logs.csv'))
    
    return final_train_df, final_test_df