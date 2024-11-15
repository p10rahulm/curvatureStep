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

import psutil
from collections import defaultdict


def process_train_logs(train_df, num_epochs):
    """Process training logs to get mean and std per epoch for each optimizer"""
    # Group by optimizer and epoch
    grouped = train_df.groupby(['Optimizer Name', 'Epoch number'])['Average Training Loss'].agg(['mean', 'std']).reset_index()
    
    # Prepare the new dataframe structure
    mean_cols = [f'Mean_Training_Loss_epoch{i}' for i in range(1, num_epochs + 1)]
    std_cols = [f'Std_Training_Loss_epoch{i}' for i in range(1, num_epochs + 1)]
    
    # Initialize the result dataframe
    result = pd.DataFrame(columns=['Optimizer Name'] + mean_cols + std_cols)
    
    # Fill in the data
    for optimizer in train_df['Optimizer Name'].unique():
        row = {'Optimizer Name': optimizer}
        
        # Get data for this optimizer
        opt_data = grouped[grouped['Optimizer Name'] == optimizer]
        
        # Fill means
        for epoch in range(1, num_epochs + 1):
            epoch_data = opt_data[opt_data['Epoch number'] == epoch]
            if not epoch_data.empty:
                row[f'Mean_Training_Loss_epoch{epoch}'] = epoch_data['mean'].iloc[0]
                row[f'Std_Training_Loss_epoch{epoch}'] = epoch_data['std'].iloc[0]
            else:
                row[f'Mean_Training_Loss_epoch{epoch}'] = None
                row[f'Std_Training_Loss_epoch{epoch}'] = None
        
        result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)
    
    return result

def process_test_logs(test_df):
    """Process test logs to get mean and std of accuracy and loss for each optimizer"""
    # Group by optimizer
    grouped = test_df.groupby('Optimizer Name').agg({
        'Average Test set Loss': ['mean', 'std'],
        'Average Test set accuracy': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['Optimizer Name', 
                      'Mean_Test_Loss', 'Std_Test_Loss',
                      'Mean_Test_Accuracy', 'Std_Test_Accuracy']
    
    return grouped

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
    previous_train_logs=None,
    previous_test_logs=None
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
    memory_logs = []
    
    # Load data
    train_loader, test_loader = dataset_loader()
    criterion = loss_criterion()
    
    # Function to get current memory usage
    def get_memory_usage():
        memory_stats = {}
        
        # CPU Memory
        process = psutil.Process(os.getpid())
        memory_stats['cpu_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        # GPU Memory if available
        if torch.cuda.is_available():
            memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(device) / (1024 * 1024)
            memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(device) / (1024 * 1024)
            
        return memory_stats
    
    # Track peak memory per run
    for run_number in range(num_runs):
        print(f"Running {optimizer_class.__name__} - Run {run_number + 1}/{num_runs}")
        
        # Initialize model
        if model_hyperparams is None:
            model = model_class().to(device)
        else:
            model = model_class(**model_hyperparams).to(device)
            
        # Initialize optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        # Get initial memory state
        initial_memory = get_memory_usage()
        
        # Train and log results
        train_losses = train_fn(model, train_loader, criterion, optimizer, device, num_epochs)
        
        # Get peak memory after training
        peak_memory = get_memory_usage()
        
        # Log memory usage
        memory_log = {
            'Optimizer Name': optimizer_class.__name__,
            'Running loop number': run_number + 1,
            'Initial CPU Memory (MB)': initial_memory.get('cpu_memory_mb', 0),
            'Peak CPU Memory (MB)': peak_memory.get('cpu_memory_mb', 0),
            'Initial GPU Memory Allocated (MB)': initial_memory.get('gpu_allocated_mb', 0),
            'Peak GPU Memory Allocated (MB)': peak_memory.get('gpu_allocated_mb', 0),
            'Initial GPU Memory Reserved (MB)': initial_memory.get('gpu_reserved_mb', 0),
            'Peak GPU Memory Reserved (MB)': peak_memory.get('gpu_reserved_mb', 0)
        }
        memory_logs.append(memory_log)
        
        # Save memory logs after each run
        pd.DataFrame(memory_logs).to_csv(
            os.path.join(output_dir, f'{dataset_name}_memory_logs.csv'), 
            index=False
        )
        
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
        
        # Clear memory after each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to DataFrames for return
    train_df = pd.DataFrame(train_logs)
    test_df = pd.DataFrame(test_logs)
    memory_df = pd.DataFrame(memory_logs)
    
    # Process and save memory statistics
    memory_stats = memory_df.groupby('Optimizer Name').agg({
        'Peak CPU Memory (MB)': ['mean', 'std'],
        'Peak GPU Memory Allocated (MB)': ['mean', 'std'],
        'Peak GPU Memory Reserved (MB)': ['mean', 'std']
    }).round(2)
    
    memory_stats.to_csv(
        os.path.join(output_dir, f'{dataset_name}_memory_stats.csv'),
        sep='|'
    )
    
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
    print(f"Process ID: {os.getpid()}\nRunning Dataset: {dataset_name}")  # Print PID
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
    
    # Process and save grouped results
    grouped_train_df = process_train_logs(final_train_df, num_epochs)
    grouped_test_df = process_test_logs(final_test_df)

    # Save grouped results
    grouped_train_df.to_csv(
        os.path.join(output_dir, f'{dataset_name}_grouped_train.csv'),
        index=False,
        sep='|'
    )
    grouped_test_df.to_csv(
        os.path.join(output_dir, f'{dataset_name}_grouped_test.csv'),
        index=False,
        sep='|'
    )
    
    return final_train_df, final_test_df