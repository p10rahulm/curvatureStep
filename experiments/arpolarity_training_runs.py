import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from models.simpleRNN_multiclass import SimpleRNN

from data_loaders.arpolarity import load_dataset
from data_loaders.arpolarity import vocab_size, pad_idx

import torch
import torch.nn as nn
from train import train_lm
from test import test_lm_multiclass

results = []

total_epochs = 10
total_runs = 2

print("#", "-" * 100)
print(f"# Running {total_epochs} epochs of training - {total_runs} runs")
print("#", "-" * 100)

for optimizer_class, default_params in optimizers:
    print(f"\nRunning amazon-review-polarity training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    if str(optimizer_class.__name__) in ["SimpleSGDCurvature", "HeavyBallCurvature", "NAGCurvature"]:
        params['clip_radius'] = 10
    
    # Set device to GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    dataset_loader = load_dataset

    # Hyperparameters
    model_hyperparams = {
        'vocab_size': vocab_size,
        'embed_dim': 100,
        'hidden_dim': 256,
        'output_dim': 5,
        'pad_idx': pad_idx,
    }
    # model = SimpleRNN
    model = SimpleRNN
    trainer_function = train_lm
    test_function = test_lm_multiclass
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


