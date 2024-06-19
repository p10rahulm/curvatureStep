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
from models.bert_model import BERTClassifier, TinyBERTClassifier
from data_loaders.ag_news_bert import load_ag_news
import torch
import torch.nn as nn
from train import train_bert
from test import test_bert

results = []

total_epochs = 5
total_runs = 2


print("#", "-" * 100)
print(f"# Running {total_epochs} epochs of training - {total_runs} runs")
print("#", "-" * 100)


for optimizer_class, default_params in optimizers:
    print(f"\nRunning AGNews training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    # Set device to GPU
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    dataset_loader = load_ag_news

    # Hyperparameters
    model_hyperparams = {
        'num_classes': 4,
        'freeze_bert': False
    }
    # model = BERTClassifier
    model = TinyBERTClassifier
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

write_to_file('outputs/ag_news_training_logs.csv', results)
