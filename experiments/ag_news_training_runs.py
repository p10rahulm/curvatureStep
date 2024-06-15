# ag_news_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from data_loaders.ag_news import load_ag_news
from models.simpleRNN import SimpleRNN
import torch

results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)

dataset_loader = load_ag_news
TEXT, LABEL = None, None  # Initialize TEXT and LABEL variables
model_class = SimpleRNN

for optimizer_class, default_params in optimizers:
    print(f"\nRunning AG News Training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    def ag_news_dataset_loader():
        global TEXT, LABEL
        train_loader, test_loader, TEXT, LABEL = load_ag_news(batch_size=64, device='cpu')
        return train_loader, test_loader

    input_dim = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 4
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    model = SimpleRNN(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx)

    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=ag_news_dataset_loader,
        model_class=lambda: model,
        num_runs=10,
        num_epochs=10,
        debug_logs=True
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/ag_news_training_logs.csv', results)
