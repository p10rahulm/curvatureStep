# timit_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from data_loaders.timit import load_timit
from models.simpleRNN_speech import SimpleRNNSpeech
import torch

results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)

dataset_loader = load_timit
model_class = SimpleRNNSpeech

for optimizer_class, default_params in optimizers:
    print(f"\nRunning TIMIT Training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    def timit_dataset_loader():
        return load_timit(batch_size=64)

    input_dim = 13  # Number of MFCC features
    hidden_dim = 256
    output_dim = 61  # Number of phonemes
    n_layers = 2
    bidirectional = True
    dropout = 0.5

    model = SimpleRNNSpeech(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=timit_dataset_loader,
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

write_to_file('outputs/timit_training_logs.csv', results)
