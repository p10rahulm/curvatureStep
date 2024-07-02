# imdb_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from models.simpleRNN import SimpleRNN
from data_loaders.imdb import vocab
import torch
import torch.nn as nn
from data_loaders.imdb import load_imdb_reviews
from train import train_lm
from test import test_lm


from optimizers.simplesgd_curvature import SimpleSGDCurvature
from optimizers.simplesgd_diffcurvature import SimpleSGDCurvatureDiff
results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)
optimizers = [
# (SimpleSGDCurvature, {'lr': 1e-3, 'epsilon': 0.01, 'clip_radius': 500}),
(SimpleSGDCurvatureDiff, {'lr': 1e-3, 'epsilon': 0.01, 'clip_radius': 500}),

]
print("#","-"*100)
print("# Running 10 epochs of training - 10 runs")
print("#","-"*100)

for optimizer_class, default_params in optimizers:
    print(f"\nRunning MNIST training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()
    mean_accuracy, std_accuracy = run_experiment(optimizer_class, params, num_runs=2, num_epochs=2, debug_logs=True)
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })