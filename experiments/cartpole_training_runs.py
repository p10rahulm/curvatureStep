# cartpole_training_runs.py

# Define the relative path to the project root from the current script
import os
import sys
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from data_loaders.cartpole import load_cartpole
from models.simpleDQN import SimpleDQN
import torch

results = []

print("#", "-" * 100)
print("# Running 10 episodes of training - 10 runs")
print("#", "-" * 100)

env = load_cartpole()
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n
model_class = lambda: SimpleDQN(obs_space, action_space)

for optimizer_class, default_params in optimizers:
    print(f"\nRunning CartPole Training with Optimizer = {str(optimizer_class.__name__)}")
    params = default_params.copy()

    def cartpole_dataset_loader():
        return load_cartpole()

    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader=cartpole_dataset_loader,
        model_class=model_class,
        num_runs=10,
        num_epochs=10,
        debug_logs=True
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/cartpole_training_logs.csv', results)
