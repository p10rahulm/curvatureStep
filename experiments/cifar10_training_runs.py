import csv
from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers
from data_loaders.cifar10 import load_cifar10  # Change to the dataset loader you want to use
from models.simpleCNN import SimpleCNN  # Change to the model you want to use

results = []

print("#", "-" * 100)
print("# Running 10 epochs of training - 10 runs")
print("#", "-" * 100)

dataset_loader = load_cifar10  # Set the dataset loader
model_class = SimpleCNN  # Set the model class

for optimizer_class, default_params in optimizers:
    params = default_params.copy()
    mean_accuracy, std_accuracy = run_experiment(
        optimizer_class,
        params,
        dataset_loader,
        model_class,
        num_runs=10,
        num_epochs=10,
        debug_logs=True
    )
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/cifar10_training_logs.csv', results)
