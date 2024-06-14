import csv
from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers

results = []

print("#","-"*100)
print("# Running 10 epochs of training - 10 runs")
print("#","-"*100)

for optimizer_class, default_params in optimizers:
    params = default_params.copy()
    mean_accuracy, std_accuracy = run_experiment(optimizer_class, params, num_runs=10, num_epochs=10, debug_logs=True)
    results.append({
        'optimizer': optimizer_class.__name__,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    })

write_to_file('outputs/mnist_training_logs.csv', results)
