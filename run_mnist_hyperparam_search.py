import csv
from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers

results = []

momentum_values = [round(x * 0.05, 2) for x in range(20)]

for optimizer_class, default_params in optimizers:
    for momentum in momentum_values:
        params = default_params.copy()
        if 'momentum' in params:
            params['momentum'] = momentum
        elif 'betas' in params:
            params['betas'] = (momentum, params['betas'][1])

        mean_accuracy, std_accuracy = run_experiment(optimizer_class, params)
        results.append({
            'optimizer': optimizer_class.__name__,
            'momentum': momentum,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        })

write_to_file('outputs/optimizer_results.csv', results)
