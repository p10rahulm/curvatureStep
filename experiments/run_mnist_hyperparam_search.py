import csv
from experiment_utils import run_experiment
from utilities import write_to_file
from optimizer_params import optimizers

results = []

momentum_values = [round(x * 0.05, 2) for x in range(20)]
print("#","-"*100)
print("# Hyperparameter Search")
print("#","-"*100)
for optimizer_class, default_params in optimizers:
    has_momentum_param = True
    for momentum in momentum_values:
        print(f"\nRunning Hyperparameter grid search with "
              f"Optimizer = {str(optimizer_class.__name__)} and momentum={momentum}.")
        params = default_params.copy()
        if 'momentum' in params:
            params['momentum'] = momentum
        elif 'betas' in params:
            params['betas'] = (momentum, params['betas'][1])
        elif 'rho' in params:
            params['rho'] = momentum
        elif 'alpha' in params:
            params['alpha'] = momentum
        elif 'weight_decay' in params:
            params['weight_decay'] = momentum
        else:
            has_momentum_param = False

        mean_accuracy, std_accuracy = run_experiment(optimizer_class, params, debug_logs=True)
        results.append({
            'optimizer': optimizer_class.__name__,
            'momentum': momentum,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        })

        if not has_momentum_param:
            break

write_to_file('../outputs/optimizer_results.csv', results)
