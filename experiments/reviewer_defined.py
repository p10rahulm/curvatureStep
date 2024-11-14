import os
import sys
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)


import torch
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np
from optimizers.simplesgd import SimpleSGD
from optimizers.simplesgd_curvature import SimpleSGDCurvature
from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature
from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature

def quadratic_loss(x, y):
    return 0.5 * x**2 + 0.01/2 * y**2

def run_optimization(optimizer_class, lr, steps, x0):
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float32)
    if str(optimizer_class.__name__) in ['SimpleSGDCurvature', 'HeavyBallCurvature', 'NAGCurvature']:
        optimizer = optimizer_class([x], lr=lr, epsilon=0, r_max=10)
    else:
        optimizer = optimizer_class([x], lr=lr)
    path = []

    for _ in range(steps):
        def closure():
            optimizer.zero_grad()
            loss = quadratic_loss(x[0], x[1])
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        path.append(x.detach().numpy().copy())

    return np.array(path)

# Create a grid for the quadratic loss function
x = np.linspace(-12, 12, 200)
y = np.linspace(-12, 12, 200)
X, Y = np.meshgrid(x, y)
Z = quadratic_loss(X, Y)

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]


# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Optimization Paths on Quadratic Loss Function', fontsize=24)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

optima = [(0, 0)]  # Global minimum is at (0,0)
lr = 0.001  # As specified in the problem
num_steps = 2000
x0 = [10.0, 10.0]  # As specified in the problem

# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=lr, steps=num_steps, x0=x0)

    # Plot the quadratic function and optimization path
    ax.contour(X, Y, Z, levels=np.logspace(-5, 2, 20), cmap='jet')
    ax.plot(path[:, 0], path[:, 1], 'ro-', label=f'{optimizer_class.__name__}')
    for optimum_num in range(len(optima)):
        optimum = optima[optimum_num]
        if len(optima)==1:
            optima_label='Global Optimum'
        else:
            optima_label = f'Global Optimum {optimum_num}'
        ax.plot(optimum[0], optimum[1], 'x', markersize=8, label=optima_label)

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title(f'{optimizer_class.__name__.replace("Curvature","-ACSS")}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image file
output_file = "outputs/plots/quadratic_loss128.pdf"
plt.savefig(output_file)

plt.show()