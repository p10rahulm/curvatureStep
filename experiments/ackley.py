import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

from optimizers.simplesgd import SimpleSGD
from optimizers.simplesgd_curvature import SimpleSGDCurvature
from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature
from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature


def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - \
           torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))) + \
           torch.e + 20


# Function to run optimization
def run_optimization(optimizer_class, lr, steps, x0):
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float32)
    if str(optimizer_class.__name__) in ['SimpleSGDCurvature', 'HeavyBallCurvature', 'NAGCurvature']:
        optimizer = optimizer_class([x], lr=lr, clip_radius=10)
    else:
        optimizer = optimizer_class([x], lr=lr)
    path = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = ackley(x[0], x[1])
        loss.backward()
        optimizer.step()
        path.append(x.detach().numpy().copy())

    return np.array(path)


# Create a grid for the Ackley function
x = np.linspace(-1.25, 1.25, 200)
y = np.linspace(-1.25, 1.25, 200)
X, Y = np.meshgrid(x, y)
Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
    np.exp(0.5 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))) + np.e + 20

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimization Paths on the Ackley Function', fontsize=16)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

optima = [(0, 0)]
lr = 5e-3
x0 = [0.6, 0.5]
num_steps = 25

# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=lr, steps=num_steps, x0=x0)

    # Plot the Ackley function and optimization path
    ax.contour(X, Y, Z, levels=np.logspace(-1, 1, 20), cmap='jet')
    ax.plot(path[:, 0], path[:, 1], 'ro-', label=f'{optimizer_class.__name__}')
    for optimum_num in range(len(optima)):
        optimum = optima[optimum_num]
        if len(optima) == 1:
            optima_label = 'Global Optimum'
        else:
            optima_label = f'Global Optimum {optimum_num}'
        ax.plot(optimum[0], optimum[1], 'x', markersize=8, label=optima_label)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{optimizer_class.__name__}')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image file
output_file = "outputs/plots/ackley_function.pdf"
plt.savefig(output_file)

plt.show()
