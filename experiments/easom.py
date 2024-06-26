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


def easom(x, y):
    return -torch.cos(x) * torch.cos(y) * torch.exp(-((x - np.pi)**2 + (y - np.pi)**2))


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
        loss = easom(x[0], x[1])
        loss.backward()
        optimizer.step()
        path.append(x.detach().numpy().copy())

    return np.array(path)


# Create a grid for the Easom function
x = np.linspace(0, 2*np.pi, 200)
y = np.linspace(0, 2*np.pi, 200)
X, Y = np.meshgrid(x, y)
Z = -np.cos(X) * np.cos(Y) * np.exp(-((X - np.pi)**2 + (Y - np.pi)**2))

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 7))
fig.suptitle('Optimization Paths on the Easom Function', fontsize=27)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

optima = [(np.pi, np.pi)]
lr = 2.0e-3
x0 = [np.pi-1.4, np.pi-1.4]
num_steps = 200

# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=lr, steps=num_steps, x0=x0)

    # Plot the Easom function and optimization path
    ax.contour(X, Y, Z, levels=np.logspace(-5, 0, 20), cmap='jet')
    ax.plot(path[:, 0], path[:, 1], 'ro-', label=f'{optimizer_class.__name__}')
    for optimum_num in range(len(optima)):
        optimum = optima[optimum_num]
        if len(optima) == 1:
            optima_label = 'Global Optimum'
        else:
            optima_label = f'Global Optimum {optimum_num}'
        ax.plot(optimum[0], optimum[1], 'x', markersize=8, label=optima_label)

    ax.set_xlabel('x', fontsize=20)
    ax.set_xlabel('y', fontsize=20)
    ax.set_title(f'{optimizer_class.__name__.replace("Curvature","-ACSS")}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image file
output_file = "outputs/plots/easom7.pdf"
plt.savefig(output_file)

plt.show()
