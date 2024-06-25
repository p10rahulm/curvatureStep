import os
import sys
# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

# Import your optimizer classes here (same as in your original code)

# Define the Himmelblau function
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Define gradients of the Himmelblau function
def himmelblau_grad(x, y):
    grad_x = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    grad_y = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([grad_x, grad_y])

# Function to run optimization
def run_optimization(optimizer_class, lr, steps, x0):
    x = torch.tensor(x0, requires_grad=True, dtype=torch.float32)
    if str(optimizer_class.__name__) in ['SimpleSGDCurvature','HeavyBallCurvature','NAGCurvature']:
        optimizer = optimizer_class([x], lr=lr, clip_radius=10)
    else:
        optimizer = optimizer_class([x], lr=lr)
    path = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
        loss.backward()
        optimizer.step()
        path.append(x.detach().numpy().copy())

    return np.array(path)

# Create a grid for the Himmelblau function
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)
num_steps = 1500

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimization Paths on Himmelblau Function', fontsize=16)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=1e-3, steps=num_steps, x0=[-4, 4])

    # Plot the Himmelblau function and optimization path
    cs = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    ax.plot(path[:, 0], path[:, 1], 'ro-', label=f'{optimizer_class.__name__}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{optimizer_class.__name__}')
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image file
output_file = "outputs/plots/himmelblau.pdf"
plt.savefig(output_file)

plt.show()