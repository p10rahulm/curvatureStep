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


def goldstein_price(x, y):
    term1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    term2 = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    return term1 * term2


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
        loss = goldstein_price(x[0], x[1])
        loss.backward()
        optimizer.step()
        path.append(x.detach().numpy().copy())

    return np.array(path)


# Create a grid for the Rosenbrock function
x = np.linspace(-1.5, 2.5, 200)
y = np.linspace(-2, 1, 200)
X, Y = np.meshgrid(x, y)
Z = goldstein_price(X, Y)

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimization Paths on the Goldstein Price Function', fontsize=27)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

optima = [(0, -1)]
lr = 2.5e-5
x0 = [0.5,0]
num_steps = 5000
# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=lr, steps=num_steps, x0=x0)

    # Plot the Rosenbrock function and optimization path
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
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
output_file = "outputs/plots/goldstein_price2.pdf"
plt.savefig(output_file)

plt.show()
