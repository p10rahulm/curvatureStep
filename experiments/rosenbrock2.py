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
from optimizers.adagrad import Adagrad
from optimizers.adagrad_curvature import AdagradCurvature
from optimizers.adam import Adam
from optimizers.adam_curvature import AdamCurvature
from optimizers.adamw import AdamW
from optimizers.adamw_curvature import AdamWCurvature
from optimizers.amsgrad import AMSGrad
from optimizers.amsgrad_curvature import AMSGradCurvature
from optimizers.rmsprop import RMSProp
from optimizers.rmsprop_curvature import RMSPropCurvature
from optimizers.rmsprop_with_momentum import RMSPropMomentum
from optimizers.rmsprop_with_momentum_curvature import RMSPropMomentumCurvature
from optimizers.heavyball import HeavyBall
from optimizers.heavyball_curvature import HeavyBallCurvature
from optimizers.nadam import NAdam
from optimizers.nadam_curvature import NAdamCurvature
from optimizers.nadamw import NAdamW
from optimizers.nadamw_curvature import NAdamWCurvature
from optimizers.nag import NAG
from optimizers.nag_curvature import NAGCurvature
from optimizers.adadelta import Adadelta
from optimizers.adadelta_curvature import AdadeltaCurvature

# Define the Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Define gradients of the Rosenbrock function
def rosenbrock_grad(x, y):
    grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    grad_y = 200 * (y - x**2)
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
        loss = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        loss.backward()
        optimizer.step()
        path.append(x.detach().numpy().copy())

    return np.array(path)

# Create a grid for the Rosenbrock function
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
num_steps = 1500

# List of optimizers to visualize
optimizers = [
    SimpleSGD, HeavyBall, NAG, SimpleSGDCurvature, HeavyBallCurvature, NAGCurvature
]

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Optimization Paths on Rosenbrock Function', fontsize=16)

# Adjust space between plots
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Run optimizations and plot results for each optimizer
for ax, optimizer_class in zip(axs.flatten(), optimizers):
    path = run_optimization(optimizer_class, lr=1.5e-3, steps=num_steps, x0=[-1.5, 2])

    # Plot the Rosenbrock function and optimization path
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    ax.plot(path[:, 0], path[:, 1], 'ro-', label=f'{optimizer_class.__name__}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{optimizer_class.__name__}')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as an image file
output_file = "outputs/plots/rosenblock.pdf"
plt.savefig(output_file)

plt.show()
