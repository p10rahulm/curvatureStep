import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from optimizers.simplesgd import SimpleSGD
from optimizers.simplesgd_curvature import SimpleSGDCurvature
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

# Run optimizations
sgd_path = run_optimization(SimpleSGD, lr=1e-3, steps=1000, x0=[-1.5, 2])
acss_path = run_optimization(SimpleSGDCurvature, lr=1e-3, steps=1000, x0=[-1.5, 2])

# Plot the Rosenbrock function and optimization paths
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
plt.colorbar(label='f(x, y)')

plt.plot(acss_path[:, 0], acss_path[:, 1], 'ro-', label='ACSS')
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'bo-', label='SGD')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Paths on Rosenbrock Function')
plt.legend()
plt.show()
