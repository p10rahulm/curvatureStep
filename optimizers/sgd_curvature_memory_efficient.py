import torch
from torch.optim import Optimizer

class SGDCurvatureMemoryEfficient(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, epsilon=epsilon, r_max=r_max)
        super(SGDCurvatureMemoryEfficient, self).__init__(params, defaults)

    def step(self, closure):
        loss = closure()  # Compute loss and perform backward pass to get g_t
        g_norm_sq = 0.0

        # Step 1: Compute ||g_t||² and store scalar quantities
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g_norm_sq += p.grad.data.pow(2).sum().item()

        # Step 2: Update parameters to tentative point w_tilde = w_t - lr * g_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=-group['lr'])

        # Step 3: Compute loss and gradients at tentative point (g'_t)
        loss = closure()  # This will overwrite p.grad with g'_t
        gprime_norm_sq = 0.0
        g_gprime_inner = 0.0

        # Step 4: Compute ||g'_t||² and inner product <g_t, g'_t>
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    gprime_norm_sq += p.grad.data.pow(2).sum().item()

        # Step 5: Restore original parameters w_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(p.grad.data, alpha=group['lr'])  # Reverse the tentative update

        # Step 6: Recompute original gradients g_t
        loss = closure()  # Recompute gradients at w_t

        # Step 7: Compute inner product <g_t, g'_t> and difference norm ||g_t - g'_t||
        diff_norm_sq = 0.0
        orig_grad_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g_t = p.grad.data.clone()
                    # Since p.grad now contains g_t, and we have g'_t from before restoring parameters
                    # Compute inner product and difference norm
                    g_gprime_inner += (g_t * p.grad.data).sum().item()
                    diff_norm_sq += (g_t - p.grad.data).pow(2).sum().item()

        # Step 8: Compute normalized radius of curvature r_t
        epsilon = self.defaults['epsilon']
        r_t = (g_norm_sq ** 0.5) / (diff_norm_sq ** 0.5 + epsilon)
        r_t = min(r_t, self.defaults['r_max'])

        # Step 9: Final parameter update
        g_norm = (g_norm_sq ** 0.5) + epsilon
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g_t = p.grad.data
                    p.data = p.data - group['lr'] * g_t - group['lr'] * r_t * (g_t / g_norm)

        return loss
