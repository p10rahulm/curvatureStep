import torch
from torch.optim import Optimizer

class AdadeltaCurvature(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.95, eps=1e-6, weight_decay=0, epsilon=0.01):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        super(AdadeltaCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # First loop to store the last gradient
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['acc_delta'] = torch.zeros_like(p.data)

                state['last_grad'] = p.grad.data.clone()

        # Closure call to get the new gradient
        if closure is not None:
            loss = closure()

        # Second loop to update the parameters
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1
                step_size = group['lr']
                rho = group['rho']
                eps = group['eps']
                weight_decay = group['weight_decay']
                epsilon = group['epsilon']

                last_grad = state['last_grad']
                current_grad = grad

                radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))
                normed_last_grad = last_grad / torch.norm(last_grad)
                grad = radius * normed_last_grad

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                square_avg = state['square_avg']
                acc_delta = state['acc_delta']

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.data.add_(delta, alpha=-step_size)
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        return loss
