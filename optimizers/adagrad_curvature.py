import torch
from torch.optim import Optimizer

class AdagradCurvature(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0, epsilon=0.01):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        super(AdagradCurvature, self).__init__(params, defaults)

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
                    state['sum'] = torch.zeros_like(p.data)

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
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']

                last_grad = state['last_grad']
                current_grad = grad

                if weight_decay != 0:
                    current_grad = current_grad.add_(p.data, alpha=weight_decay)

                radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))
                normed_last_grad = last_grad / torch.norm(last_grad)
                grad = radius * normed_last_grad

                sum_ = state['sum']
                eps = group['eps']

                sum_.addcmul_(grad, grad)
                std = sum_.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-step_size)

        return loss
