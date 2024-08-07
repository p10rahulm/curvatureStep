import torch
from torch.optim import Optimizer

class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'sum' not in state:
                    state['sum'] = torch.zeros_like(p.data)

                sum_ = state['sum']
                eps = group['eps']
                step_size = group['lr']
                weight_decay = group['weight_decay']

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                sum_.addcmul_(grad, grad)
                std = sum_.sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-step_size)

        return loss
