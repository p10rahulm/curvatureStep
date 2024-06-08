import torch
from torch.optim import Optimizer

class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta, self).__init__(params, defaults)

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
                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(p.data)
                if 'acc_delta' not in state:
                    state['acc_delta'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                acc_delta = state['acc_delta']
                rho = group['rho']
                eps = group['eps']
                step_size = group['lr']
                weight_decay = group['weight_decay']

                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.data.add_(delta, alpha=-step_size)
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        return loss
