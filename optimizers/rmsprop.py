import torch
from torch.optim import Optimizer


class RMSProp(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(RMSProp, self).__init__(params, defaults)

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
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                eps = group['eps']
                step_size = group['lr']
                weight_decay = group['weight_decay']

                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                avg = square_avg.sqrt().add_(eps)

                p.data.addcdiv_(grad, avg, value=-step_size)

        return loss
