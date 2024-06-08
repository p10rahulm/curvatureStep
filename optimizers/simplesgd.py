import torch
from torch.optim import Optimizer

class SimpleSGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(SimpleSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                step_size = group['lr']

                # Simple SGD update
                p.data = p.data - step_size * grad

        return loss
