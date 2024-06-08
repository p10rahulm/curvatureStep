import torch
from torch.optim import Optimizer

class HeavyBall(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(HeavyBall, self).__init__(params, defaults)

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
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum = group['momentum']
                step_size = group['lr']
                buf = state['momentum_buffer']

                # Heavy Ball Momentum update
                buf.mul_(momentum).add_(grad)
                p.data = p.data - step_size * buf

        return loss
