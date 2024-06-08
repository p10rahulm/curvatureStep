import torch
from torch.optim import Optimizer

class NAG(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(NAG, self).__init__(params, defaults)

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
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum = group['momentum']
                step_size = group['lr']
                buf = state['momentum_buffer']

                # Nesterov accelerated gradient update
                prev_buf = buf.clone()
                buf.mul_(momentum).add_(grad)
                p.data.add_(-step_size, grad.add(momentum, prev_buf))
                p.data.add_(grad, alpha=-step_size).add_(prev_buf, alpha=-step_size * momentum)


        return loss
