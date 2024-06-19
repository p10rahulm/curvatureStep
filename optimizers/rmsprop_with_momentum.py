import torch
from torch.optim import Optimizer


class RMSPropMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-6, weight_decay=0.01, momentum=0.05):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super(RMSPropMomentum, self).__init__(params, defaults)

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
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                eps = group['eps']
                momentum = group['momentum']
                step_size = group['lr']

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                avg = square_avg.sqrt().add_(eps)

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-step_size)
                else:
                    p.data.addcdiv_(grad, avg, value=-step_size)

        return loss
