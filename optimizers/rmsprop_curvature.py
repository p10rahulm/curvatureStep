import torch
from torch.optim import Optimizer

class RMSPropCurvature(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-8, weight_decay=0, epsilon=0.01):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        super(RMSPropCurvature, self).__init__(params, defaults)

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
                alpha = group['alpha']
                eps = group['eps']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']

                last_grad = state['last_grad']
                current_grad = grad

                if weight_decay != 0:
                    current_grad = current_grad.add(weight_decay, p.data)

                radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))
                normed_last_grad = last_grad / torch.norm(last_grad)
                grad = radius * normed_last_grad

                square_avg = state['square_avg']

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                avg = square_avg.sqrt().add_(eps)

                p.data.addcdiv_(grad, avg, value=-step_size)

        return loss
