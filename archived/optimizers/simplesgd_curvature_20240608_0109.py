import torch
from torch.optim import Optimizer


class SimpleSGDCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, momentum_mult=0.9, epsilon=0.01):
        defaults = dict(lr=lr, momentum_mult=momentum_mult, epsilon=epsilon)
        super(SimpleSGDCurvature, self).__init__(params, defaults)

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

                last_grad = state['last_grad']
                current_grad = grad

                # radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                #                                                torch.norm(current_grad - last_grad)) # 95.30
                radius = torch.norm(current_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))  # 95.27
                # radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                #                                                   torch.norm(current_grad - last_grad))  # 95.27
                # radius = torch.norm(current_grad - last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                #                                                               torch.norm(last_grad))
                normed_curr_grad = current_grad / torch.norm(current_grad)
                normed_last_grad = last_grad / torch.norm(last_grad)
                # p.data = p.data - step_size * last_grad - step_size * radius * normed_curr_grad #95.27
                p.data = p.data - step_size * radius * normed_curr_grad # 95.37
                # p.data = p.data - step_size * radius * normed_last_grad  # 95.37 also!

        return loss

