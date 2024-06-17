import torch
from torch.optim import Optimizer


class CustomOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, momentum_mult=0.9, epsilon=0.001):
        defaults = dict(lr=lr, momentum_mult=momentum_mult, epsilon=epsilon)
        super(CustomOptimizer, self).__init__(params, defaults)

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
                radius2 = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                                  torch.norm(current_grad - last_grad))  # 95.27
                # radius = torch.norm(current_grad - last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                #                                                               torch.norm(last_grad))
                normed_curr_grad = current_grad / torch.norm(current_grad)
                normed_last_grad = last_grad / torch.norm(last_grad)
                # p.data = p.data - step_size * last_grad - step_size * radius * normed_curr_grad #95.27
                # p.data = p.data - step_size * radius * normed_curr_grad # 95.37
                p.data = p.data - step_size * radius2 * normed_last_grad  # 95.37 also!

        return loss


class CustomOptimizerOld(Optimizer):
    def __init__(self, params, lr=1e-3, momentum_mult=0.9, epsilon=1e-6):
        defaults = dict(lr=lr, momentum_mult=momentum_mult, epsilon=epsilon)
        super(CustomOptimizer, self).__init__(params, defaults)

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
                    state['step'] = 0
                    state['current_iterate'] = torch.clone(p.data).detach()
                    state['last_grad'] = torch.zeros_like(p.data)

                state['step'] += 1
                step_size = group['lr']
                epsilon = group['epsilon']

                if state['step'] % 2 == 1:  # Odd step, simply compute SGD and store last_grad
                    state['last_grad'] = grad
                    p.data = p.data - step_size * grad


                else:  # Even step, compute iterate1 and iterate2 using stored and current grad
                    current_grad = grad
                    last_grad = state['last_grad']
                    radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                                   torch.norm(current_grad - last_grad))  # 81.92 best

                    # radius = torch.linalg.vector_norm(current_grad - last_grad) / torch.maximum(
                    #     torch.tensor(epsilon, device=grad.device), torch.linalg.vector_norm(last_grad)) #83.18 best

                    # print(f"radius = {step_size * radius}")
                    # p.data = p.data - step_size * radius * last_grad  # 81.12 accuracy
                    p.data = (
                                         p.data - step_size * last_grad) - step_size * radius * current_grad  # 80.95 wo odd 84.78 with odd
                    # p.data = p.data - step_size * radius * current_grad # 80.92 accuracy
                    # state['last_grad'] = grad
                    # state['current_iterate'] = p.data

        return loss
