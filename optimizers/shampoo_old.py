import torch
from torch.optim import Optimizer

class Shampoo(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum=0.9, rho=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum, rho=rho)
        super(Shampoo, self).__init__(params, defaults)

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
                    state['grad_gram'] = torch.zeros_like(p.data)
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['preconditioner'] = torch.eye(p.data.size(0), device=p.data.device)

                grad_gram = state['grad_gram']
                momentum_buffer = state['momentum_buffer']
                preconditioner = state['preconditioner']

                step_size = group['lr']
                momentum = group['momentum']
                rho = group['rho']
                eps = group['eps']

                # Update step count
                state['step'] += 1

                # Compute the gradient gram matrix
                grad_gram.mul_(rho).addmm_(grad, grad.t(), beta=1-rho)

                # Preconditioner update using gradient gram matrix
                eigvals, eigvecs = torch.symeig(grad_gram + eps * torch.eye(grad_gram.size(0), device=grad_gram.device), eigenvectors=True)
                eigvals = eigvals.sqrt().diag_embed()
                preconditioner = eigvecs @ eigvals @ eigvecs.t()

                # Preconditioned gradient
                preconditioned_grad = torch.mm(preconditioner.inverse(), grad)

                # Apply momentum
                if momentum > 0:
                    momentum_buffer.mul_(momentum).add_(preconditioned_grad)
                    p.data.add_(-step_size, momentum_buffer)
                else:
                    p.data.add_(-step_size, preconditioned_grad)

        return loss