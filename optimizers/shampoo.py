import torch
from torch.optim import Optimizer

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # Use CPU for SVD for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

class Shampoo(Optimizer):
    """
    Implements Shampoo algorithm.
    It has been proposed in https://arxiv.org/pdf/2002.09018
    Inspired by https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum=0.9, rho=0.9, update_freq=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum, rho=rho, update_freq=update_freq)
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
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                lr = group["lr"]
                rho = group["rho"]
                eps = group["eps"]
                update_freq = group["update_freq"]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['preconds'] = [group['eps'] * torch.eye(dim, device=p.data.device) for dim in grad.size()]
                    state['inv_preconds'] = [torch.zeros(dim, dim, device=p.data.device) for dim in grad.size()]

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update step count
                state['step'] += 1

                # Compute preconditioners
                for dim_id, dim in enumerate(grad.size()):
                    precond = state['preconds'][dim_id]
                    inv_precond = state['inv_preconds'][dim_id]

                    grad_reshaped = grad.transpose(0, dim_id).contiguous().view(dim, -1)
                    grad_gram = grad_reshaped @ grad_reshaped.t()
                    precond.mul_(rho).add_(grad_gram, alpha=1 - rho)

                    if state["step"] % update_freq == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / grad.ndimension()))

                    if dim_id == grad.ndimension() - 1:
                        grad = grad_reshaped.t() @ inv_precond
                        grad = grad.view_as(p.data)
                    else:
                        grad = inv_precond @ grad_reshaped
                        grad = grad.view_as(p.data.transpose(0, dim_id)).transpose(0, dim_id)

                state['momentum_buffer'] = grad
                p.data.add_(grad, alpha=-lr)

        return loss
