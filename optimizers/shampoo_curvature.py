import torch
from torch.optim import Optimizer

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # Use CPU for SVD for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

class ShampooCurvature(Optimizer):
    """
    Implements Shampoo algorithm with curvature step.
    It has been proposed in https://arxiv.org/pdf/2002.09018
    Inspired by https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum=0.1, rho=0.9, update_freq=1, epsilon=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum=momentum, rho=rho, update_freq=update_freq, epsilon=epsilon)
        super(ShampooCurvature, self).__init__(params, defaults)

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
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['preconds'] = [group['eps'] * torch.eye(dim, device=p.data.device) for dim in p.data.size()]
                    state['inv_preconds'] = [torch.zeros(dim, dim, device=p.data.device) for dim in p.data.size()]

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
                rho = group['rho']
                eps = group['eps']
                update_freq = group['update_freq']
                momentum = group['momentum']
                epsilon = group['epsilon']

                last_grad = state['last_grad']
                current_grad = grad

                radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))
                normed_last_grad = last_grad / torch.norm(last_grad)
                grad = radius * normed_last_grad

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if group['weight_decay'] > 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

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
                p.data.add_(grad, alpha=-step_size)

        return loss
