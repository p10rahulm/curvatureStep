import torch
from torch.optim import Optimizer

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

class Shampoo(Optimizer):
    """
    Implements Shampoo algorithm.
    Inspired by https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py
    """
    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0, epsilon=1e-4, update_freq=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if update_freq < 1:
            raise ValueError("Invalid update_freq value: {}".format(update_freq))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state["precond_{}".format(dim_id)] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state["inv_precond_{}".format(dim_id)] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)

                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state["precond_{}".format(dim_id)]
                    inv_precond = state["inv_precond_{}".format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
