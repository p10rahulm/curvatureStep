import torch
from torch.optim import Optimizer

class NAdamWCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, epsilon=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        super(NAdamWCurvature, self).__init__(params, defaults)

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
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

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
                if grad.is_sparse:
                    raise RuntimeError('NAdamW does not support sparse gradients')

                state = self.state[p]

                state['step'] += 1
                step_size = group['lr']
                beta1, beta2 = group['betas']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']

                last_grad = state['last_grad']
                current_grad = grad

                radius = torch.norm(last_grad) / torch.maximum(torch.tensor(epsilon, device=grad.device),
                                                               torch.norm(current_grad - last_grad))
                normed_last_grad = last_grad / torch.norm(last_grad)
                grad = radius * normed_last_grad

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                momentum_cache_t = beta1 * (1.0 - 0.5 * (0.96 ** (state['step'] * 0.004)))
                momentum_cache_t_1 = beta1 * (1.0 - 0.5 * (0.96 ** ((state['step'] + 1) * 0.004)))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(momentum_cache_t).add_(grad, alpha=1 - momentum_cache_t)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - momentum_cache_t_1
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * step_size)

        return loss
