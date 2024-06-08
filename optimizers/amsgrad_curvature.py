import torch
from torch.optim import Optimizer

class AMSGradCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, epsilon=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=True, epsilon=epsilon)
        super(AMSGradCurvature, self).__init__(params, defaults)

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
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

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
                    raise RuntimeError('AMSGrad does not support sparse gradients')

                state = self.state[p]

                state['step'] += 1
                step_size = group['lr']
                beta1, beta2 = group['betas']
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

                exp_avg, exp_avg_sq, max_exp_avg_sq = state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Maintain the maximum of all 2nd moment running average till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                denom = max_exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
