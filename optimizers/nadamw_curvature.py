import torch
from torch.optim import Optimizer

class NAdamWCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        self.r_max = r_max
        super(NAdamWCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        # Store initial parameters and states
        saved_states = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'exp_avg': state['exp_avg'].clone(),
                    'exp_avg_sq': state['exp_avg_sq'].clone(),
                    'step': state['step']
                })
                
        # Move to tentative point w_t - ηg_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * p.grad.data
                
        # Get gradient at tentative point
        if closure is not None:
            closure()
            
        # Update with curvature and NAdamW
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise RuntimeError('NAdamW does not support sparse gradients')
                    
                state = self.state[p]
                saved = saved_states[param_idx]
                
                # Get saved states
                orig_data = saved['data']
                orig_grad = saved['grad']
                tentative_grad = p.grad.data
                
                # Restore moment states
                state['exp_avg'].copy_(saved['exp_avg'])
                state['exp_avg_sq'].copy_(saved['exp_avg_sq'])
                state['step'] = saved['step'] + 1
                
                # Get hyperparameters
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']
                
                # Compute radius of curvature
                grad_norm = torch.norm(orig_grad)
                grad_diff_norm = torch.norm(orig_grad - tentative_grad)
                r_t = grad_norm / (grad_diff_norm + group['epsilon'])
                r_t = torch.minimum(torch.tensor(self.r_max), r_t)
                
                # Compute curvature-adjusted gradient
                normed_grad = orig_grad / grad_norm
                grad_with_curvature = r_t * normed_grad
                
                # NAdamW update with curvature
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Compute NAdam-specific momentum cache
                momentum_cache_t = beta1 * (1.0 - 0.5 * (0.96 ** (state['step'] * 0.004)))
                momentum_cache_t_1 = beta1 * (1.0 - 0.5 * (0.96 ** ((state['step'] + 1) * 0.004)))
                
                # Update biased first moment estimate with momentum cache
                exp_avg.mul_(momentum_cache_t).add_(grad_with_curvature, alpha=1 - momentum_cache_t)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_with_curvature, 
                                              grad_with_curvature, 
                                              value=1 - beta2)
                
                # Compute bias corrections with momentum cache
                bias_correction1 = 1 - momentum_cache_t_1
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute adaptive learning rate
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply decoupled weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * group['lr'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss