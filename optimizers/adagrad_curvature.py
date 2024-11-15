import torch
from torch.optim import Optimizer

class AdagradCurvature(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        self.r_max = r_max
        super(AdagradCurvature, self).__init__(params, defaults)

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
                    state['sum'] = torch.zeros_like(p.data)
                    
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'sum': state['sum'].clone()
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
            
        # Update with curvature and Adagrad
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                saved = saved_states[param_idx]
                
                # Get saved states
                orig_data = saved['data']
                orig_grad = saved['grad']
                tentative_grad = p.grad.data
                
                # Restore sum state
                state['sum'].copy_(saved['sum'])
                
                # Compute radius of curvature
                grad_norm = torch.norm(orig_grad)
                grad_diff_norm = torch.norm(orig_grad - tentative_grad)
                r_t = grad_norm / (grad_diff_norm + group['epsilon'])
                r_t = torch.minimum(torch.tensor(self.r_max), r_t)
                
                # Compute curvature-adjusted gradient
                normed_grad = orig_grad / grad_norm
                grad_with_curvature = r_t * normed_grad
                
                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    grad_with_curvature.add_(orig_data, alpha=group['weight_decay'])
                
                # Adagrad update with curvature
                sum_ = state['sum']
                eps = group['eps']
                
                # Update accumulated squared gradients
                sum_.addcmul_(grad_with_curvature, grad_with_curvature)
                
                # Compute update
                std = sum_.sqrt().add_(eps)
                
                # Apply update
                p.data = orig_data
                p.data.addcdiv_(grad_with_curvature, std, value=-group['lr'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss