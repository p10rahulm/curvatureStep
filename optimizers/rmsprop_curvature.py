import torch
from torch.optim import Optimizer

class RMSPropCurvature(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-8, weight_decay=0, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        self.r_max = r_max
        super(RMSPropCurvature, self).__init__(params, defaults)

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
                    state['square_avg'] = torch.zeros_like(p.data)
                    
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'square_avg': state['square_avg'].clone()
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
            
        # Update with curvature and RMSProp
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
                
                # Restore square_avg state
                state['square_avg'].copy_(saved['square_avg'])
                
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
                
                # RMSProp update with curvature
                square_avg = state['square_avg']
                
                # Update running average of squared gradients
                square_avg.mul_(group['alpha']).addcmul_(grad_with_curvature, 
                                                        grad_with_curvature, 
                                                        value=1 - group['alpha'])
                
                # Compute denominator
                avg = square_avg.sqrt().add_(group['eps'])
                
                # Apply update
                p.data.addcdiv_(grad_with_curvature, avg, value=-group['lr'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss