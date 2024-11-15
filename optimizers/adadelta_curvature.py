import torch
from torch.optim import Optimizer

class AdadeltaCurvature(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.95, eps=1e-6, weight_decay=0, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay, epsilon=epsilon)
        self.r_max = r_max
        super(AdadeltaCurvature, self).__init__(params, defaults)

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
                    state['acc_delta'] = torch.zeros_like(p.data)
                    
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'square_avg': state['square_avg'].clone(),
                    'acc_delta': state['acc_delta'].clone()
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
            
        # Update with curvature and Adadelta
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
                
                # Restore accumulator states
                state['square_avg'].copy_(saved['square_avg'])
                state['acc_delta'].copy_(saved['acc_delta'])
                
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
                
                # Adadelta update with curvature
                square_avg = state['square_avg']
                acc_delta = state['acc_delta']
                
                # Update running average of squared gradients
                square_avg.mul_(group['rho']).addcmul_(grad_with_curvature, 
                                                      grad_with_curvature, 
                                                      value=1 - group['rho'])
                
                # Compute update
                std = square_avg.add(group['eps']).sqrt_()
                delta = acc_delta.add(group['eps']).sqrt_()
                delta.div_(std).mul_(grad_with_curvature)
                
                # Apply update
                p.data.add_(delta, alpha=-group['lr'])
                
                # Update running average of squared updates
                acc_delta.mul_(group['rho']).addcmul_(delta, delta, value=1 - group['rho'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss