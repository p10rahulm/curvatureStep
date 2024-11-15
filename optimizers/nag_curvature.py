
import torch
from torch.optim import Optimizer

class NAGCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.55, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, momentum=momentum, epsilon=epsilon)
        self.r_max = r_max
        super(NAGCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        # Store initial parameters, gradients, and momentum buffers
        saved_states = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'momentum_buffer': state['momentum_buffer'].clone()
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
            
        # Update with curvature and NAG
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
                
                # Restore momentum buffer
                buf = state['momentum_buffer']
                buf.copy_(saved['momentum_buffer'])
                
                # Compute radius of curvature
                grad_norm = torch.norm(orig_grad)
                grad_diff_norm = torch.norm(orig_grad - tentative_grad)
                r_t = grad_norm / (grad_diff_norm + group['epsilon'])
                r_t = torch.minimum(torch.tensor(self.r_max), r_t)
                
                # Compute curvature-adjusted gradient
                normed_grad = orig_grad / grad_norm
                grad_with_curvature = r_t * normed_grad
                
                # NAG update with curvature
                prev_buf = buf.clone()
                buf.mul_(group['momentum']).add_(grad_with_curvature)
                
                # Restore data and apply NAG update
                p.data = orig_data
                p.data.add_(grad_with_curvature, alpha=-group['lr'])
                p.data.add_(prev_buf, alpha=-group['lr'] * group['momentum'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss