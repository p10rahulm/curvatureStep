import torch
from torch.optim import Optimizer

class RMSPropMomentumCurvature(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0.05, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, 
                       momentum=momentum, epsilon=epsilon)
        self.r_max = r_max
        super(RMSPropMomentumCurvature, self).__init__(params, defaults)

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
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'square_avg': state['square_avg'].clone(),
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
            
        # Update with curvature and RMSProp with momentum
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
                
                # Restore states
                state['square_avg'].copy_(saved['square_avg'])
                state['momentum_buffer'].copy_(saved['momentum_buffer'])
                
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
                
                # RMSProp with momentum update with curvature
                square_avg = state['square_avg']
                momentum_buffer = state['momentum_buffer']
                
                # Update running average of squared gradients
                square_avg.mul_(group['alpha']).addcmul_(grad_with_curvature, 
                                                        grad_with_curvature, 
                                                        value=1 - group['alpha'])
                
                # Compute denominator
                avg = square_avg.sqrt().add_(group['eps'])
                
                # Apply momentum update
                if group['momentum'] > 0:
                    momentum_buffer.mul_(group['momentum']).addcdiv_(grad_with_curvature, avg)
                    
                    # Apply update with momentum
                    p.data = orig_data
                    p.data.add_(momentum_buffer, alpha=-group['lr'])
                else:
                    # Apply regular RMSProp update if momentum is 0
                    p.data = orig_data
                    p.data.addcdiv_(grad_with_curvature, avg, value=-group['lr'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss