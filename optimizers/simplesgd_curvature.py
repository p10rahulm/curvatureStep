import torch
from torch.optim import Optimizer

class SimpleSGDCurvature(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, epsilon=epsilon)
        self.r_max = r_max
        super(SimpleSGDCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        # Store initial parameters and gradients
        saved_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                saved_params.append((p.data.clone(), p.grad.data.clone()))
                
        # Move to tentative point w_t - ηg_t
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * p.grad.data
                
        # Get gradient at tentative point
        if closure is not None:
            closure()
            
        # Update with curvature
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                orig_data, orig_grad = saved_params[param_idx]
                tentative_grad = p.grad.data
                
                # Compute radius of curvature
                grad_norm = torch.norm(orig_grad)
                grad_diff_norm = torch.norm(orig_grad - tentative_grad)
                r_t = grad_norm / (grad_diff_norm + group['epsilon'])
                r_t = torch.minimum(torch.tensor(self.r_max), r_t)
                
                # SGD update with curvature
                p.data = p.data - group['lr'] * r_t * (orig_grad / grad_norm)
                p.grad.data = orig_grad  # Restore original gradient
                param_idx += 1
                
        return loss