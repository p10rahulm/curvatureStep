import torch
from torch.optim import Optimizer

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # Use CPU for SVD for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

class ShampooCurvature(Optimizer):
    """
    Implements Shampoo algorithm with curvature step.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, momentum=0.1, rho=0.9, update_freq=1, 
                 epsilon=1e-8, r_max=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       momentum=momentum, rho=rho, update_freq=update_freq, epsilon=epsilon)
        self.r_max = r_max
        super(ShampooCurvature, self).__init__(params, defaults)

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
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['preconds'] = [group['eps'] * torch.eye(dim, device=p.data.device) 
                                       for dim in p.data.size()]
                    state['inv_preconds'] = [torch.zeros(dim, dim, device=p.data.device) 
                                           for dim in p.data.size()]
                
                saved_states.append({
                    'data': p.data.clone(),
                    'grad': p.grad.data.clone(),
                    'momentum_buffer': state['momentum_buffer'].clone(),
                    'preconds': [precond.clone() for precond in state['preconds']],
                    'inv_preconds': [inv_precond.clone() for inv_precond in state['inv_preconds']],
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
            
        # Update with curvature and Shampoo
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
                state['momentum_buffer'].copy_(saved['momentum_buffer'])
                for i, (precond, inv_precond) in enumerate(zip(saved['preconds'], saved['inv_preconds'])):
                    state['preconds'][i].copy_(precond)
                    state['inv_preconds'][i].copy_(inv_precond)
                state['step'] = saved['step'] + 1
                
                # Compute radius of curvature
                grad_norm = torch.norm(orig_grad)
                grad_diff_norm = torch.norm(orig_grad - tentative_grad)
                r_t = grad_norm / (grad_diff_norm + group['epsilon'])
                r_t = torch.minimum(torch.tensor(self.r_max), r_t)
                
                # Compute curvature-adjusted gradient
                normed_grad = orig_grad / grad_norm
                grad_with_curvature = r_t * normed_grad
                
                # Apply momentum if specified
                if group['momentum'] > 0:
                    grad_with_curvature.mul_(1 - group['momentum']).add_(
                        state['momentum_buffer'], 
                        alpha=group['momentum']
                    )
                
                # Apply weight decay if specified
                if group['weight_decay'] > 0:
                    grad_with_curvature.add_(orig_data, alpha=group['weight_decay'])
                
                # Compute preconditioners
                for dim_id, dim in enumerate(grad_with_curvature.size()):
                    precond = state['preconds'][dim_id]
                    inv_precond = state['inv_preconds'][dim_id]
                    
                    grad_reshaped = grad_with_curvature.transpose(0, dim_id).contiguous().view(dim, -1)
                    grad_gram = grad_reshaped @ grad_reshaped.t()
                    precond.mul_(group['rho']).add_(grad_gram, alpha=1 - group['rho'])
                    
                    if state['step'] % group['update_freq'] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / grad_with_curvature.ndimension()))
                    
                    if dim_id == grad_with_curvature.ndimension() - 1:
                        grad_with_curvature = grad_reshaped.t() @ inv_precond
                        grad_with_curvature = grad_with_curvature.view_as(p.data)
                    else:
                        grad_with_curvature = inv_precond @ grad_reshaped
                        grad_with_curvature = grad_with_curvature.view_as(
                            p.data.transpose(0, dim_id)
                        ).transpose(0, dim_id)
                
                # Store momentum state
                state['momentum_buffer'] = grad_with_curvature
                
                # Apply update
                p.data = orig_data
                p.data.add_(grad_with_curvature, alpha=-group['lr'])
                
                # Restore original gradient
                p.grad.data = orig_grad
                param_idx += 1
                
        return loss