import torch
from torch.autograd import backward,grad

import torch

def zero_grad(params):
    for name in params:
        param = params[name]
        if param.grad is not None:
            param.grad.data.zero_()

def reset_grad(params,grads):
    for name in grads:
        param = params[name]
        if param.grad is not None:
            param.grad.data = grads[name]

def compute_grad_backward(losses,params):
    grads = {}
    ntask = len(losses)
    for i_task,task in enumerate(losses):
        loss = losses[task]
        loss.backward(retain_graph=True if i_task!=ntask-1 else False)
        grads[task] = {}
        for name in params:
            param = params[name]
            if param.grad is not None:
                grads[task][name] = param.grad.clone()
        zero_grad(params)
    return grads

def compute_grad_autograd(losses,params):
    grads = {}
    keys = []
    values = []
    for key,value in params.items():
        if value.requires_grad:
            keys.append(key)
            values.append(value)
    for task in losses:
        loss = losses[task]
        grads_ = torch.autograd.grad(loss, values, retain_graph=True)
        grads[task] = {}
        for i,key in enumerate(keys):
            grads[task][key] = grads_[i]
    return grads

def grads_clip(losses, params,clip_value=0.01,clip=True,norm=False):
    grads = compute_grad_backward(losses,params)
    for key in params:
        param = params[key]
        if not param.requires_grad: continue
        grad = []
        for task in losses:
            
            g = grads[task][key]
            # if key=='transl' and task=='v': g = torch.zeros_like(g)
            # elif key=='body_pose'
            # else:
            # if key=='transl' and task

            if clip: 
                if norm:
                    n = g.norm(keepdim=True)
                    g = g*n.clip(max=clip_value*g.numel()**(0.5))/(n+1e-12)
                else:
                    g = g.clip(min=-clip_value,max=clip_value)
            grad.append(g)
        if len(grad)==1:
            grad = grad[0]
        else:
            grad = torch.stack(grad,dim=0).sum(0)
        param.grad.data = grad
    
class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        with torch.no_grad():
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e)#, eigenvectors=True)
            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )
            rank = sum(singulars > tol)

            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            if scale_mode == 'min':
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                weights = basis * torch.sqrt(singulars.mean())

            weights = weights / torch.sqrt(singulars).view(1, -1)
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            return grads, weights, singulars

def get_G_wrt_shared(losses, shared_params, update_decoder_grads=False):
    grads = []
    for task_id in losses:
        cur_loss = losses[task_id]
        if not update_decoder_grads:
            grad = torch.cat([p.flatten() if p is not None else torch.zeros_like(shared_params[i]).flatten()
                                for i, p in enumerate(torch.autograd.grad(cur_loss, shared_params,
                                                            retain_graph=True, allow_unused=True))])
        else:
            for p in shared_params:
                if p.grad is not None:
                    p.grad.data.zero_()

            cur_loss.backward(retain_graph=True)
            grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                                for p in shared_params])

        grads.append(grad)

    for p in shared_params:
        if p.grad is not None:
            p.grad.data.zero_()

    return torch.stack(grads, dim=0)

def set_shared_grad(shared_params, grad_vec):
    offset = 0
    for p in shared_params:
        if p.grad is None:
            continue
        _offset = offset + p.grad.shape.numel()
        p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
        offset = _offset

def aligned_mtl(losses, shared_params, scale_mode='min'):
    grads = get_G_wrt_shared(losses, shared_params)
    grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), scale_mode)
    grad = grads[0].sum(-1)
    set_shared_grad(shared_params,grad)

def aligned_mtl_clip(losses, shared_params, clip_value=0.01, scale_mode='min'):
    grads = get_G_wrt_shared(losses, shared_params)
    grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), scale_mode)
    grads = grads.clip(min=-clip_value,max=clip_value)
    grad = grads[0].sum(-1)
    set_shared_grad(shared_params,grad)