import numpy as np
import torch

################################################################################
# pcd rotation (right multiply)
################################################################################
def rotate_np(pcd,R):  return np.einsum('ij,jk->ik',pcd,R)

def rotate_pt(pcd,R): return torch.einsum('ij,jk->ik', pcd, R)

################################################################################
# coordinate conversion
################################################################################
def spherical_from_cartesian(xyz):
    u = torch.arctan2(xyz[...,:2].norm(dim=-1),xyz[...,2])
    v = torch.arctan2(xyz[...,1],xyz[...,0])
    uv = torch.stack([u,v],dim=-1)
    return uv

def cartesian_from_spherical(uv):
    x = torch.sin(uv[...,1])*torch.cos(uv[...,0])
    y = torch.sin(uv[...,1])*torch.sin(uv[...,0])
    z = torch.cos(uv[...,1])
    xyz = torch.stack([x,y,z],dim=-1)
    return xyz

def rotate_pole(xyz,uv):
    x_ = xyz[...,0]*torch.cos(uv[...,1])+xyz[...,2]*torch.sin(uv[...,1])
    y_ = xyz[...,1]
    z_ = -xyz[...,0]*torch.sin(uv[...,1])+xyz[...,2]*torch.cos(uv[...,1])
    x = x_*torch.cos(uv[...,0])-y_*torch.sin(uv[...,0])
    y = x_*torch.sin(uv[...,0])+y_*torch.cos(uv[...,0])
    z = z_
    xyz = torch.stack([x,y,z],dim=-1)
    return xyz
