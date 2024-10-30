import torch
import trimesh
from trimesh import nsphere

def get_loc(pcd):
    pcd_max = torch.max(pcd, dim=0)[0]
    pcd_min = torch.min(pcd, dim=0)[0]
    pcd_center = (pcd_max + pcd_min) / 2
    return pcd_center

def get_scale(pcd):
    pcd_max = torch.max(pcd, dim=0)[0]
    pcd_min = torch.min(pcd, dim=0)[0]
    pcd_scale = (pcd_max - pcd_min).norm(p=2,dim=-1).squeeze()
    return pcd_scale

# def get_loc_scale(pcd):
#     # box = trimesh.PointCloud(pcd.cpu().numpy()).bounding_box_oriented
#     center,scale_half = nsphere.minimum_nsphere(pcd.cpu().numpy())
#     # center = box.centroid
#     # scale = box.scale
#     pcd_center = torch.tensor(center).to(pcd)
#     pcd_scale = torch.tensor(scale_half*2).to(pcd)
#     return pcd_center,pcd_scale

def get_loc_scale(pcd):
    pcd_center = pcd.mean(0)
    pcd_max = torch.max(pcd, dim=0)[0]
    pcd_min = torch.min(pcd, dim=0)[0]
    pcd_scale = (pcd_max - pcd_min).norm(p=2,dim=-1).squeeze()
    return pcd_center,pcd_scale
