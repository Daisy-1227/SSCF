import torch
import torch.nn as nn
import torch.nn.functional as F

# from pytorch3d.ops import knn_points
# def knn(x,k):
#     x = x.permute(0,2,1)
#     nn = knn_points(x,x,K=k)
#     return nn.idx

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is not None: # dynamic knn graph
            idx = knn(x_coord, k=k)   # (batch_size, num_points, k)
        else:             # fixed knn graph with input point coordinates
            idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # device = torch.device('cuda')

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = torch.arange(0, batch_size).to(x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_local(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    bias = torch.mean(x, dim=-1, keepdim=True)
    x = x - bias
    
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # device = torch.device('cuda')

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = torch.arange(0, batch_size).to(x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_mean(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3).mean(2, keepdim=False)
    x = x.view(batch_size, num_points, num_dims, 3)
    
    feature = (feature-x).permute(0, 2, 3, 1).contiguous()
  
    return feature


def get_shell_mean_cross(x, k=10, nk=4, idx_all=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points).contiguous()
    if idx_all is None:
        idx_all = knn(x, k=nk*k)   # (batch_size, num_points, k)
    device = torch.device('cuda')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    
    idx = []
    for i in range(nk):
        idx.append(idx_all[:, :, i*k:(i+1)*k])
        idx[i] = idx[i] + idx_base
        idx[i] = idx[i].view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.view(batch_size, num_points, num_dims, 3)
    feature = []
    for i in range(nk):
        feature.append(x.view(batch_size*num_points, -1)[idx[i], :])
        feature[i] = feature[i].view(batch_size, num_points, k, num_dims, 3).mean(2, keepdim=False)
        feature[i] = feature[i] - x
        cross = torch.cross(feature[i], x, dim=3)
        feature[i] = torch.cat((feature[i], cross), dim=2)
    
    feature = torch.cat(feature, dim=2).permute(0, 2, 3, 1).contiguous()
  
    return feature