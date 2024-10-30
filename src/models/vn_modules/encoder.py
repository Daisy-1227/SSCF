import torch
import torch.nn as nn
import torch.nn.functional as F

from .pc_utils import get_graph_feature_cross,get_graph_feature_local
from .vn_layers import VNLinear,VNLeakyReLU,VNLinearLeakyReLU,VNResnetBlockFC,VNStdFeature

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None,last_pool=True,feat_fn='cross'):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.last_pool = last_pool
        self.feat_fn = feat_fn

        self.conv_pos = VNLinearLeakyReLU(3 if feat_fn=='cross' else 1, 128, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.2, share_nonlinearity=False)
        self.pool = meanpool

        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        # feat = get_graph_feature_cross(p, k=self.k)
        if self.feat_fn=='cross':
            feat = get_graph_feature_cross(p,k=self.k)
        else:
            feat = get_graph_feature_local(p,k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        if self.last_pool:
            net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c

class VNNEncoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VNN_ResnetPointnet(c_dim=latent_dim,last_pool=True) # modified resnet-18

    def forward(self, input):
        enc_in = input['pcd']
        B,N,_ = enc_in.shape

        z = self.encoder(enc_in)
        out_dict = {
            'z': z
        }
        return out_dict

class REncoder(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VNN_ResnetPointnet(c_dim=latent_dim,last_pool=True) # modified resnet-18
        self.std_feature = VNStdFeature(latent_dim, dim=3, normalize_frame=True, use_batchnorm=False)
    
    def forward(self, pcd):
        B,N,_ = pcd.shape

        z = self.encoder(pcd)
        z_std,R = self.std_feature(z)
        z_std = z_std.reshape(B,-1)

        out_dict = {
            'R':R,
            'z': z_std
        }
        return out_dict