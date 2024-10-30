import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deepsdf_modules.layers import ResnetBlockFC,CResnetBlockConv1d,CBatchNorm1d,CBatchNorm1d_legacy
from .vn_layers import VNLinear

class DecoderInner2(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''
    def __init__(self, z_dim=128, c_dim=128, o_dim=1,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)
        if c_dim > 0:
            self.c_in = VNLinear(c_dim, c_dim)

        self.fc_in = nn.Linear(z_dim*2+c_dim*2+1, hidden_size)
        self.block0 = CResnetBlockConv1d(hidden_size, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(hidden_size, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(hidden_size, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(hidden_size, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(hidden_size, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(hidden_size, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(hidden_size, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, o_dim, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)


    def forward(self, p, z, **kwargs):
        batch_size, T, D = p.size()

        net = (p * p).sum(2, keepdim=True)

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        net = self.fc_in(net).permute(0,2,1)

        net = self.block0(net, z)
        net = self.block1(net, z)
        net = self.block2(net, z)
        net = self.block3(net, z)
        net = self.block4(net, z)

        out = self.fc_out(self.actvn(self.bn(net, z)))
        out = out.squeeze(1)

        return out

class DecoderInner(nn.Module):
    ''' Decoder class.

    It does not perform any form of normalization.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, z_dim=128, c_dim=128, o_dim=1,
                 hidden_size=128, leaky=False, return_features=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)
        if c_dim > 0:
            self.c_in = VNLinear(c_dim, c_dim)

        self.fc_in = nn.Linear(z_dim*2+c_dim*2+1, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)
        self.return_features = return_features

        self.fc_out = nn.Linear(hidden_size, o_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()

        if isinstance(c, tuple):
            c, c_meta = c

        net = (p * p).sum(2, keepdim=True)

        if self.z_dim != 0:
            z = z.view(batch_size, -1, D).contiguous()
            net_z = torch.einsum('bmi,bni->bmn', p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        if self.c_dim != 0:
            c = c.view(batch_size, -1, D).contiguous()
            net_c = torch.einsum('bmi,bni->bmn', p, c)
            c_dir = self.c_in(c)
            c_inv = (c * c_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
            net = torch.cat([net, net_c, c_inv], dim=2)

        net = self.fc_in(net)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out
    
    def forward_ndf(self, p, z):
        ndf = []
        batch_size, T, D = p.size()
        net = (p * p).sum(2, keepdim=True)
        z = z.view(batch_size, -1, D).contiguous()
        net_z = torch.einsum('bmi,bni->bmn', p, z)
        z_dir = self.z_in(z)
        z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
        
        net = torch.cat([net, net_z, z_inv], dim=2)
        ndf.append(net)
        net = self.fc_in(net)
        ndf.append(net)
        net = self.block0(net)
        ndf.append(net)
        net = self.block1(net)
        ndf.append(net)
        net = self.block2(net)
        ndf.append(net)
        net = self.block3(net)
        ndf.append(net)
        net = self.block4(net)
        ndf.append(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        
        ndf = torch.cat(ndf,dim=-1)
        ndf = F.normalize(ndf, p=2, dim=-1)
        return ndf,out