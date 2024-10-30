import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim,sdf=False):
        super(generator, self).__init__()
        self.sdf = sdf
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias,0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias,0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias,0)
    
    def forward(self, points, z, is_training=False):
        zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
        pointz = torch.cat([points,zs],2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        #l7 = torch.clamp(l7, min=0, max=1)
        if not self.sdf:
            l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)

        return l7.squeeze(2)

class DecoderCBatchNorm(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        # self.z_dim = z_dim
        # if not z_dim == 0:
        #     self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        # if self.z_dim != 0:
        #     net_z = self.fc_z(z).unsqueeze(2)
        #     net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
    
    def forward_ndf(self, p, c, **kwargs):
        ndf = []
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)
        ndf.append(net)

        net = self.block0(net, c)
        ndf.append(net)
        net = self.block1(net, c)
        ndf.append(net)
        net = self.block2(net, c)
        ndf.append(net)
        net = self.block3(net, c)
        ndf.append(net)
        net = self.block4(net, c)
        ndf.append(net)
        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        ndf = torch.cat(ndf,dim=1).transpose(1,2)
        ndf = F.normalize(ndf, p=2, dim=-1)
        return ndf,out