import torch
import torch.nn as nn
import torch.nn.functional as F

from .dif_modules.loss import implicit_loss
from .vn_modules.encoder import VNN_ResnetPointnet
from .vn_modules.decoder import DecoderInner

class NDF(nn.Module):
    def __init__(self,latent_dim=256) -> None:
        super().__init__()
        self.encoder = VNN_ResnetPointnet(latent_dim)
        self.decoder = DecoderInner(z_dim=latent_dim, c_dim=0,hidden_size=latent_dim, leaky=True)

    # for training
    def forward(self,x):
        x['pts'] = x['pts'].requires_grad_(True)
        z = self.encoder(x['pcd'])
        pts = x['pts']#.requires_grad_(True)
        sdf = self.decoder(pts,z).unsqueeze(-1)
        normals = torch.autograd.grad(sdf, [pts], grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
        
        loss = implicit_loss(x,{
            'z':z,
            'sdf':sdf,
            'normals':normals
        })
        return loss
    
    # for generation
    def inference(self,pts,pcd=None,z=None,return_ndf=False):
        if z is None:
            z = self.encoder(pcd)
        ndf,sdf = self.decoder.forward_ndf(pts,z)
        out = {'z':z,'sdf':sdf}
        if return_ndf:
            out['ndf'] = ndf
        return out
