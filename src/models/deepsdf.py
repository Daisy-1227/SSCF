import torch
import torch.nn as nn
import torch.nn.functional as F

from .deepsdf_modules.encoder import ResnetPointnet
from .deepsdf_modules.decoder import DecoderCBatchNorm
from .dif_modules.loss import implicit_loss


class DeepSDF(nn.Module):
    def __init__(self,latent_dim=512) -> None:
        super().__init__()
        self.encoder = ResnetPointnet(c_dim=latent_dim,hidden_dim=latent_dim)
        self.decoder = DecoderCBatchNorm(z_dim=0,c_dim=latent_dim,hidden_size=latent_dim)
    
    def forward(self,x):
        x['pts'] = x['pts'].requires_grad_(True)

        z = self.encoder(x['pcd'])
        pts = x['pts']
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
