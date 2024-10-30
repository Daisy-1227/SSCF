import torch
import torch.nn as nn
import torch.nn.functional as F

from .vn_modules.encoder import REncoder
from .dif_modules.decoder import DIFDecoder

class VNNDIF(nn.Module):
    def __init__(self,latent_dim=256,pred_R=True,loss_R=False) -> None:
        super().__init__()
        self.encoder = REncoder(latent_dim)
        self.decoder = DIFDecoder(latent_dim*3)
        self.pred_R = pred_R
        self.loss_R = loss_R
    
    # for training
    def forward(self,x):
        out_enc = self.encoder(x['pcd'])
        z,R = out_enc['z'],out_enc['R']
        pts = torch.einsum('bij,bjk->bik', x['pts'], R if self.pred_R else x['R'])
        loss = self.decoder({'embedding':z,'coords':pts},x)
        if self.loss_R:
            loss['R'] = F.mse_loss(z['R'],x['R'])
        return loss

    # for generation
    def inference(self,pts,pcd=None,z=None,return_sdf=True,template=False):
        if template:
            out = self.decoder.inference(pts,return_sdf=True,template=True)
        else:
            if z is None:
                z = self.encoder(pcd)
            pts_canonical = torch.einsum('bij,bjk->bik', pts, z['R'])
            out = self.decoder.inference(pts_canonical,z['z'],return_sdf=return_sdf,template=False)
            out['z'] = z
            out['pts_canonical'] = pts_canonical
        return out