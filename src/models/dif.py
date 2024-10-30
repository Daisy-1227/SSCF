import torch
import torch.nn as nn
import torch.nn.functional as F

from .deepsdf_modules.encoder import ResnetPointnet,RResnetPointnet
from .dif_modules.decoder import DIFDecoder

class DIF(nn.Module):
    def __init__(self,latent_dim=256) -> None:
        super().__init__()
        self.encoder = ResnetPointnet(c_dim=latent_dim*3,hidden_dim=latent_dim)
        self.decoder = DIFDecoder(latent_dim*3)

    # for training
    def forward(self,x):
        z = self.encoder(x['pcd'])
        loss = self.decoder({'embedding':z,'coords':x['pts']},x)
        return loss

    # for generation
    def inference(self,pts,pcd=None,z=None,return_sdf=True,template=False):
        if template:
            out = self.decoder.inference(pts,return_sdf=True,template=True)
        else:
            if z is None:
                z = self.encoder(pcd)
            out = self.decoder.inference(pts,z,return_sdf=return_sdf,template=False)
            out['z'] = z
        return out


class RDIF(nn.Module):
    def __init__(self,latent_dim=256,pred_R=True,loss_R=False) -> None:
        super().__init__()
        self.encoder = RResnetPointnet(c_dim=latent_dim*3,hidden_dim=latent_dim)
        self.decoder = DIFDecoder(latent_dim*3)
        self.pred_R = pred_R
        self.loss_R = loss_R
        
    # for training
    def forward(self,x):
        out_enc = self.encoder(x['pcd'])
        z,R = out_enc['z'],out_enc['R']
        pts = torch.einsum('bij,bjk->bik', x['pts'], R if self.pred_R else x['R'])
        loss = self.decoder({'embedding':z,'coords':pts},x)
        if not self.pred_R:
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