import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from scipy.spatial import cKDTree as KDTree

from smplx.joint_names import JOINT_NAMES

import trimesh
from trimesh import creation
import point_cloud_utils as pcu

from tqdm import tqdm

from ...models import get_model
from ...human.hand import load_mano,forward_mano
from ...utils.array_utils import to_np,to_pt
from ...utils.rot_utils import rotate_np,rotate_pt
from ...utils.pcd_utils import get_loc,get_scale,get_loc_scale
from ...utils.plot_utils import vis_loss
from ...utils.sdf_utils import sdf2mesh


class CPDPCA(nn.Module):
    def __init__(self,cpdpca_path,npcd=2048) -> None:
        super().__init__()
        self.npcd = npcd
        pt = torch.load(cpdpca_path)
        self.n_components = pt['n_components']
        self.register_buffer('mean',pt['mean'])
        self.register_buffer('components',pt['components'])
        self.register_buffer('explained_variance',pt['explained_variance'])
        self.register_buffer('C',pt['C'])
        self.register_buffer('G',pt['G'])

    def forward(self,theta):
        W = torch.matmul(theta, torch.sqrt(self.explained_variance[:, None])*self.components)+self.mean
        T = self.C+self.G@W.reshape(self.npcd,3)
        return T
    
    def register(self,O_src):
        theta = torch.zeros(self.n_components).float().to(O_src)
        theta.requires_grad = True
        optimizer = torch.optim.Adam([theta], lr=0.1)
        for i in range(250):
            optimizer.zero_grad()
            O_tgt = self(theta)
            loss = chamfer_distance(O_src[None],O_tgt[None])[0]
            loss.backward()
            optimizer.step()
        return O_tgt.detach(),theta.detach()
    

class LSNRROptimizer():
    def __init__(self,cpdpca_path,vnndif_path=None, debug=False):
        self.debug=debug

        pca = CPDPCA(cpdpca_path)
        pca = pca.cuda()
        pca.eval()
        self.pca = pca 


        if vnndif_path is not None:
            R_model = get_model('vnndif',vnndif_path)
            R_model = R_model.cuda()
            R_model.eval()
            self.R_model = R_model
        else:
            self.R_model = None

        H_model = load_mano()
        H_model = H_model.cuda()
        H_model.eval()
        self.H_model = H_model
        self.H_f = torch.tensor(self.H_model.get_mano_closed_faces()).long().cuda()
        # 固定这些关节角度不变
        self.H_seg = self.H_model.th_weights.argmax(dim=1).cpu().numpy()
        self.H_jnum = self.H_seg.max()+1

    def transfer(self,H_src_param,O_src,O_tgt):
        '''_summary_

        :param dict H_src: _description_
        :param tensor O_src: _description_
        :param tensor O_tgt: _description_
        '''
        # 保存的结果
        out_dict = {}

        ########################################################################
        # 将源物体、源人体对齐到目标物体
        ########################################################################

        # 归一化源物体
        O_src = O_src.float().cuda()
        O_src_loc, O_src_scale = get_loc_scale(O_src)
        O_src_norm_ = (O_src - O_src_loc) / O_src_scale

        # 归一化目标物体
        O_tgt = O_tgt.float().cuda()
        O_tgt_loc, O_tgt_scale = get_loc_scale(O_tgt)
        O_tgt_norm = (O_tgt - O_tgt_loc) / O_tgt_scale

        if self.R_model is not None:
            # 计算旋转到标准空间的旋转矩阵
            with torch.no_grad():
                O_src_rot_ = self.R_model.encoder(O_src_norm_[None])['R'][0]
                O_tgt_rot = self.R_model.encoder(O_tgt_norm[None])['R'][0]
            O_src_rot = O_src_rot_@O_tgt_rot.T
        else:
            O_tgt_rot = torch.eye(3).float().cuda()
            O_src_rot = torch.eye(3).float().cuda()
        
        # 对齐物体
        O_src_norm = rotate_pt(O_src_norm_,O_src_rot)
        # 对齐人体
        H_src_param = {k:v.float().cuda() for k,v in H_src_param.items()}
        with torch.no_grad():
            H_src_out = forward_mano(self.H_model,**H_src_param)
        H_src = H_src_out.verts[0]
        H_src_norm = rotate_pt((H_src-O_src_loc)/O_src_scale,O_src_rot)
    
        out_dict['O_src_loc'] = to_np(O_src_loc)
        out_dict['O_src_scale'] = to_np(O_src_scale)
        out_dict['O_tgt_loc'] = to_np(O_tgt_loc)
        out_dict['O_tgt_scale'] = to_np(O_tgt_scale)
        out_dict['O_src_rot'] = to_np(O_src_rot)
        out_dict['O_tgt_rot'] = to_np(O_tgt_rot)
        out_dict['O_src_norm'] = to_np(O_src_norm)
        out_dict['H_src_norm'] = to_np(H_src_norm)
        out_dict['O_tgt_norm'] = to_np(O_tgt_norm)

        ########################################################################
        # 找源物体到目标物体的对应点
        ########################################################################

        # 点云配准：目标变模板
        O_src_reg,theta_src_reg = self.pca.register(O_src_norm)
        O_tgt_reg,theta_tgt_reg =  self.pca.register(O_tgt_norm)

        if self.debug:
            O_tmp = to_np(self.pca.C)
            import pyvista as pv
            pl = pv.Plotter(shape=(2,2))
            pl.subplot(0,0)
            pl.add_mesh(to_np(O_src_norm))
            pl.add_mesh(O_tmp,scalars=(O_tmp+0.5),rgb=True)
            pl.add_mesh(to_np(H_src_norm))
            pl.subplot(1,0)
            pl.add_mesh(to_np(O_src_norm))
            pl.add_mesh(to_np(O_src_reg),scalars=(O_tmp+0.5),rgb=True)
            pl.add_mesh(to_np(H_src_norm))

            pl.subplot(0,1)
            pl.add_mesh(to_np(O_tgt_norm))
            pl.add_mesh(O_tmp,scalars=(O_tmp+0.5),rgb=True)
            pl.add_mesh(to_np(H_src_norm))
            pl.subplot(1,1)
            pl.add_mesh(to_np(O_tgt_norm))
            pl.add_mesh(to_np(O_tgt_reg),scalars=(O_tmp+0.5),rgb=True)
            pl.add_mesh(to_np(H_src_norm))
            pl.link_views()
            pl.show()

        H_src_nn = knn_points(O_src_reg[None],H_src_norm[None])
        idx_src = H_src_nn.idx.squeeze()
        d_src = H_src_nn.dists.sqrt().squeeze()
        vec_src = O_src_reg-H_src_norm[idx_src]

        out_dict['O_src_reg'] = to_np(O_src_reg)
        out_dict['O_tgt_reg'] = to_np(O_tgt_reg)
        out_dict['theta_src_reg'] = to_np(theta_src_reg)
        out_dict['theta_tgt_reg'] = to_np(theta_tgt_reg)

        ########################################################################
        # 基于坐标优化人体参数
        ########################################################################

        # 设置优化参数
        H_tgt_param = {key:H_src_param[key].detach().clone() for key in H_src_param}
        for key in H_tgt_param:
            if key in ['hand_pose','hand_tsl']:
                H_tgt_param[key].requires_grad = True
            else:
                H_tgt_param[key].requires_grad = False

        # 创建优化器和优化闭包
        opt = torch.optim.Adam([H_tgt_param[key] for key in H_tgt_param],lr=1e-2)
        def closure():
            opt.zero_grad()
            H_tgt = forward_mano(self.H_model,**H_tgt_param).verts[0]
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)

            vec_tgt = (O_tgt_reg-H_tgt_norm[idx_src])
            d_tgt = vec_tgt.norm(dim=-1,p=2)

            loss_dist = ((d_tgt-d_src).abs()).sum()

            loss_prior = 0 
            for key in ['hand_pose']:
                if key == 'hand_pose':
                    loss_prior+= (H_tgt_param[key][1:]-H_src_param[key][1:]).square().sum()
            
            loss = loss_dist+25*loss_prior
            loss.backward()
            return [
                loss_prior.item(),
                loss_dist.item(),
                loss.item()
            ]

        loss_all = []
        for i in tqdm(range(250)):
            global_i = i
            loss = opt.step(closure)
            loss_all.append(loss)
        if self.debug:
            vis_loss(loss_all)

        with torch.no_grad():
            H_tgt = forward_mano(self.H_model,**H_tgt_param).verts[0]
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)
        
        out_dict['H_tgt_norm'] = to_np(H_tgt_norm)
        out_dict['H_tgt_param'] = {key:H_tgt_param[key].detach().clone().cpu() for key in H_tgt_param}
        return out_dict