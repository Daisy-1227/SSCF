
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from smplx.joint_names import JOINT_NAMES

import trimesh
from trimesh import creation

from tqdm import tqdm

from ...models import get_model
from ...human.hand import load_mano,forward_mano
from ...utils.array_utils import to_np,to_pt
from ...utils.rot_utils import rotate_np,rotate_pt
from ...utils.pcd_utils import get_loc,get_scale,get_loc_scale
from ...utils.plot_utils import vis_loss


class NDFOptimizer():
    def __init__(self,ckpt_path,debug=False):
        self.debug=debug
        O_model = get_model('ndf',ckpt_path)
        O_model = O_model.cuda()
        O_model.eval()
        H_model = load_mano()
        H_model = H_model.cuda()
        H_model.eval()
        self.O_model = O_model
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
        # 归一化物体
        ########################################################################

        # 归一化源物体
        O_src = O_src.float().cuda()
        O_src_loc, O_src_scale = get_loc_scale(O_src)
        O_src_norm = (O_src - O_src_loc) / O_src_scale

        # 归一化目标物体
        O_tgt = O_tgt.float().cuda()
        O_tgt_loc, O_tgt_scale = get_loc_scale(O_tgt)
        O_tgt_norm = (O_tgt - O_tgt_loc) / O_tgt_scale

        # 归一化人体
        H_src_param = {k:v.float().cuda() for k,v in H_src_param.items()}
        with torch.no_grad():
            H_src_out = forward_mano(self.H_model,**H_src_param)
        H_src = H_src_out.verts[0]
        H_src_norm = (H_src-O_src_loc)/O_src_scale
        

        out_dict['O_src_loc'] = to_np(O_src_loc)
        out_dict['O_src_scale'] = to_np(O_src_scale)
        out_dict['O_tgt_loc'] = to_np(O_tgt_loc)
        out_dict['O_tgt_scale'] = to_np(O_tgt_scale)
        out_dict['O_src_norm'] = to_np(O_src_norm)
        out_dict['H_src_norm'] = to_np(H_src_norm)
        out_dict['O_tgt_norm'] = to_np(O_tgt_norm)

        ########################################################################
        # 计算人体NDF特征
        ########################################################################

        with torch.no_grad():
            O_src_z = self.O_model.encoder(O_src_norm[None])
            H_src_ndf = self.O_model.inference(H_src_norm[None],z=O_src_z,return_ndf=True)['ndf']
            O_tgt_z = self.O_model.encoder(O_tgt_norm[None])

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
            H_tgt_norm = (H_tgt-O_src_loc)/O_src_scale

            # NDF 目标
            H_tgt_feat = self.O_model.inference(H_tgt_norm[None],z=O_tgt_z,return_ndf=True)
            H_tgt_ndf = H_tgt_feat['ndf']
            H_tgt_sdf = H_tgt_feat['sdf']
            sdf_th = 0
            loss_sdf = (H_tgt_sdf-sdf_th).clamp(max=0).square().sum()

            # 先验约束
            loss_prior = 0 
            for key in ['hand_pose']:
                if key == 'hand_pose':
                    loss_prior+= (H_tgt_param[key][1:]-H_src_param[key][1:]).square().sum()

            # 优化目标
            loss_v = ((H_tgt_ndf-H_src_ndf).square()).sum()

            loss = loss_v+25*loss_prior+100*loss_sdf
            loss.backward()
            return [
                loss_v.item(),
                loss_prior.item(),
                loss_sdf.item(),
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
            H_tgt_norm = (H_tgt-O_src_loc)/O_src_scale
        
        out_dict['H_tgt_norm'] = to_np(H_tgt_norm)
        out_dict['H_tgt_param'] = {key:H_tgt_param[key].detach().clone().cpu() for key in H_tgt_param}
        return out_dict