
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from smplx.joint_names import JOINT_NAMES

import trimesh
from trimesh import creation
import pyvista as pv

from tqdm import tqdm

from ...models import get_model
from ...human.hand import load_mano,forward_mano
from ...utils.array_utils import to_np,to_pt
from ...utils.rot_utils import rotate_np,rotate_pt
from ...utils.pcd_utils import get_loc,get_scale,get_loc_scale
from ...utils.plot_utils import vis_loss
from ...utils.sdf_utils import sdf2mesh,template2mesh

from ...spatialmap.nrr import DeformLaplacian
from ...spatialmap.hull import tet_deform,tet_to_world,world_to_tet


class SpatialMapOptimizer():
    def __init__(self,ckpt_path,debug=False):
        self.debug=debug
        if ckpt_path is not None:
            O_model = get_model('vnndif',ckpt_path)
            O_model = O_model.cuda()
            O_model = O_model.eval()
            self.O_model = O_model
        else:
            self.O_model = None
        H_model = load_mano()
        H_model = H_model.cuda()
        H_model.eval()
        # self.O_model = O_model
        self.H_model = H_model
        self.H_f = torch.tensor(self.H_model.get_mano_closed_faces()).long().cuda()
        # 固定这些关节角度不变
        self.H_seg = self.H_model.th_weights.argmax(dim=1).cpu().numpy()
        self.H_jnum = self.H_seg.max()+1

    def transfer(self,H_src_param,O_src,O_tgt,O_src_g0,O_tgt_g0,O_src_tet):
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

        if self.O_model is not None:
            # 计算旋转到标准空间的旋转矩阵
            with torch.no_grad():
                O_src_rot_ = self.O_model.encoder(O_src_norm_[None])['R'][0]
                O_tgt_rot = self.O_model.encoder(O_tgt_norm[None])['R'][0]
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
        # Spatial Map 计算
        ########################################################################

        # 源 genus0 归一化
        O_src_g0_norm =  O_src_g0.copy()
        O_src_g0_norm.vertices = (O_src_g0_norm.vertices-out_dict['O_src_loc'])/out_dict['O_src_scale']
        O_src_g0_norm.vertices = rotate_np(O_src_g0_norm.vertices,out_dict['O_src_rot'])
        
        # 源包围四面体归一化
        O_src_tet_norm = {key:O_src_tet[key].copy() for key in O_src_tet}
        O_src_tet_norm['V'] = (O_src_tet_norm['V']-out_dict['O_src_loc'])/out_dict['O_src_scale']
        O_src_tet_norm['V'] = rotate_np(O_src_tet_norm['V'],out_dict['O_src_rot'])

        # 目标 genus0 归一化
        O_tgt_g0_norm =  O_tgt_g0.copy()
        O_tgt_g0_norm.vertices = (O_tgt_g0_norm.vertices-out_dict['O_tgt_loc'])/out_dict['O_tgt_scale']

        # 源 genus0 变形成目标 genus0
        dl = DeformLaplacian(O_src_g0_norm,O_tgt_g0_norm)
        O_src_g0_reg = dl.deform()

        # 源包围四面体边界变形成 变形后的genus0
        O_src_tet_deform = {key:O_src_tet_norm[key].copy() for key in O_src_tet_norm}
        O_src_tet_deform['V']=tet_deform(O_src_tet_norm['V'],O_src_tet_norm['T'],O_src_g0_norm.vertices, O_src_g0_reg.vertices)

        # 计算源交互在源四面体的坐标
        mask,tet_points = world_to_tet(O_src_tet_norm['V'],O_src_tet_norm['T'],to_np(H_src_norm))
        # 计算四面体坐标在变形四面体的目标坐标
        H_tgt_norm_ = tet_to_world(O_src_tet_deform['V'],O_src_tet_deform['T'],mask,tet_points)


        # out_dict['O_src_tet_norm'] = O_src_tet_norm
        # out_dict['O_src_tet_deform'] = O_src_tet_deform
        out_dict['H_tgt_norm_'] = H_tgt_norm_
        out_dict['mask'] = mask
        
        ########################################################################
        # 基于坐标优化人体参数
        ########################################################################
        mask = torch.tensor(mask).bool().cuda()
        H_tgt_norm_ = torch.tensor(H_tgt_norm_).float().cuda()

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

            # 先验约束
            loss_prior = 0 
            for key in ['hand_pose']:
                if key == 'hand_pose':
                    loss_prior+= (H_tgt_param[key][1:]-H_src_param[key][1:]).square().sum()

            loss_v = ((H_tgt_norm-H_tgt_norm_).square()).sum()
            loss = loss_v + 25*loss_prior
            loss.backward()
            return [
                loss_v.item(),
                loss_prior.item()
            ]

        loss_all = []
        for i in tqdm(range(100)):
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