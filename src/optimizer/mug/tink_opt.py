import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
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

class TinkOptimizer():
    def __init__(self,ckpt_path,vnndif_path=None, debug=False):
        self.debug=debug
        O_model = get_model('deepsdf',ckpt_path)
        O_model = O_model.cuda()
        O_model.eval()
        
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

        # 计算隐向量
        with torch.no_grad():
            O_src_z = self.O_model.encoder(O_src_norm[None])
            O_tgt_z = self.O_model.encoder(O_tgt_norm[None])

        # 插值&重建物体
        N=10
        alpha = torch.linspace(0,1,N+2)[1:-1].float().cuda()    
        pcds = []
        for i in tqdm(range(N)):
            z = O_src_z*(1-alpha[i])+O_tgt_z*(alpha[i])
            m = sdf2mesh(self.O_model,z=z,level=0)
            p = trimesh.sample.sample_surface_even(m, 5000)[0]
            pcds.append(p)

        # 传递最近邻
        p_src = to_np(O_src_norm)
        p_tgt = to_np(O_tgt_norm)
        h_src = to_np(H_src_norm)
        p_i = p_src
        pcds.append(p_tgt)
        C_i = KDTree(h_src).query(p_src)
        C_i = {'dist':C_i[0],'idx':C_i[1],'vec':p_src-h_src[C_i[1]]}
        for i in tqdm(range(N+1)):
            p_i1 = pcds[i]
            idx = KDTree(p_i).query(p_i1)[1]
            C_i1 = {key:C_i[key][idx] for key in C_i}
            p_i = p_i1
            C_i = C_i1

        d_src = torch.tensor(C_i['dist']).float().cuda()
        vec_src = torch.tensor(C_i['vec']).float().cuda()
        idx_src = torch.tensor(C_i['idx']).long().cuda()

        out_dict['O_inter'] = pcds
        out_dict['C'] = C_i

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

            vec_tgt = (O_tgt_norm-H_tgt_norm[idx_src])
            d_tgt = vec_tgt.norm(dim=-1,p=2)

            loss_dist = ((d_tgt-d_src).abs()).sum()

            # SDF约束
            sdf_tgt = self.O_model.inference(H_tgt_norm[None],z=O_tgt_z)['sdf'].squeeze()
            sdf_th = 0
            loss_sdf = (sdf_tgt-sdf_th).clamp(max=0).square().sum()

            loss_prior = 0 
            for key in ['hand_pose']:
                if key == 'hand_pose':
                    loss_prior+= (H_tgt_param[key][1:]-H_src_param[key][1:]).square().sum()
            
            loss = loss_dist+25*loss_prior+100*loss_sdf
            loss.backward()
            return [
                loss_prior.item(),
                loss_dist.item(),
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
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)
        
        out_dict['H_tgt_norm'] = to_np(H_tgt_norm)
        out_dict['H_tgt_param'] = {key:H_tgt_param[key].detach().clone().cpu() for key in H_tgt_param}
        return out_dict