
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree as KDTree
import cycpd

from smplx.joint_names import JOINT_NAMES

import trimesh
from trimesh import creation
import point_cloud_utils as pcu

from tqdm import tqdm

from ...models import get_model
from ...human.body import load_smplx
from ...utils.array_utils import to_np,to_pt
from ...utils.rot_utils import rotate_np,rotate_pt
from ...utils.pcd_utils import get_loc,get_scale,get_loc_scale
from ...utils.plot_utils import vis_loss
from ...utils.sdf_utils import sdf2mesh

class CPDOptimizer():
    def __init__(self,vnndif_path=None, debug=False):
        self.debug=debug

        if vnndif_path is not None:
            R_model = get_model('vnndif',vnndif_path)
            R_model = R_model.cuda()
            R_model.eval()
            self.R_model = R_model
        else:
            self.R_model = None

        H_model = load_smplx()
        H_model = H_model.cuda()
        H_model.eval()
        self.H_model = H_model
        self.H_f = torch.tensor(self.H_model.faces.astype('int')).long().cuda()
        # 固定这些关节角度不变
        mask = torch.zeros([1,21,3])
        mask[:,[3,6,9,10,11,12,13,14,15]] = 1
        mask[:,22:] = 1
        mask = mask.reshape(1,-1)
        self.mask = mask.cuda()
        

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
            H_src_out = self.H_model(**H_src_param)
        H_src = H_src_out.vertices[0]
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

        # 点云配准：目标变源
        cpd = cycpd.deformable_registration(X=to_np(O_src_norm).astype('float64'),Y=to_np(O_tgt_norm).astype('float64'),w=0.5,alpha=3,beta=1,tolerance=1e-6,max_iterations=150,verbose=True,print_reg_params=False)
        # out = probreg.cpd.registration_cpd(to_np(O_tgt_norm),to_np(O_src_norm),'nonrigid',use_cuda=True)

        O_src_reg, T_src_reg = cpd.register()

        # if self.debug:
        #     import pyvista as pv
        #     pl = pv.Plotter(shape=(1,4))
        #     pl.subplot(0,0)
        #     pl.add_mesh(to_np(O_src_norm))
        #     pl.subplot(0,1)
        #     pl.add_mesh(to_np(O_tgt_norm),scalars=to_np(O_tgt_norm)+0.5,rgb=True)
        #     pl.subplot(0,2)
        #     pl.add_mesh(O_src_reg,scalars=to_np(O_tgt_norm)+0.5,rgb=True)
        #     pl.subplot(0,3)
        #     pl.add_mesh(to_np(O_src_norm))
        #     pl.add_mesh(O_src_reg,scalars=to_np(O_tgt_norm)+0.5,rgb=True)
        #     pl.link_views()
        #     pl.show()

        O_src_reg = torch.tensor(O_src_reg).float().cuda()
        idx = knn_points(O_src_reg[None],O_src_norm[None]).idx.squeeze()
        
        H_src_nn = knn_points(O_src_norm[None],H_src_norm[None])
        C_i = {
            'dist': H_src_nn.dists.sqrt().squeeze(),
            'idx': H_src_nn.idx.squeeze(),
            'vec': O_src_norm-H_src_norm[H_src_nn.idx.squeeze()]
        }

        C_i = {key:C_i[key][idx] for key in C_i}

        d_src = torch.tensor(C_i['dist']).float().cuda()
        vec_src = torch.tensor(C_i['vec']).float().cuda()
        idx_src = torch.tensor(C_i['idx']).long().cuda()

        out_dict['O_src_reg'] = to_np(O_src_reg)
        out_dict['C'] = {key:to_np(C_i[key]) for key in C_i}

        ########################################################################
        # 基于坐标优化人体参数
        ########################################################################

        # 设置优化参数
        H_tgt_param = {key:H_src_param[key].detach().clone() for key in H_src_param}
        for key in H_tgt_param:
            if key in ['transl','global_orient','body_pose']:
                H_tgt_param[key].requires_grad = True
            else:
                H_tgt_param[key].requires_grad = False

        # 创建优化器和优化闭包
        opt = torch.optim.Adam([H_tgt_param[key] for key in H_tgt_param],lr=1e-2)
        def closure():
            opt.zero_grad()
            H_tgt = self.H_model(**H_tgt_param).vertices[0]
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)
            # idx_src: 离src物体最近的相应的src人体坐标, 希望transfer之后tgt相应的人体坐标也与tgt物体尽可能接近
            vec_tgt = (O_tgt_norm-H_tgt_norm[idx_src])
            d_tgt = vec_tgt.norm(dim=-1,p=2)

            loss_dist = ((d_tgt-d_src).abs()).sum()

            # SDF约束
            # sdf_tgt = self.O_model.inference(H_tgt_norm[None],z=O_tgt_z)['sdf'].squeeze()
            # sdf_th = 0
            # loss_sdf = (sdf_tgt-sdf_th).clamp(max=0).square().sum()

            # 先验约束
            loss_prior = 0 
            for key in ['body_pose']:
                loss_prior+= (H_tgt_param[key]-H_src_param[key]).square().sum()
            loss_prior+=100*((H_tgt_param['body_pose']-H_src_param['body_pose'])*self.mask).square().sum()

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
        # if self.debug:
        #     vis_loss(loss_all)

        with torch.no_grad():
            H_tgt = self.H_model(**H_tgt_param).vertices[0]
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)
        
        out_dict['H_tgt_norm'] = to_np(H_tgt_norm)
        out_dict['H_tgt_param'] = {key:H_tgt_param[key].detach().clone().cpu() for key in H_tgt_param}
        return out_dict