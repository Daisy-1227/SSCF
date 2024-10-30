import numpy as np
import torch

from src.models import get_model
from src.human.body import load_smplx
from src.utils.pcd_utils import get_loc_scale
from src.utils.rot_utils import rotate_pt
from src.utils.plot_utils import vis_loss
from src.utils.array_utils import to_np
from src.utils.grad_utils import grads_clip

from trimesh import creation
import pyvista as pv
from pytorch3d.ops import knn_points,norm_laplacian
from pytorch3d.structures import Meshes

from scipy.spatial import Delaunay

from tqdm import tqdm

class SSCFOptimizer():
    def __init__(self,ckpt_path,optimizer='vL',debug=False,ckpt_path2=None):
        self.optimizer = optimizer
        print(self.optimizer)
        self.debug = debug

        # load object model
        O_model = get_model('vnndif',ckpt_path)
        O_model = O_model.cuda()
        O_model.eval()
        self.O_model = O_model

        if ckpt_path2 is not None:
            O_model2 = get_model('dif',ckpt_path2)
            O_model2 = O_model2.cuda()
            O_model2.eval()
            self.O_model2 = O_model2
        else:
            self.O_model2 = None

        # load human model
        H_model = load_smplx()
        H_model = H_model.cuda()
        H_model.eval()
        self.H_model = H_model

        # get mesh faces of human model
        self.H_f =  torch.tensor(self.H_model.faces.astype('int')).long().cuda()

    def get_delaunay_edges(self,V):
        tet = Delaunay(V.cpu().numpy())
        neighbors_indptr,neighbors_indices =  tet.vertex_neighbor_vertices
        edges = []
        for i in range(len(V)):
            neighbors = neighbors_indices[neighbors_indptr[i]:neighbors_indptr[i+1]]
            for neighbor_vert in neighbors:
                edges.append([i,neighbor_vert])
        edges = torch.tensor(edges).cuda()
        return edges
    
    def get_laplacian_coordinates(self,V,edges,eps=1e-12):
        with torch.no_grad():
            L = norm_laplacian(V,edges).to_dense()
            L_sum = L.sum(dim=-1,keepdim=True)
            L = L/(L_sum+eps)
        # delta = (L_sum*V-L@V)
        delta = (V-L@V)
        return delta

    def get_nearest_grid_points(self,points,z,resolution=128,scale=1.5):
        O_model = self.O_model2 if self.O_model2 is not None else self.O_model
        grid = creation.box().bounding_box.sample_grid(resolution)*scale
        grid = torch.tensor(grid).float().cuda()
        with torch.no_grad():
            grid_temp = O_model.inference(grid[None],z=z)['pts_template'][0]
            points_grid = grid[knn_points(points[None],grid_temp[None],K=1).idx.squeeze()]
        return points_grid
    
    def optimize_v_temp(self,v_init,z,v_temp):        
        O_model = self.O_model2 if self.O_model2 is not None else self.O_model

        v_opt = v_init.detach().clone().requires_grad_()
        opt = torch.optim.SGD([v_opt],lr=1)
        explr = torch.optim.lr_scheduler.StepLR(opt,gamma=0.5,step_size=10)
        losses = []
        for i in tqdm(range(30)):
            opt.zero_grad()
            v_temp_opt = O_model.inference(v_opt[None],z=z)['pts_template'][0]
            loss_temp = (v_temp_opt - v_temp).square().sum()
            loss = loss_temp
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(
                v_opt,0.001
            )
            opt.step()
            explr.step()

            losses.append(loss.item())
        if self.debug:
            losses = np.array(losses).reshape(-1,1)
            vis_loss(losses,'log')

        v_opt.requires_grad = False
        return v_opt

    def optimize_L_free(self,v_init,O,edges,L_ref):
        v_opt = v_init.detach().clone().requires_grad_()
        opt = torch.optim.SGD([v_opt],lr=1)
        explr = torch.optim.lr_scheduler.StepLR(opt,gamma=0.5,step_size=100)
        losses = []
        for i in tqdm(range(200)):
            opt.zero_grad()
            V = torch.cat([v_opt,O],dim=0)
            L = self.get_laplacian_coordinates(V,edges)
            loss = (L-L_ref)[:len(v_opt)].square().sum()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(
                v_opt,0.001
            )
            opt.step()
            explr.step()
            losses.append(loss.item())
        
        if self.debug:
            losses = np.array(losses).reshape(-1,1)
            vis_loss(losses,'log')

        v_opt.requires_grad = False
        return v_opt
    
    def optimize_v(self,param_init,v_ref,O_sdf=None):
        # set up variables
        param_opt = {key:param_init[key].detach().clone() for key in param_init}
        for key in param_opt:
            if key in ['transl','global_orient','body_pose']:
                param_opt[key].requires_grad = True
            else:
                param_opt[key].requires_grad = False

        # set optimizer
        opt = torch.optim.SGD([param_opt[key] for key in param_opt],lr=1)
        explr = torch.optim.lr_scheduler.StepLR(opt,gamma=0.5,step_size=10)

        def step():
            opt.zero_grad()
            v = self.H_model(**param_opt).vertices[0]
            loss_v = (v-v_ref).square().sum()

            # SDF约束
            knn = knn_points(O_sdf[None],v[None])
            nn = knn.idx.squeeze()
            vec = O_sdf-v[nn]
            depth = knn.dists.sqrt().squeeze()
            m = Meshes([v],[self.H_f])
            vn = m.verts_normals_packed()
            dot = (vec*vn[nn]).sum(-1)
            loss_sdf = dot.clamp(max=0).square().sum()

            losses = {
                'v':loss_v,
                'sdf':loss_sdf
            }
            grads_clip(losses,param_opt,0.01,True)

            opt.step()
            explr.step()

            return [losses[key].item() for key in losses]
        
        losses = []
        for i in tqdm(range(100)):
            loss = step()
            losses.append(loss)
        
        if self.debug:
            vis_loss(losses,'log')      
                
        return param_opt

    def optimize_L(self,param_init,O,edges,L_ref,O_sdf=None):
        # set up variables
        param_opt = {key:param_init[key].detach().clone() for key in param_init}
        for key in param_opt:
            if key in ['transl','global_orient','body_pose']:
                param_opt[key].requires_grad = True
            else:
                param_opt[key].requires_grad = False

        # set optimizer
        opt = torch.optim.SGD([param_opt[key] for key in param_opt],lr=1)
        explr = torch.optim.lr_scheduler.StepLR(opt,gamma=0.5,step_size=10)
        
        def step():
            opt.zero_grad()
            v = self.H_model(**param_opt).vertices[0]

            V = torch.cat([v,O],dim=0)
            L = self.get_laplacian_coordinates(V,edges)
            loss_L = (L-L_ref)[:len(v)].square().sum()

            # SDF约束
            knn = knn_points(O_sdf[None],v[None])
            nn = knn.idx.squeeze()
            vec = O_sdf-v[nn]
            depth = knn.dists.sqrt().squeeze()
            m = Meshes([v],[self.H_f])
            vn = m.verts_normals_packed()
            dot = (vec*vn[nn]).sum(-1)
            loss_sdf = dot.clamp(max=0).square().sum()

            losses = {
                'L':loss_L,
                'sdf':loss_sdf
            }
            grads_clip(losses,param_opt,0.01,True)

            opt.step()
            explr.step()

            return [losses[key].item() for key in losses]
        
        losses = []
        for i in tqdm(range(100)):
            loss = step()
            losses.append(loss)
        
        if self.debug:
            vis_loss(losses,'log')
        
        return param_opt

    def optimize_vL(self,param_init,v_ref,O,edges,L_ref,O_sdf=None):
        # set up variables
        param_opt = {key:param_init[key].detach().clone() for key in param_init}
        for key in param_opt:
            if key in ['global_orient','body_pose','transl']:
                param_opt[key].requires_grad = True
            else:
                param_opt[key].requires_grad = False

        # set optimizer
        opt = torch.optim.SGD([param_opt[key] for key in param_opt],lr=1)
        explr = torch.optim.lr_scheduler.StepLR(opt,gamma=0.5,step_size=10)

        def step():
            opt.zero_grad()
            v = self.H_model(**param_opt).vertices[0]
            loss_v = (v-v_ref).square().sum()

            V = torch.cat([v,O],dim=0)
            L = self.get_laplacian_coordinates(V,edges)
            loss_L = (L-L_ref)[:len(v)].square().sum()

            # SDF约束
            knn = knn_points(O_sdf[None],v[None])
            nn = knn.idx.squeeze()
            vec = O_sdf-v[nn]
            depth = knn.dists.sqrt().squeeze()
            m = Meshes([v],[self.H_f])
            vn = m.verts_normals_packed()
            dot = (vec*vn[nn]).sum(-1)
            loss_sdf = dot.clamp(max=0).square().sum()

            losses = {
                'v':loss_v,
                'L':loss_L,
                'sdf':loss_sdf
            }
            grads_clip(losses,param_opt,0.01,norm=True)
            # grads_clip(losses,param_opt,0.01)

            opt.step()
            explr.step()

            return [losses[key].item() for key in losses]
        
        losses = []
        for i in tqdm(range(100)):
            loss = step()
            losses.append(loss)
        
        if self.debug:
            vis_loss(losses,'log')
        
        return param_opt

    def transfer(self,H_src_param,O_src,O_tgt,scale=1.25):
        out_dict = {}

        O_src = O_src.float().cuda()
        O_tgt = O_tgt.float().cuda()

        ########################################################################
        # 1. align objects in canonical poses
        ########################################################################

        # normalize source object by centroid
        O_src_loc, O_src_scale = get_loc_scale(O_src)
        O_src_norm_ = (O_src-O_src_loc)/O_src_scale

        # normalize target object by centroid
        O_tgt_loc, O_tgt_scale = get_loc_scale(O_tgt)
        O_tgt_scale *= scale
        O_tgt_norm = (O_tgt - O_tgt_loc) / O_tgt_scale

        # predict pose with VNNDIF
        with torch.no_grad():
            O_src_rot_ = self.O_model.encoder(O_src_norm_[None])['R'][0]
            O_tgt_rot = self.O_model.encoder(O_tgt_norm[None])['R'][0]

        # this is relative rotation from target pose to source pose
        # rotate source object to canonical pose (right multiply)
        O_src_rot = O_src_rot_@O_tgt_rot.T
        O_src_norm = rotate_pt(O_src_norm_,O_src_rot)

        # rotate source human to canonical pose for mapping
        with torch.no_grad():
            H_src_param = {k:v.float().cuda() for k,v in H_src_param.items()}
            H_src_out = self.H_model(**H_src_param)
            H_src = H_src_out.vertices[0]
        H_src_norm = rotate_pt((H_src-O_src_loc)/O_src_scale,O_src_rot)

        # transform target object to source pose for optimization
        O_tgt_ = rotate_pt(O_tgt_norm,O_src_rot.T)*O_src_scale+O_src_loc

        out_dict['O_src_loc'] = to_np(O_src_loc)
        out_dict['O_src_scale'] = to_np(O_src_scale)
        out_dict['O_src_rot'] = to_np(O_src_rot)
        out_dict['O_src_norm'] = to_np(O_src_norm)

        out_dict['O_tgt_loc'] = to_np(O_tgt_loc)
        out_dict['O_tgt_scale'] = to_np(O_tgt_scale)
        out_dict['O_tgt_rot'] = to_np(O_tgt_rot)
        out_dict['O_tgt_norm'] = to_np(O_tgt_norm)

        out_dict['H_src_norm'] = to_np(H_src_norm)
        ########################################################################
        # 2. map objects and human to tempalte field
        ########################################################################
        O_model = self.O_model2 if self.O_model2 is not None else self.O_model
        with torch.no_grad():
            # map source object and source human to template field
            O_src_z = O_model.encoder(O_src_norm[None])
            O_src_feat = O_model.inference(O_src_norm[None],z=O_src_z)
            H_src_feat = O_model.inference(H_src_norm[None],z=O_src_z)

            # map target object to template field
            O_tgt_z = O_model.encoder(O_tgt_norm[None])
            O_tgt_feat = O_model.inference(O_tgt_norm[None],z=O_tgt_z)

            O_src_temp = O_src_feat['pts_template'][0]
            H_src_temp = H_src_feat['pts_template'][0]
            O_tgt_temp = O_tgt_feat['pts_template'][0]

        idx_tgt2src = knn_points(O_src_temp[None],O_tgt_temp[None]).idx.squeeze()

        ########################################################################
        # 4. optimize 
        ########################################################################
        if self.optimizer == 'v_temp':
            H_tgt_norm = self.get_nearest_grid_points(H_src_temp,O_tgt_z)
            H_tgt_norm = self.optimize_v_temp(H_tgt_norm,O_tgt_z,H_src_temp)
            H_tgt = rotate_pt(H_tgt_norm,O_src_rot.T)*O_src_scale+O_src_loc
            
        if self.optimizer == 'v':
            H_tgt_norm_ = self.get_nearest_grid_points(H_src_temp,O_tgt_z)
            H_tgt_norm_ = self.optimize_v_temp(H_tgt_norm_,O_tgt_z,H_src_temp)
            H_tgt_ = rotate_pt(H_tgt_norm_,O_src_rot.T)*O_src_scale+O_src_loc

            H_tgt_param = self.optimize_v(H_src_param,H_tgt_,O_sdf=O_tgt_)
            with torch.no_grad():
                H_tgt = self.H_model(**H_tgt_param).vertices[0]
                H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)

        if self.optimizer == 'L_free':
            V_src = torch.cat([H_src,O_src],dim=0)
            edges = self.get_delaunay_edges(V_src)
            L_src = self.get_laplacian_coordinates(V_src,edges)
            H_tgt = self.optimize_L_free(H_src,O_tgt_[idx_tgt2src],edges,L_src)
            H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)

        elif self.optimizer == 'L':
            V_src = torch.cat([H_src,O_src],dim=0)
            edges = self.get_delaunay_edges(V_src)
            L_src = self.get_laplacian_coordinates(V_src,edges)
            H_tgt_param = self.optimize_L(H_src_param,O_tgt_[idx_tgt2src],edges,L_src,O_sdf=O_tgt_)
            with torch.no_grad():
                H_tgt = self.H_model(**H_tgt_param).vertices[0]
                H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)

        elif self.optimizer == 'vL':
            H_tgt_norm_ = self.get_nearest_grid_points(H_src_temp,O_tgt_z)
            H_tgt_norm_ = self.optimize_v_temp(H_tgt_norm_,O_tgt_z,H_src_temp)
            H_tgt_ = rotate_pt(H_tgt_norm_,O_src_rot.T)*O_src_scale+O_src_loc

            V_src = torch.cat([H_src,O_src],dim=0)
            edges = self.get_delaunay_edges(V_src)
            L_src = self.get_laplacian_coordinates(V_src,edges)
        
            H_tgt_param = self.optimize_vL(H_src_param,H_tgt_,O_tgt_[idx_tgt2src],edges,L_src,O_sdf=O_tgt_)
            with torch.no_grad():
                H_tgt = self.H_model(**H_tgt_param).vertices[0]
                H_tgt_norm = rotate_pt((H_tgt-O_src_loc)/O_src_scale,O_src_rot)
        
        out_dict['H_tgt_norm'] = to_np(H_tgt_norm)
        return out_dict