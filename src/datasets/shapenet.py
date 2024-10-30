import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points
from scipy.spatial.transform import Rotation
import point_cloud_utils as pcu
import trimesh
from src.utils.rot_utils import rotate_np, rotate_pt
from ..spatialmap.hull import mesh_to_genus0

class ShapeNetDataset(Dataset):
    def __init__(self,root,category,split='train',partial=False,
                 npcd=5000,npts=4096,query_surf=True,
                 rotate=False,scale=False,noise=False) -> None:
        super().__init__()
        self.root = root
        self.category = category
        self.split = split
        self.partial = partial

        self.npcd = npcd
        self.npts = npts
        self.query_surf = query_surf

        self.rotate = rotate
        self.scale = scale
        self.noise = noise

        lst_path = f'{root}/lst/{category}.lst'
        self.names = open(lst_path).read().split()

    def __len__(self): return len(self.names)

    def get_mesh_wt(self,index):
        name = self.names[index]
        mesh = pcu.TriangleMesh(f'{self.root}/watertight/{self.category}/{name}.ply')
        return trimesh.Trimesh(mesh.v,mesh.f,process=False)
    
    def get_pcd_fixed(self,index,partial=False,idx_view=0):
        name = self.names[index]
        if partial:
            idx_view = idx_view if idx_view>=0 else np.random.randint(26)
            pcd_partial = np.load(f'{self.root}/partial/{self.category}/pcd/{name}/{idx_view}.npy')
            # idx_pcd = np.random.permutation(len(pcd_partial))[:self.npcd]
            # pcd = pcd_partial[idx_pcd]
            pcd = sample_farthest_points(torch.tensor(pcd_partial)[None],K=self.npcd)[0].squeeze().numpy()
            pcd = pcd_partial
        else:
            fixed_path = f'{self.root}/fixed_{self.npcd}/{self.category}/{name}.npy'
            if os.path.exists(fixed_path):
                pcd = np.load(fixed_path)
            else:
                data = np.load(f'{self.root}/samples/{self.category}/{name}.npz')
                # pcd = data['p_surf'][:self.npcd]
                pcd = sample_farthest_points(torch.tensor(data['p_surf'])[None].cuda(),K=self.npcd)[0].squeeze().cpu().numpy()
                os.makedirs(os.path.dirname(fixed_path),exist_ok=True)
                np.save(fixed_path,pcd)
        return pcd

    def __getitem__(self, index, return_np=False, idx_view=0):
        # name = self.names[index]
        # data = np.load(f'{self.root}/samples/{self.category}/{name}.npz')
        # num_surf = len(data['p_surf'])
        # num_vol = len(data['p_vol'])
        #
        # item = {}
        #
        # if self.partial:
        #     idx_view = np.random.randint(26)
        #     pcd_partial = np.load(f'{self.root}/partial/{self.category}/pcd/{name}/{idx_view}.npy')
        #     print(len(pcd_partial))
        #     idx_pcd = np.random.permutation(len(pcd_partial))[:self.npcd]
        #     pcd = pcd_partial[idx_pcd]
        #     item['pcd'] = pcd
        # else:
        #     idx_pcd = np.random.permutation(num_surf)[:self.npcd]
        #     pcd = data['p_surf'][idx_pcd]
        #     item['pcd'] = pcd
        #
        # if self.query_surf:
        #     idx_vol = np.random.permutation(num_vol)[:self.npts//2]
        #     idx_surf = np.random.permutation(num_vol)[:self.npts//2]
        #
        #     pts_vol = data['p_vol'][idx_vol]
        #     pts_surf = data['p_surf'][idx_surf]
        #
        #     norm_vol = np.ones_like(pts_vol)*-1
        #     norm_surf = data['n_surf'][idx_surf]
        #
        #     sdf_vol = data['sdf_vol'][idx_vol]
        #     sdf_surf = np.zeros(self.npts//2)
        #
        #     item['pts'] = np.concatenate([pts_vol,pts_surf],axis=0)
        #     item['sdf'] = np.concatenate([sdf_vol,sdf_surf])
        #     item['normals'] = np.concatenate([norm_vol,norm_surf],axis=0)
        # else:
        #     idx = np.random.permutation(num_vol)[:self.npts]
        #     item['pts'] = data['p_vol'][idx]
        #     item['sdf'] = data['sdf_vol'][idx]
        #
        # if self.scale:
        #     scale = np.random.uniform(0.9,1.1)
        #     item['pcd'] = item['pcd']*scale
        #     item['pts'] = item['pts']*scale
        #     item['sdf'] = item['sdf']*scale
        # item['sdf'][np.abs(item['sdf'])>0.5] = -1
        #
        # if self.rotate:
        #     R = Rotation.random().as_matrix()
        #     item['pcd'] = rotate_np(item['pcd'],R)
        #     item['pts'] = rotate_np(item['pts'],R)
        #
        # if self.noise:
        #     noise = 0.005*np.random.randn(*item['pcd'].shape)
        #     item['pcd'] = item['pcd']+noise
        #
        # if return_np:
        #     return item
        #
        # for key in item:
        #     item[key] = torch.tensor(item[key]).float()
        # item['sdf'] = item['sdf'].unsqueeze(-1)
        #
        # return item
        name = self.names[index]
        if self.partial:
            idx_view = idx_view if idx_view >= 0 else np.random.randint(26)
            pcd_partial = np.load(f'{self.root}/partial/{self.category}/pcd/{name}/{idx_view}.npy')
            # idx_pcd = np.random.permutation(len(pcd_partial))[:self.npcd]
            # pcd = pcd_partial[idx_pcd]
            pcd = sample_farthest_points(torch.tensor(pcd_partial)[None], K=self.npcd)[0].squeeze().numpy()
            # pcd = pcd_partial
        else:
            fixed_path = f'{self.root}/fixed_{self.npcd}/{self.category}/{name}.npy'
            if os.path.exists(fixed_path):
                pcd = np.load(fixed_path)
            else:
                data = np.load(f'{self.root}/samples/{self.category}/{name}.npz')
                # pcd = data['p_surf'][:self.npcd]
                pcd = sample_farthest_points(torch.tensor(data['p_surf'])[None].cuda(), K=self.npcd)[
                    0].squeeze().cpu().numpy()
                os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
                np.save(fixed_path, pcd)

        if return_np:
            return pcd
        else:
            return torch.tensor(pcd).float()

class ShapeNetTestDataset(Dataset):
    def __init__(self,root,category,partial=False,npcd=5000) -> None:
        super().__init__()
        self.root = root
        self.category = category
        self.partial = partial
        self.npcd = npcd
        lst_path = f'{root}/lst/{category}.lst'

        # print(lst_path)
        self.names = open(lst_path).read().split()

    def __len__(self): return len(self.names)
    
    def get_mesh_wt(self,index):
        name = self.names[index]
        mesh = pcu.TriangleMesh(f'{self.root}/watertight/{self.category}/{name}.ply')
        return trimesh.Trimesh(mesh.v,mesh.f,process=False)

    def get_mesh_g0(self, index):
        name = self.names[index]
        if os.path.exists(name):
            return trimesh.load(f'{self.root}/g0/{self.category}/{name}.ply')
        mesh = self.get_mesh_wt(index)
        mesh_g0 = mesh_to_genus0(mesh)
        os.makedirs(f'{self.root}/g0/{self.category}', exist_ok=True)
        mesh_g0.export(f'{self.root}/g0/{self.category}/{name}.ply')
        return mesh_g0

    def __getitem__(self, index, return_np=False, idx_view=0):
        name = self.names[index]
        if self.partial:
            idx_view = idx_view if idx_view>=0 else np.random.randint(26)
            pcd_partial = np.load(f'{self.root}/partial/{self.category}/pcd/{name}/{idx_view}.npy')
            # idx_pcd = np.random.permutation(len(pcd_partial))[:self.npcd]
            # pcd = pcd_partial[idx_pcd]
            pcd = sample_farthest_points(torch.tensor(pcd_partial)[None],K=self.npcd)[0].squeeze().numpy()
            # pcd = pcd_partial
        else:
            fixed_path = f'{self.root}/fixed_{self.npcd}/{self.category}/{name}.npy'
            if os.path.exists(fixed_path):
                pcd = np.load(fixed_path)
            else:
                data = np.load(f'{self.root}/samples/{self.category}/{name}.npz')
                # pcd = data['p_surf'][:self.npcd]
                pcd = sample_farthest_points(torch.tensor(data['p_surf'])[None].cuda(),K=self.npcd)[0].squeeze().cpu().numpy()
                os.makedirs(os.path.dirname(fixed_path),exist_ok=True)
                np.save(fixed_path,pcd)

        if return_np:
            return pcd
        else:
            return torch.tensor(pcd).float()