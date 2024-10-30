import os
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from ..spatialmap.hull import mesh_to_genus0, genus0_to_tet
import point_cloud_utils as pcu


class ChairsDataset(Dataset):
    def __init__(self, root) -> None:
        self.root = root
        self.names = open(f'{root}/names.lst').read().split()

    def __len__(self):
        return len(self.names)

    def get_mesh(self, index):
        name = self.names[index]
        return trimesh.load(f'{self.root}/mesh/{name}.ply')

    def get_mesh_g0(self, index):
        name = self.names[index]
        if os.path.exists(f'{self.root}/mesh/{name}_g0.ply'):
            return trimesh.load(f'{self.root}/mesh/{name}_g0.ply')
        mesh = self.get_mesh(index)
        mesh_g0 = mesh_to_genus0(mesh)
        mesh_g0.export(f'{self.root}/mesh/{name}_g0.ply')
        return mesh_g0

    def get_mesh_tet(self, index, return_vtk=False):
        name = self.names[index]
        if os.path.exists(f'{self.root}/mesh/{name}_tet.npz'):
            tet = np.load(f'{self.root}/mesh/{name}_tet.npz')
            V, T = tet['V'], tet['T']
        else:
            mesh_g0 = self.get_mesh_g0(index)
            V, T = genus0_to_tet(mesh_g0)
            np.savez(f'{self.root}/mesh/{name}_tet.npz', V=V, T=T)
        # if return_vtk:
        #     tet = pv.PolyData(V, np.concatenate([np.full((len(T), 1), 4), T], axis=-1).flatten())
        #     return tet
        return {'V': V, 'T': T}

    def get_mesh_v(self, index):
        name = self.names[index]
        v = pcu.load_mesh_v(f'{self.root}/mesh/{name}.ply')
        v = torch.tensor(v).float()
        return v

    def __getitem__(self, index):
        name = self.names[index]
        data = torch.load(f'{self.root}/samples/{name}.pt')
        return data
