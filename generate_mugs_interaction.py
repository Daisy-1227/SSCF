import os
import torch
from src.datasets.mugs import MugsDataset
from src.datasets.shapenet import ShapeNetTestDataset
from src.human.hand import load_mano, param2mesh
import numpy as np
import pyvista as pv
import trimesh
from tqdm import tqdm
from src.optimizer.mug.sscf_opt import SSCFOptimizer
from src.utils.rot_utils import rotate_np


# Load mano model
H_model = load_mano(mano_assets_root='data/human_model/mano')

# Load source interaction dataset
dataset_src = MugsDataset(f'data/hoi/mugs')

# Load target object dataset
dataset_tgt = ShapeNetTestDataset('data/shapenet','03797390')

# Load SSCF optimzer
optimizer = SSCFOptimizer('data/ckpts/mug/checkpoint.ckpt')

color_anc = 'w'
color_src = '#15c4ed'
color_tgt = '#fe7a89'

visualize = True
save_results = True

save_folder = f'results/mug'

if not os.path.exists(save_folder):
    os.makedirs(save_folder, exist_ok=True)

src_list = [i for i in range(len(dataset_src))]

tgt_list = [i for i in range(len(dataset_tgt))]

for i_src in src_list:

    i_tgt = tgt_list[np.random.randint(len(tgt_list))]

    print(f'i_src: {i_src}, i_tgt: {i_tgt}')

    # data_src: src_interaction
    data_src = dataset_src[i_src]
    # O_tgt: tgt_object
    O_tgt = dataset_tgt[i_tgt]
    # O_src: src_object
    O_src = data_src['O_src']

    # H_src_param: src_human
    H_src_param = data_src['H_src_param']

    # transfer
    out_dict = optimizer.transfer(H_src_param, O_src, O_tgt)

    # visualize results
    if visualize:
        O_src_mesh = dataset_src.get_mesh(i_src)
        O_src_loc = out_dict['O_src_loc']
        O_src_scale = out_dict['O_src_scale']
        if 'O_src_rot' not in out_dict:
            O_src_rot = np.eye(3)
        else:
            O_src_rot = out_dict['O_src_rot']
        O_src_mesh.vertices = rotate_np((O_src_mesh.vertices - O_src_loc) / O_src_scale, O_src_rot)
        H_src_norm = out_dict['H_src_norm']
        H_src = trimesh.Trimesh(H_src_norm, H_model.get_mano_closed_faces().numpy(), process=False)

        pl = pv.Plotter(shape=(1,2))
        pl.subplot(0,0)
        pl.add_mesh(O_src_mesh, color_src)
        pl.add_mesh(H_src, color_anc)
        
        pl.subplot(0,1)
        O_tgt_mesh = dataset_tgt.get_mesh_wt(i_tgt)
        O_tgt_mesh.vertices = (O_tgt_mesh.vertices - out_dict['O_tgt_loc']) / out_dict['O_tgt_scale']
        pl.add_mesh(O_tgt_mesh, color_tgt)
        pl.add_mesh(trimesh.Trimesh(out_dict['H_tgt_norm'], H_model.get_mano_closed_faces().numpy(), process=False), color_anc)
        pl.link_views()
        pl.show()
    
    # save results
    if save_results:
        sub_dir = f"{save_folder}/{i_src}_{i_tgt}"
        os.makedirs(sub_dir, exist_ok=True)

        O_tgt_mesh = dataset_tgt.get_mesh_wt(i_tgt)
        O_tgt_mesh.vertices = (O_tgt_mesh.vertices - out_dict['O_tgt_loc']) / out_dict['O_tgt_scale']
        O_tgt_mesh.export(f'{sub_dir}/{i_src}_{i_tgt}_tgt_obj.obj')
        H_tgt = trimesh.Trimesh(out_dict['H_tgt_norm'], H_model.get_mano_closed_faces().numpy(), process=False)
        H_tgt.export(f'{sub_dir}/{i_src}_{i_tgt}_tgt_H.obj')
        
        O_src_mesh = dataset_src.get_mesh(i_src)
        O_src_loc = out_dict['O_src_loc']
        O_src_scale = out_dict['O_src_scale']
        if 'O_src_rot' not in out_dict:
            O_src_rot = np.eye(3)
        else:
            O_src_rot = out_dict['O_src_rot']
        O_src_mesh.vertices = rotate_np((O_src_mesh.vertices - O_src_loc) / O_src_scale, O_src_rot)
        H_src = trimesh.Trimesh(H_src_norm, H_model.get_mano_closed_faces().numpy(), process=False)

        O_src_mesh.export(f'{sub_dir}/{i_src}_{i_tgt}_src_obj.obj')
        H_src.export(f'{sub_dir}/{i_src}_{i_tgt}_src_H.obj')

        torch.save(out_dict,f'{sub_dir}/{i_src}_{i_tgt}.pt')
        print(f'Saved to {sub_dir}/{i_src}_{i_tgt}.pt')
