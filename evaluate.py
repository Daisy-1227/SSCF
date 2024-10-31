import os
import torch
import sys
sys.append('/mnt/d/projects/sscf')

from src.datasets.chairs import ChairsDataset
from src.datasets.shapenet import ShapeNetTestDataset
from src.human.body import load_smplx, param2mesh
import numpy as np
import pyvista as pv
import trimesh
from src.utils.rot_utils import rotate_np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
from trimesh import creation
from tqdm import tqdm

# TODO: the following code snippet is an example of evaluating the generated results of chairs, 
# for the mug dataset you can easily modify the H_model, dataset_src, and dataset_tgt

H_model = load_smplx(smplx_root='/mnt/d/datasets/sscf/human_model/smplx')
dataset_src = ChairsDataset(f'/mnt/d/datasets/sscf/hoi/chairs')
dataset_tgt = ShapeNetTestDataset('/mnt/d/datasets/sscf/shapenet', '03001627')
color_anc = 'w'
color_src = '#15c4ed'
color_tgt = '#fe7a89'

# Modify the src_interaction_num and tgt_object_num based on your own generated data
# We use 10 source interactions and 100 target objects in this example
src_interaction_num, tgt_object_num = 10, 100

depth_pen = np.zeros([src_interaction_num, tgt_object_num])
volume_isect = np.zeros([src_interaction_num, tgt_object_num])
contact_sim = np.zeros([src_interaction_num, tgt_object_num])

# TODO: define your own tgt_list and src_list
tgt_list = [i for i in range(tgt_object_num)]
src_list = [i for i in range(src_interaction_num)]

for i_view, i_tgt in enumerate(tgt_list):
    print(f'tgt_id:{i_view}')
    O_tgt_mesh = dataset_tgt.get_mesh_wt(i_tgt)

    grid = creation.box().bounding_box.sample_grid(128).astype('float32')
    grid_sdf = pcu.signed_distance_to_mesh(grid, O_tgt_mesh.vertices.astype('float32'), O_tgt_mesh.faces)[0]
    grid_inside = grid[grid_sdf <= 0]

    for i_src in range(src_interaction_num):
        # define the path to your own generated results
        out_dict = torch.load(f'/mnt/d/sdb/honghao/siga/opt2/results/chairs/{i_tgt}_{i_src}.pt')
        data_src = dataset_src[i_src]
        H_src_param = data_src['H_src_param']
        H_tgt_param = out_dict['H_tgt_param']
        # key_pose[i_method, i_src, i_view] = (H_tgt_param['body_pose'] - H_src_param['body_pose']).square().sum()
        O_src = data_src['O_src']
        O_src_mesh = dataset_src.get_mesh(i_src)
        O_src_loc = out_dict['O_src_loc']
        O_src_scale = out_dict['O_src_scale']
        if 'O_src_rot' not in out_dict:
            O_src_rot = np.eye(3)
        else:
            O_src_rot = out_dict['O_src_rot']
        O_src_mesh.vertices = rotate_np((O_src_mesh.vertices - O_src_loc) / O_src_scale, O_src_rot)

        O_src_norm = out_dict['O_src_norm']
        O_tgt_norm = out_dict['O_tgt_norm']
        H_src_norm = out_dict['H_src_norm']
        H_tgt_norm = out_dict['H_tgt_norm']

        O_tgt = dataset_tgt[i_tgt]
        O_tgt_mesh = dataset_tgt.get_mesh_wt(i_tgt)
        O_tgt_loc = out_dict['O_tgt_loc']
        O_tgt_scale = out_dict['O_tgt_scale']
        O_tgt_mesh.vertices = (O_tgt_mesh.vertices - O_tgt_loc) / O_tgt_scale

        # Note: if the method is 'cpd' or 'cpdt', we need to rotate the source human vertices
        # if method in ['cpd', 'cpdt']:
        #     O_src_norm = rotate_np((O_src - O_src_loc) / O_src_scale, O_src_rot)
        #     O_tgt_norm = (O_tgt - O_tgt_loc) / O_tgt_scale
        #     O_tgt_norm = O_tgt_norm.numpy()

        grid_inside_norm = (grid_inside - O_tgt_loc) / O_tgt_scale

        H_src = trimesh.Trimesh(H_src_norm, H_model.faces, process=False)
        H_tgt = trimesh.Trimesh(H_tgt_norm, H_model.faces, process=False)

        d_h_src = pcu.k_nearest_neighbors(H_src_norm, O_src_norm, 1)[0].squeeze()
        d_h_tgt = pcu.k_nearest_neighbors(H_tgt_norm, O_tgt_norm, 1)[0].squeeze()

        # for the mug, please set d_th to 0.002
        d_th = 0.02
        contact_src = ((d_h_src * O_src_scale) < d_th)
        contact_tgt = ((d_h_tgt * O_src_scale) < d_th)
        # contact_sim[i_method, i_idx, i_view] = (contact_src * contact_tgt).sum() / (
        #         np.linalg.norm(contact_src) * np.linalg.norm(contact_tgt) + 1e-9)
        contact_sim[i_src, i_view] = (contact_src & contact_tgt).sum() / (
                    (contact_src | contact_tgt).sum() + 1e-9)
        # contact_src = ((d_h_src * O_src_scale) < 0.05)
        # contact_tgt = ((d_h_tgt * O_src_scale) < 0.05)
        # contact_sim[i_method, i_src, i_view] = (contact_src * contact_tgt).sum() / (
        #             np.linalg.norm(contact_src) * np.linalg.norm(contact_tgt) + 1e-9)

        grid_inside_sdf = pcu.signed_distance_to_mesh(grid_inside_norm, H_tgt_norm, H_model.faces.astype('int32'))[0]
        grid_inside_inside_num = (grid_inside_sdf <= 0).sum()
        # pl = pv.Plotter()
        # pl.add_mesh(H_tgt,opacity=0.5)
        # pl.add_mesh(O_tgt_mesh,opacity=0.1)
        # pl.add_mesh(grid_inside[grid_inside_sdf<0],scalars=grid_inside_sdf[grid_inside_sdf<0])
        # pl.show()

        v_sdf = pcu.signed_distance_to_mesh(O_tgt_mesh.vertices.astype('float32'), H_tgt_norm,
                                            H_model.faces.astype('int32'))[0]
        # pl = pv.Plotter()
        # pl.add_mesh(H_tgt,opacity=0.5)
        # pl.add_mesh(O_tgt_mesh,opacity=0.1)
        # pl.add_mesh(O_tgt_mesh.vertices[v_sdf<0],scalars=v_sdf[v_sdf<0])
        # pl.show()

        depth_pen[i_src, i_view] = -np.clip(v_sdf, None, 0).min() * O_src_scale
        volume_isect[i_src, i_view] = grid_inside_inside_num * (1 / 127 * O_src_scale) ** 3

print(f"depth_pen: {depth_pen.mean()}, volume_isect: {volume_isect.mean()}, contact_sim: {contact_sim.mean()}")
