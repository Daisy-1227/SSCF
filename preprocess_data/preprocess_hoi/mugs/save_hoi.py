import os
import torch
import trimesh
import open3d as o3d
import numpy as np
import pyvista as pv
from manotorch.manolayer import ManoLayer, MANOOutput

import sys
sys.path.append('/mnt/d/projects/sscf/')

# Modify the path direct to your mano
mano_layer = ManoLayer(center_idx=0, mano_assets_root='/mnt/d/datasets/sscf/human_model/mano')
faces = mano_layer.get_mano_closed_faces().cpu().numpy()

from preprocess_data.preprocess_hoi.mugs.utils import (
    ALL_CAT,
    ALL_INTENT,
    ALL_SPLIT,
    CENTER_IDX,
    check_valid,
    get_hand_parameter,
    get_obj_path,
    to_list,
)


# Modify the path to your OakInkShape
data_dir = 'preprocess_data/preprocess_hoi/mugs/'
meta_dir = f'{data_dir}/metaV2'
real_dir = f'{data_dir}/OakInkObjectsV2'
virutal_dir = f'{data_dir}/OakInkVirtualObjectsV2'

# Modify the path to your target dir
save_dir = '/mnt/d/projects/sscf/data/hoi/'

# input the id of the object you want to save based on the provide imgs folder

save_dict = {
    "mugs": [x for x in range(439, 440)]
    # "object_type": [id_list]
}


grasp_list = torch.load('preprocess_data/preprocess_hoi/mugs/grasp_list_real.pt')

def get_obj_mesh(idx,use_downsample_mesh=False):
    obj_id = grasp_list[idx]['obj_id']
    obj_path = get_obj_path(obj_id, data_dir, meta_dir, use_downsample=use_downsample_mesh)
    obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
    bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
    obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
    return obj_trimesh

for cat in save_dict:
    ids = save_dict[cat]

    root = f'{save_dir}/{cat}'
    samples_dir = f'{root}/samples'
    mesh_dir = f'{root}/mesh'
    os.makedirs(samples_dir,exist_ok=True)
    os.makedirs(mesh_dir,exist_ok=True)

    names = []
    for i in ids:
        print(f'{cat}_{i}')
        grasp = grasp_list[i]
        obj_mesh = get_obj_mesh(i)

        object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_mesh.sample(100000)))
        object_pcd = np.array(object_pcd.farthest_point_down_sample(5000).points)
        object_pcd = torch.tensor(object_pcd).float()
        item = {
            'H_src_param': {
                'hand_pose':torch.tensor(grasp['hand_pose']).float(),
                'hand_shape':torch.tensor(grasp['hand_shape']).float(),
                'hand_tsl':torch.tensor(grasp['joints'][0]).float()
            },
            'O_src': object_pcd
        }

        # p = pv.Plotter()
        # p.add_mesh(obj_mesh)
        # p.add_mesh(pv.make_tri_mesh(grasp['verts'],faces))
        # p.add_text(f'{cat}_{i}')
        # p.show()

        torch.save(item, f'{samples_dir}/{i}.pt')
        obj_mesh.export(f'{mesh_dir}/{i}.ply')
        names.append(f'{i}')
    open(f'{root}/names.lst', 'w').write('\n'.join(names))
    