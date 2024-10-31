import os
import numpy as np
import pyvista as pv
import pickle
import trimesh
import ahoi_utils
import visualize
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

print("read interaction data...")

interaction_data_folder = 'preprocess_data/preprocess_hoi/chairs/AHOI_Data/DATA_FOLDER'

data = {}
for f in os.listdir(interaction_data_folder):
    key = f.split('.')[0]
    data[key] = ahoi_utils.load_data(f'{interaction_data_folder}/{f}')

# select the ids you want
all_ids = [
    ['0241','2','0412'],
    ['1119','2','0491'],
]

# Define the target dir to save the hoi

workspace = "/mnt/d/projects/sscf"

root = f'{workspace}/data/hoi/chairs'
samples_dir = f'{root}/samples'
mesh_dir = f'{root}/mesh'
os.makedirs(samples_dir,exist_ok=True)
os.makedirs(mesh_dir,exist_ok=True)

names = []

for seq_id,view_id,frame_id in all_ids:
    if os.path.exists(f'{samples_dir}/{seq_id}_{view_id}_{frame_id}.pt'): 
        names.append(f'{seq_id}_{view_id}_{frame_id}')
        continue
    img_name = f'{seq_id}/{view_id}/rgb_{seq_id}_{view_id}_{frame_id}.jpg'
    hoi_id = data['img_name'].tolist().index(img_name)

    kwargs = {}
    for key in data: 
        if key in ['joint_prox','object_root_location','pare_human_betas', 'pare_img_name', 'img_name','pare_cam','pare_bbox','object_root_rotation','pare_human_orient','pare_joints2d','pare_human_pose']:continue
        kwargs[key] = data[key][hoi_id]
    torch_param,human_mesh,object_mesh = visualize.save_result(**kwargs)
    object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_mesh.sample(100000)))
    object_pcd = np.array(object_pcd.farthest_point_down_sample(5000).points)
    object_pcd = torch.tensor(object_pcd).float()
    item = {
        'H_src_param':torch_param,
        'O_src':object_pcd,
    }
    names.append(f'{seq_id}_{view_id}_{frame_id}')

    torch.save(item,f'{samples_dir}/{seq_id}_{view_id}_{frame_id}.pt')
    object_mesh.export(f'{mesh_dir}/{seq_id}_{view_id}_{frame_id}.ply')

    print(f'saved {seq_id}_{view_id}_{frame_id}')

    vis = False
    if vis:
        pl = pv.Plotter()
        pl.add_mesh(human_mesh)
        pl.add_mesh(object_mesh,opacity=0.5)
        pl.add_mesh(object_pcd.detach().numpy())
        pl.add_text(f'{seq_id}/{view_id}/{frame_id}')
        pl.show()

open(f'{root}/names.lst','w').write('\n'.join(names))