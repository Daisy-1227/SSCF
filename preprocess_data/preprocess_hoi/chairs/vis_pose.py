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

print("读取所有交互数据...")
data = {}
for f in os.listdir('../AHOI_Data/DATA_FOLDER'):
    key = f.split('.')[0]
    data[key] = ahoi_utils.load_data(f'../AHOI_Data/DATA_FOLDER/{f}')


seq_id = '1119'
view_id = '2'

print([img_name[:-4].split('_')[-1] for img_name in data['img_name'] if  f'{seq_id}/{view_id}/rgb_{seq_id}_{view_id}' in img_name])


frame_id = '0666'
img_name = f'{seq_id}/{view_id}/rgb_{seq_id}_{view_id}_{frame_id}.jpg'



hoi_id = data['img_name'].tolist().index(img_name)
# print("序列",seq_id,"视角",view_id,"帧",frame_id,"数组下标",hoi_id)
print(f'seq_id,view_id,frame_id,hoi_id={seq_id},{view_id},{frame_id},{hoi_id}')

kwargs = {}
for key in data: 
    if key in ['joint_prox','object_root_location','pare_human_betas','img_name','pare_cam','pare_bbox','object_root_rotation','pare_human_orient','pare_joints2d','pare_human_pose']:continue
    kwargs[key] = data[key][hoi_id]
torch_param,human_mesh,object_mesh = visualize.save_result(**kwargs)

pl = pv.Plotter()
pl.add_mesh(human_mesh)
pl.add_mesh(object_mesh)
pl.add_text(f'{seq_id}/{view_id}/{frame_id}')
pl.show()

