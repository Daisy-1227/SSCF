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
    print(key,data[key].shape)


oids = data['object_id']
u_oids = np.unique(oids)
print("交互物体数量",len(oids))

seqv_names = ['/'.join(n.split('/')[:2]) for n in data['img_name']]
u_seqv_names = np.unique(seqv_names)
print("交互序列数量",len(u_seqv_names))

seq_names = [n.split('/')[0] for n in seqv_names]
u_seq_names =   np.unique(seq_names)
print("不重复交互序列数量",len(seq_names))

print("每个物体对应的交互序列")
obj_dicts = {oid:[] for oid in oids}
for i in range(len(data['object_id'])):
    oid = oids[i]
    seq_name = seq_names[i]
    obj_dicts[oid].append(seq_name)

for oid in obj_dicts:
    obj_dicts[oid] = np.unique(obj_dicts[oid])
    print(oid,obj_dicts[oid])

open('seqs.txt','w').write('\n'.join([f'{oid}_'+','.join(obj_dicts[oid])   for oid in obj_dicts]    ))
