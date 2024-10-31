import numpy as np
import pyvista as pv
import pickle
import trimesh

meta = pickle.load(open('../AHOI_Data/AHOI_ROOT/Metas/object_meta.pkl','rb'))
info = np.load('../AHOI_Data/AHOI_ROOT/Metas/object_info.npy',allow_pickle=True).item()

n_obj = len(info['face_len'])

pl = pv.Plotter(shape=(10,10),window_size=(1500,1500))
idx = 0
for i_obj in range(n_obj):
    oid = info['object_ids'][i_obj]
    # pl = pv.Plotter()
    mesh = trimesh.Trimesh()
    for i_part in range(7):
        n_face = info['face_len'][i_obj,i_part]
        if n_face==0:continue
        n_vert = info['vertex_len'][i_obj,i_part]
        f = info['faces'][i_obj,i_part,:n_face]
        v = info['vertices'][i_obj,i_part,:n_vert,:3]
        t = info['init_shift'][i_obj,i_part]
        m = trimesh.Trimesh(v+t,f)
        mesh+=m
    pl.subplot(idx//10,idx%10)
    pl.add_mesh(mesh)
    pl.add_text(f'{oid}')
    idx+=1
pl.link_views()
pl.show()