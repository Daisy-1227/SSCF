import os
import numpy as np
import trimesh
import point_cloud_utils as pcu
import sys
sys.path.append('/mnt/d/projects/sscf/')
from src.utils.warp_utils_np import closest_point
import pyvista as pv
from joblib import Parallel, delayed

def create_one(mesh_path, npz_path, wt_path,sampling='nearfar',manifold_resolution=20_000, num_vol_pts= 100_000,num_surf_pts=100_000):
    
    v, f = pcu.load_mesh_vf(mesh_path)

    # Convert mesh to watertight manifold
    vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution,num_vol_pts)
    pcu.save_mesh_vf(wt_path, vm, fm)

    # if os.path.exists(npz_path): return
    # Convert mesh to watertight manifold
    m = trimesh.load(wt_path,process=False,maintain_order=True)
    vm, fm = m.vertices, m.faces
    fm = m.face_normals 

    # Generate random points in the volume around the shape
    # NOTE: ShapeNet shapes are normalized within [-0.5, 0.5]^3
    if sampling=='uniform':
        p_vol = (np.random.rand(num_vol_pts, 3) - 0.5) * 1.1
    else:
        stddev = 0.05
        p_vol_far = (np.random.rand(num_vol_pts//2, 3) - 0.5)
        # fid_near, bc_near = pcu.sample_mesh_random(vm, fm, num_vol_pts//2)
        p_vol_near = m.sample(num_vol_pts//2)+stddev * np.random.randn(num_vol_pts//2,3)
        p_vol = np.concatenate((p_vol_far, p_vol_near), axis=0)

    # Comput the SDF of the random points
    # sdf, _, _  = pcu.signed_distance_to_mesh(p_vol, vm, fm)
    closest,distance,triangle_id,sign = closest_point(m,p_vol)
    sdf = (distance*sign).astype('float32')


    # Sample points on the surface as face ids and barycentric coordinates
    # fid_surf, bc_surf = pcu.sample_mesh_random(vm, fm, num_surf_pts)
    p_surf,fid_surf = m.sample(num_vol_pts,return_index=True)

    # Compute 3D coordinates and normals of surface samples
    # n_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, nm)
    n_surf = fm[fid_surf]

    # pl = pv.Plotter()
    # pl.add_mesh(pv.make_tri_mesh(v,f))
    # pl.add_mesh(p_surf,scalars=(n_surf+1)/2,rgb=True)
    # pl.add_mesh(p_vol,scalars=sdf)
    # pl.show()

    # Save volume points + SDF and surface points + normals
    # Load using np.load()
    np.savez(npz_path, p_vol=p_vol, sdf_vol=sdf, p_surf=p_surf, n_surf=n_surf)
    print(f'save: {npz_path}')


# TODO: Modify the path to your ShapeNet dataset

# cid = '03001627'
cid = '03797390'
shapenet_dir = '/mnt/d/datasets/ShapeNet/ShapeNetCore.v1'
category_dir = f'{shapenet_dir}/{cid}'
dataset_dir = f'data/shapenet'
samples_dir = f'{dataset_dir}/samples/{cid}'
watertight_dir = f'{dataset_dir}/watertight/{cid}'
lst_dir = f'{dataset_dir}/lst'
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(watertight_dir, exist_ok=True)
os.makedirs(lst_dir, exist_ok=True)

lst = []
tasks = []
for model_dir in os.listdir(category_dir):
    mesh_path =f'{category_dir}/{model_dir}/model.obj'
    if os.path.exists(mesh_path): lst.append(model_dir)
    npz_path = f'{samples_dir}/{model_dir}.npz'
    wt_path = f'{watertight_dir}/{model_dir}.ply'
    tasks.append([mesh_path,npz_path,wt_path])
    # create_one(npz_path,wt_path)

# open(f'{lst_dir}/{cid}.lst','w').write('\n'.join(lst))
with open(f'{lst_dir}/{cid}.lst', 'w') as f:
    f.write('\n'.join(lst))


with Parallel(n_jobs=8) as parallel:
    parallel(delayed(create_one)(mesh_path, npz_path, wt_path) for mesh_path,npz_path,wt_path in tasks)