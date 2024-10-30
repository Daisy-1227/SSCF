import torch
import numpy as np
from skimage.measure import marching_cubes

import trimesh
import pyvista as pv

def sdf2mesh(model,pcd=None,z=None,N=128,max_batch=64 ** 3,level=0.0,shrink=False,return_vtk=False,device='cuda'):
    # (botton,left,down)
    voxel_origin = [-.5, -.5, -.5]
    voxel_size = 1.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False

    with torch.no_grad():
        if pcd is not None:
            z = model.encoder(pcd[None].to(device))

        head = 0
        while head < num_samples:
            # print(head)
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(device)[None,...]
            samples[head : min(head + max_batch, num_samples), 3] = (
                model.inference(sample_subset,z=z)['sdf'].squeeze().detach().cpu()
            )
            head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    sdf_values = sdf_values.numpy()
    if return_vtk:
        vtk = pv.wrap(sdf_values)
        vtk.origin = voxel_origin
        vtk.dimensions = (N,N,N)
        vtk.spacing = (voxel_size,voxel_size,voxel_size)
        return vtk

    if sdf_values.min()<level:
        verts, faces, normals, values = marching_cubes(
            sdf_values, level=level, spacing=[voxel_size] * 3
        )
    else:
        level = sdf_values.min()+voxel_size
        verts, faces, normals, values = marching_cubes(
            sdf_values, level=level, spacing=[voxel_size] * 3
        )

    verts = voxel_origin+verts
    mesh = trimesh.Trimesh(verts,faces,process=False)
    if shrink:
        mesh.vertices-=mesh.vertex_normals*level
    return mesh


def template2mesh(model,N=128,max_batch=64 ** 3,scale=1.0,level=0,return_vtk=False,device='cuda'):
    # (botton,left,down)
    voxel_origin = [-.5, -.5, -.5]
    voxel_size = 1.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3
    samples.requires_grad = False
    samples = samples*scale


    with torch.no_grad():
        head = 0
        while head < num_samples:
            sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(device)[None,...]
            samples[head : min(head + max_batch, num_samples), 3] = (
                model.inference(sample_subset,template=True)['sdf'].squeeze().detach().cpu()
            )
            head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)
    sdf_values = sdf_values.numpy()

    if return_vtk:
        vtk = pv.wrap(sdf_values)
        vtk.origin = samples[0]
        vtk.dimensions = (N,N,N)
        vtk.spacing = (voxel_size*scale,voxel_size*scale,voxel_size*scale)
        return vtk

    if sdf_values.min()<level:
        verts, faces, normals, values = marching_cubes(
            sdf_values, level=level, spacing=[voxel_size] * 3
        )
    else:
        level = sdf_values.min()+voxel_size
        verts, faces, normals, values = marching_cubes(
            sdf_values, level=level, spacing=[voxel_size] * 3
        )

    verts = voxel_origin+verts
    mesh = trimesh.Trimesh(verts,faces,process=False)
    return mesh
