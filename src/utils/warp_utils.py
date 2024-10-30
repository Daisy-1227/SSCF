import numpy as np
import torch
import warp as wp
wp.init()

@wp.kernel
def cast_rays_kernel(
    mesh:wp.uint64,
    starts:wp.array(dtype=wp.vec3),
    dirs:wp.array(dtype=wp.vec3),
    max_dist:wp.float32,
    dists:wp.array(dtype=wp.float32)):
    """kernal for cast rays

    :param wp.uint64 mesh: warp mesh id
    :param wp.array starts: start points of rays
    :param wp.array dirs: directions of rays
    :param wp.float32 max_dist: max distance to check intersection
    :param wp.array dists: output array
    """
    
    tid = wp.tid()
    # input
    start = starts[tid]
    dir = dirs[tid]
    max_t = max_dist
    # output
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    hit = wp.mesh_query_ray(
        mesh,start,dir,max_t,
        t,u,v,sign,n,f)
    
    if hit:
        dists[tid] = t
    else:
        dists[tid] = max_dist

def cast_rays(mesh,rays,max_dist=torch.inf):
    """cast rays on mesh using cuda

    :param Trimesh mesh: mesh
    :param array rays: (n,6), [[start,direction]]
    :param float max_dist: defaults to inf
    :return dists: hit distances
    """
    device = rays.device.type
    n = len(rays)
    mesh_warp = wp.Mesh(
        points=wp.array(mesh.vertices,dtype=wp.vec3,device=device),
        indices=wp.array(mesh.faces.reshape(-1),dtype=wp.int32,device=device)
    )
    starts = wp.from_torch(rays[...,:3].contiguous(),dtype=wp.vec3)
    dirs = wp.from_torch(rays[...,3:].contiguous(),dtype=wp.vec3)
    dists = wp.from_torch(torch.full((n,),torch.inf,device=device).float())

    wp.launch(
        kernel=cast_rays_kernel,
        dim=n,
        inputs=[mesh_warp.id,starts,dirs,max_dist,dists],device=device)
    wp.synchronize_device()

    return wp.to_torch(dists)


@wp.kernel
def closest_point_kernel(
    mesh:wp.uint64,
    points:wp.array(dtype=wp.vec3),
    max_dist:wp.float32,
    closest:wp.array(dtype=wp.vec3),
    distance:wp.array(dtype=wp.float32),
    triangle_id:wp.array(dtype=wp.int32),
    sign:wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    # input
    point = points[tid]
    # output
    inside = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point(mesh,point,max_dist,inside,face,u,v)
    nn = wp.mesh_eval_position(mesh,face,u,v)

    closest[tid] = nn
    distance[tid] = wp.length(nn-point)
    triangle_id[tid] = face
    sign[tid] = inside


def closest_point(mesh,points,max_dist=float('inf'),device='cuda'):
    """nearest-point query on mesh using cuda

    :param Trimesh mesh: mesh
    :param array points: (n,3)
    :param float max_dist:  defaults to float('inf')
    :return tuple: (closest,distance,triangle_id,sign)
    """
    n = len(points)
    mesh_warp = wp.Mesh(
        points=wp.array(mesh.vertices,dtype=wp.vec3,device=device),
        indices=wp.array(mesh.faces.reshape(-1),dtype=wp.int32,device=device)
    )
    points_warp = wp.from_torch(points,dtype=wp.vec3)

    closest = wp.zeros(n,dtype=wp.vec3,device=device)
    distance = wp.zeros(n,dtype=wp.float32,device=device)
    triangle_id = wp.zeros(n,dtype=wp.int32,device=device)
    sign = wp.zeros(n,dtype=wp.float32,device=device)

    wp.launch(
        kernel=closest_point_kernel,
        dim=n,
        inputs=[mesh_warp.id,points_warp,float('inf'),closest,distance,triangle_id,sign],device=device)
    wp.synchronize_device()

    return wp.to_torch(closest),wp.to_torch(distance),wp.to_torch(triangle_id),wp.to_torch(sign)