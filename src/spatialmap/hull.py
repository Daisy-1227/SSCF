import numpy as np
import pyigl
import trimesh
from scipy.spatial import cKDTree as KDTree
from trimesh import creation
from ..utils.warp_utils_np import closest_point
import pyvista as pv
import pyvista as pv

def mesh_to_sdf(mesh,dim=32,return_vtk=False):
    box = mesh.bounding_box
    c = box.centroid
    s = box.scale
    box = np.array([c-s,c+s])
    GV,D = pyigl.voxel_grid_box(box,dim,0)
    S = pyigl.signed_distance(GV,mesh.vertices,mesh.faces,pyigl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)[0]

    if return_vtk:
        vtk = pv.wrap(S.reshape(*D))
        vtk.origin = GV[0]
        vtk.dimensions = D
        vs = ((GV[-1]-GV[0])/(D-1)).max()
        vtk.spacing = (vs,vs,vs)
        return vtk
    return S,GV,D

def sdf_to_genus0(S,GV,D):
    th = ((GV[-1]-GV[0])/(D-1)).max()
    iso_max = S.max()
    
    iso_high = S.max()
    iso_low = 0
    last_iso = iso_high
    while True:
        if (iso_high-iso_low)<th: break
        iso = (iso_high+iso_low)/2
        V,F = pyigl.marching_cubes(S,GV,D[0],D[1],D[2],iso)
        C = pyigl.facet_components_number(F)
        
        if C>1 and iso<iso_max/2:
            iso_low = iso
            # print(iso,"Part")
            continue
        E = pyigl.euler_characteristic_complete(V,F)
        if -(E-2)/2!=0 and iso<iso_max/2:
            # print(iso,"Hole")
            iso_low = iso
            continue
        # print(iso,"Large")
        iso_high = iso

        last = trimesh.Trimesh(V,F,process=False)
        last_iso = iso
    return last,iso

def sdf_to_genus0_carving(S,GV,D,iso):
    th = ((GV[-1]-GV[0])/(D-1)).max()
    kdtree = KDTree(GV)
    S = S.squeeze()
    O = np.ones_like(S)
    O[S>iso] = 0 
    queue = np.argsort(S)[::-1]
    queue = queue[O[queue]==1]
    n = len(queue)
    i = 0
    last = None

    while True:
        # print(i,n)
        idx = queue[i]
        s = S[idx]
        if s<=th:
            break
        o = O[idx]
        if o==0:
            i+=1
            continue
        v = GV[idx]
        d,nn = kdtree.query(v,n-i,distance_upper_bound=s,workers=-1)
        nn = nn[d<=s]
        nn = nn[S[nn]>th]
        O_tmp  = O.copy()
        O_tmp[nn] = 0
        V,F = pyigl.marching_cubes(O_tmp,GV,D[0],D[1],D[2],0)
        # pv.plot(pv.make_tri_mesh(V,F))
        if len(V)==0 or len(F)==0:
            i+=1
            continue
        C = pyigl.facet_components_number(F)
        if C>1:
            i+=1
            continue
        E = pyigl.euler_characteristic_complete(V,F)
        if -(E-2)/2!=0:
            i+=1
            continue
        last = trimesh.Trimesh(V,F,process=False)
        O = O_tmp
        i+=1
    return last

def genus0_shrink(mesh,g0,k=3):
    g0_shrink = g0.copy()
    for i in range(k):
        nn = closest_point(mesh,g0_shrink.vertices)[0]
        g0_shrink.vertices = g0_shrink.vertices+0.5*(nn-g0_shrink.vertices)
        g0_shrink = trimesh.smoothing.filter_humphrey(g0_shrink)
    return g0_shrink

def mesh_to_genus0(mesh):
    S,GV,D = mesh_to_sdf(mesh)
    g0,iso = sdf_to_genus0(S,GV,D)
    g0_carving = sdf_to_genus0_carving(S,GV,D,iso)
    if g0_carving is not None: g0 = g0_carving
    g0 = genus0_shrink(mesh,g0)
    return g0

def genus0_to_tet(g0):
    # sphere = creation.uv_sphere(1.5)
    sphere = creation.icosphere(radius=1.5)
    sphere.apply_translation(g0.centroid)
    sphere.faces = sphere.faces[:,::-1]
    mesh_all = g0+sphere

    P = g0.sample(1000)+np.random.normal(0,0.005,(1000,3))
    S = pyigl.signed_distance(P,g0.vertices,g0.faces,pyigl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0].squeeze()
    holes =  np.ascontiguousarray(P[S<0])
    TV,TT,TF,TR,TN,PT,FT,n = pyigl.tetrahedralize(
        np.ascontiguousarray(mesh_all.vertices),
        np.ascontiguousarray(mesh_all.faces),holes,[],'pYqa0.001')
    # tet = pv.PolyData(TV,
    #     np.concatenate([np.full((len(TT),1),4),TT],axis=-1).flatten()
    # )
    return TV,TT

def tet_deform(tet_v,tet_f,src_v,tgt_v):
    nsphere = 642
    b = np.arange(len(src_v)+nsphere)
    db = np.concatenate([tgt_v-src_v,np.zeros((nsphere,3))],axis=0)
    d = pyigl.harmonic(tet_v.astype('float32'),tet_f.astype('int32'),b.astype('int32'),db.astype('float32'),2)
    tet_v_new = tet_v+d
    return tet_v_new

def world_to_tet(tet_v,tet_f,points):
    I = pyigl.in_element(tet_v,tet_f,points)
    tet_tri = tet_v[tet_f][I]
    points_tet = pyigl.barycentric_coordinates_tet(points,tet_tri[:,0],tet_tri[:,1],tet_tri[:,2],tet_tri[:,3])
    return I,points_tet

def tet_to_world(tet_v,tet_f,I,points_tet):
    tet_tri = tet_v[tet_f][I]
    points = tet_tri[:,0]*points_tet[:,[0]]+tet_tri[:,1]*points_tet[:,[1]]+tet_tri[:,2]*points_tet[:,[2]]+tet_tri[:,3]*points_tet[:,[3]]
    return points
