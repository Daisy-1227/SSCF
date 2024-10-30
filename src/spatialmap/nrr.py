
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
from pytorch3d.io import load_ply,load_obj
from pytorch3d.ops import sample_points_from_meshes,sample_farthest_points,knn_points
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing
import numpy as np

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import pyvista as pv

def trimesh_to_pytorch3d(mesh):
    v = torch.tensor(mesh.vertices).float().cuda()
    f = torch.tensor(mesh.faces).long().cuda()
    m = Meshes([v],[f])
    return m

class DeformationGraph():
    def __init__(self,mesh,mesh_target=None,
        n_sample=10000,n_node=500, n_neighbor=4,
        # w_rot=1.,w_reg=10.,w_con=100.,w_conf=100.,
        # w_rot = 1., w_reg=10. ,w_con=100. ,w_conf=100., # shrink
        # w_rot = 1., w_reg=.1 ,w_con=.1 ,w_conf=100., # register
        w_rot=1000,w_reg=100,w_conf=100,w_con=0.1,
        n_iter=500) -> None:
        # w_rot=1.,w_reg=10.,w_con=100.
        # w_rot=1000,w_reg=100,w_conf=100,w_con=0.1

        self.mesh = trimesh_to_pytorch3d(mesh)
        self.mesh_target = trimesh_to_pytorch3d(mesh_target)
        sample_points = sample_points_from_meshes(self.mesh,n_sample)
        self.nodes = sample_farthest_points(sample_points,K=n_node)[0].squeeze()

        knn = knn_points(self.mesh.verts_padded(),self.nodes[None],K=n_neighbor+1)
        dists = knn.dists.squeeze().sqrt()
        self.knn_idx = knn.idx.squeeze()[...,:-1]
        self.weights = (1-dists[...,:-1]/dists[...,[-1]]).square()
        self.weights = self.weights/self.weights.sum(-1,keepdims=True)
        self.edges = torch.cat([torch.combinations(idx) for idx in self.knn_idx])

        self.point_target = sample_points_from_meshes(self.mesh_target,n_sample).squeeze()
        knn = knn_points(self.nodes[None],self.point_target[None],K=1)
        self.corr = self.point_target[knn.idx.squeeze()]
        # d_corr = knn.dists.squeeze().sqrt()
        self.w = torch.ones(len(self.nodes)).to(self.nodes)

        self.com = sample_points.squeeze().mean(0)
        self.R_global = torch.eye(3).to(self.nodes)
        self.t_global = torch.zeros(3).to(self.nodes)
        self.R = torch.eye(3).reshape(1,3,3).repeat([n_node,1,1]).to(self.nodes)
        self.t = torch.zeros(3).reshape(1,3).repeat([n_node,1]).to(self.nodes)
        
        self.w_rot = w_rot
        self.w_reg = w_reg
        self.w_con = w_con
        self.w_conf = w_conf
        self.n_iter = n_iter
        # self.n_iter_in = n_iter_in
        # self.n_iter_out = n_iter_out
        
        self.mesh_reg = mesh.copy()
    
    def get_Erot(self):
        R = torch.cat([self.R_global[None],self.R])
        c1 = R[...,0]
        c2 = R[...,1]
        c3 = R[...,2]
        Erot = (c1*c2).sum(-1).square()
        Erot = Erot+(c1*c3).sum(-1).square()
        Erot = Erot+(c2*c3).sum(-1).square()
        Erot = Erot+((c1*c1).sum(-1)-1).square()
        Erot = Erot+((c2*c2).sum(-1)-1).square()
        Erot = Erot+((c3*c3).sum(-1)-1).square()
        return Erot.sum()

    def get_Ereg(self):
        i = self.edges[...,0]
        k = self.edges[...,1]
        g_j = self.nodes[i]
        g_k = self.nodes[k]
        R_j = self.R[i]
        R_k = self.R[k]
        t_j = self.t[i]
        t_k = self.t[k]
        Ereg = (torch.einsum('nij,nj->ni',R_j,g_k-g_j)+(g_j+t_j)-(g_k+t_k)).square().sum(-1)
        Ereg = Ereg+(torch.einsum('nij,nj->ni',R_k,g_j-g_k)+(g_k+t_k)-(g_j+t_j)).square().sum(-1)
        return Ereg.sum()

    def get_Econ(self):
        nodes = self.nodes
        nodes = nodes+self.t
        nodes = torch.einsum('ij,vj->vi',self.R_global,nodes-self.com[None])+self.com[None]+self.t_global[None]
        Econ = (self.w.square())*((nodes-self.corr).square().sum(-1))
        return Econ.sum()

    def get_Elap(self):
        v = self.mesh.verts_packed()
        nodes = self.nodes[self.knn_idx]
        v = v[:,None]-nodes
        v = torch.einsum('vnij,vnj->vni',self.R[self.knn_idx],v)+nodes+self.t[self.knn_idx]
        v = torch.einsum('vn,vni->vi',self.weights,v)
        v = torch.einsum('ij,vj->vi',self.R_global,v-self.com)+self.com[None]+self.t_global[None]
        Elap = mesh_laplacian_smoothing(Meshes([v],[self.mesh.faces_packed()]),'cotcurv')
        return Elap

    def get_Econf(self):
        return (1-self.w.square()).square().sum()
        
    def register(self):
        params = [self.R,self.t,self.R_global,self.t_global,self.w]
        for i in range(len(params)):
            params[i].requires_grad_(True)
        opt = optim.Adam([self.R,self.t,self.R_global,self.t_global,self.w],lr=1e-1)
        # opt = optim.LBFGS([self.R,self.t,self.R_global,self.t_global,self.w],lr=1.0)
        # def create_closure(self,opt):
        def closure():
            opt.zero_grad()
            Econ = self.get_Econ()
            Erot = self.get_Erot()
            Ereg = self.get_Ereg()
            Econf = self.get_Econf()
            E =self.w_rot*Erot+self.w_reg*Ereg+self.w_con*Econ+self.w_conf*Econf
            E.backward()
            return E.item()
            # return closure
        # closure = create_closure(self,opt)
        loss = 0
        step = 0 
        params_last = [p.detach().clone() for p in params]
        while step<self.n_iter:
            loss_last = loss
            params_last = [p.detach().clone() for p in params]
            loss = opt.step(closure)
            if not np.isfinite(loss):
                self.R,self.t,self.R_global,self.t_global,self.w = params_last
                break
            delta = np.abs(loss-loss_last)
            if delta<1e-5*(1+loss):
                print(step,'lr decay')
                if self.w_rot>1.: self.w_rot*=0.5
                if self.w_reg>0.1: self.w_reg*=0.5
                if self.w_conf>1.: self.w_conf*=0.5
                # for g in opt.param_groups:
                #     if g['lr']>1e-6:
                #         g['lr'] = g['lr'] * 0.5
            if delta<1e-6*(1+loss) or (step+1)%(self.n_iter//3)==0:
                print(step,'step converge')
                with torch.no_grad():
                    nodes = self.nodes
                    nodes = nodes+self.t
                    nodes = torch.einsum('ij,vj->vi',self.R_global,nodes-self.com[None])+self.com[None]+self.t_global[None]
                    knn = knn_points(self.nodes[None],self.point_target[None],K=1)
                    self.corr = self.point_target[knn.idx.squeeze()]
                    d_corr = knn.dists.squeeze().sqrt()
                self.w.requires_grad_(False)
                self.w.data = torch.ones(len(self.nodes)).to(self.nodes)
                # self.w[d_corr>0.02] = 0
                self.w.requires_grad_(True)
                # opt = optim.LBFGS([self.R,self.t,self.R_global,self.t_global,self.w],lr=1.0)
                # closure = create_closure(self,opt)
            if delta<1e-8*(1+loss):
                print(step,'final converge')
                break
            step+=1
            print(step,loss)

        for i in range(len(params)):
            params[i].requires_grad_(False)

        v = self.deform()
        self.mesh_reg.vertices = v
    
    def deform(self):
        v = self.mesh.verts_packed()
        nodes = self.nodes[self.knn_idx]
        v = v[:,None]-nodes
        v = torch.einsum('vnij,vnj->vni',self.R[self.knn_idx],v)+nodes+self.t[self.knn_idx]
        v = torch.einsum('vn,vni->vi',self.weights,v)
        v = torch.einsum('ij,vj->vi',self.R_global,v-self.com)+self.com[None]+self.t_global[None]
        return v.cpu().numpy()

class DeformLaplacian():
    def __init__(self,src_mesh ,trg_mesh,
                 Niter=250,w_chamfer=1.0,w_edge=1.0,w_normal=0.01,w_laplacian=0.1
                 ) -> None:
        self.src_mesh = trimesh_to_pytorch3d(src_mesh)
        self.trg_mesh = trimesh_to_pytorch3d(trg_mesh)
        self.deform_verts = torch.full(self.src_mesh.verts_packed().shape, 0.0, device='cuda', requires_grad=True)
        self.deform_mesh = src_mesh.copy()

        self.Niter = Niter
        self.w_chamfer = w_chamfer
        self.w_edge = w_edge
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian


    def deform(self):
        optimizer = torch.optim.SGD([self.deform_verts], lr=1.0, momentum=0.9)
        # Number of optimization steps
        Niter = self.Niter
        # Weight for the chamfer loss
        w_chamfer = self.w_chamfer 
        # Weight for mesh edge loss
        w_edge = self.w_edge
        # Weight for mesh normal consistency
        w_normal = self.w_normal
        # Weight for mesh laplacian smoothing
        w_laplacian = self.w_laplacian
        loop = range(Niter)

        chamfer_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []

        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            
            # Deform the mesh
            new_src_mesh = self.src_mesh.offset_verts(self.deform_verts)
            
            # We sample 5k points from the surface of each mesh 
            sample_trg = sample_points_from_meshes(self.trg_mesh, 5000)
            sample_src = sample_points_from_meshes(new_src_mesh, 5000)
            
            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
            
            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)
            
            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)
            
            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            
            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
            
            # Print the losses
            # loop.set_description('total_loss = %.6f' % loss)
            
            # Save the losses for plotting
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            normal_losses.append(float(loss_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))
                
            # Optimization step
            loss.backward()
            optimizer.step()
            # print(i,loss.item())
        
        with torch.no_grad():
            new_src_mesh = self.src_mesh.offset_verts(self.deform_verts)
            v = new_src_mesh.verts_packed().cpu().numpy()
            f = new_src_mesh.faces_packed().cpu().numpy()
            m = trimesh.Trimesh(v,f,process=False)
        return m