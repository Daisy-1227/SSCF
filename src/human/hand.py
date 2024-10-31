import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import trimesh
from pytorch3d.transforms import Transform3d

def load_mano(mano_assets_root='/mnt/d/projects/sscf/data/human_model/mano'):
    model = ManoLayer(rot_mode='quat',center_idx=0, mano_assets_root=mano_assets_root)
    return model

def forward_mano(model,hand_pose=None,hand_shape=None,hand_tsl=None,hand_tsf=None):
    if hand_pose.dim() == 2: 
        hand_pose = hand_pose.unsqueeze(0)
        hand_shape = hand_shape.unsqueeze(0)
        hand_tsl = hand_tsl.unsqueeze(0)
        if hand_tsf is not None: hand_tsf = hand_tsf.unsqueeze(0)
    mano_output: MANOOutput = model(pose_coeffs=hand_pose,betas=hand_shape)
    verts = mano_output.verts+hand_tsl
    joints = mano_output.joints+hand_tsl
    if hand_tsf is not None:
        T = Transform3d(matrix=hand_tsf.permute(0,2,1))
        verts = T.transform_points(verts)
        joints = T.transform_points(joints)
    mano_output = mano_output._replace(verts = verts)
    mano_output = mano_output._replace(joints = joints)
    return mano_output

@torch.no_grad()
def param2mesh(model,param):
    device = model.shapedirs.device
    param = {key:param[key].to(device) for key in param}
    o = model(return_verts=True, **param)
    v = o.vertices[0].cpu().numpy()
    f = model.faces
    m = trimesh.Trimesh(vertices=v, faces=f,process=False)
    return m

