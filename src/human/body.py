import torch
import smplx
import trimesh

def load_smplx(smplx_root='/mnt/d/datasets/sscf/human_model/smplx'):
    model = smplx.create(smplx_root, model_type='smplx',
        gender='male', ext='npz',
        num_betas=10,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=1,
    )
    return model


# def load_smplx_omomo(smplx_root='/mnt/d/sdb/honghao/siga/smplx'):
#     model = smplx.create(smplx_root, model_type='smplx',
#         gender='male', ext='npz',
#         num_betas=16,
#         flat_hand_mean=True,
#         use_pca=False,
#         create_global_orient=True,
#         create_body_pose=True,
#         create_betas=True,
#         create_left_hand_pose=True,
#         create_right_hand_pose=True,
#         create_expression=True,
#         create_jaw_pose=True,
#         create_leye_pose=True,
#         create_reye_pose=True,
#         create_transl=True,
#         batch_size=1,
#     )
#     return model

@torch.no_grad()
def param2mesh(model,param):
    device = model.shapedirs.device
    param = {key:param[key].to(device) for key in param}
    o = model(return_verts=True, **param)
    v = o.vertices[0].cpu().numpy()
    f = model.faces
    m = trimesh.Trimesh(vertices=v, faces=f,process=False)
    return m