from .deepsdf import DeepSDF
from .ndf import NDF
from .dif import DIF,RDIF
from .vnndif import VNNDIF
import torch

def get_model(name='vnndif',ckpt_path=None,**args):
    if name == 'deepsdf':
        model = DeepSDF()
    elif name == 'ndf':
        model = NDF()
    elif name == 'dif':
        model = DIF()
    elif name == 'vnndif':
        model = VNNDIF()
    elif name =='rdif':
        model = RDIF()
    else:
        raise NotImplementedError
    
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path,map_location='cpu')['model'])

    return model