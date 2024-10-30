import numpy as np
import torch

def to_np(pt): return pt.detach().cpu().numpy()

def to_pt(np,device='cuda'): return torch.tensor(np).to(device)