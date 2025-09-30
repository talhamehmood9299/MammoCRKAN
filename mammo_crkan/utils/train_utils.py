
import os, math, torch, random, numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_optimizer(params, cfg):
    name = cfg['optimizer']['name'].lower()
    if name == 'adamw':
        opt = torch.optim.AdamW(params, lr=cfg['optimizer']['lr'],
                                weight_decay=cfg['optimizer']['weight_decay'])
    elif name == 'adam':
        opt = torch.optim.Adam(params, lr=cfg['optimizer']['lr'],
                               weight_decay=cfg['optimizer']['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer {name}")
    return opt

def build_scheduler(optimizer, cfg):
    name = cfg['scheduler']['name'].lower()
    if name == 'cosine':
        # T_max is epochs; step per epoch
        return CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['scheduler']['min_lr'])
    elif name == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler {name}")

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
