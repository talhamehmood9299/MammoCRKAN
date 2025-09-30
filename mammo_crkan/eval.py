
import os, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.datasets import DualViewPairs
from .models.mammo_crkan import MammoCRKAN
from .utils.metrics import compute_metrics
from .utils.train_utils import load_checkpoint, set_seed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--test_csv', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--num_classes', type=int, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    if args.num_classes is not None:
        cfg['num_classes'] = args.num_classes

    set_seed(cfg.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = DualViewPairs(args.test_csv, image_size=cfg['image_size'],
                       channels=cfg['channels'], right_orient_flag=cfg['right_orient'],
                       num_classes=cfg['num_classes'])
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = MammoCRKAN(image_size=tuple(cfg['image_size']), in_ch=cfg['channels'],
                       num_classes=cfg['num_classes'],
                       backbone1=cfg.get('backbone1', 'resnet50'),
                       ssm_cfg=cfg.get('ssm', {}),
                       kaam_cfg=dict(
                           backbone=cfg.get('backbone2', 'resnet18'),
                           patch_size=tuple(cfg['kaam']['patch_size']),
                           num_patches=cfg['kaam']['num_patches'],
                           cover_rate=cfg['kaam']['cover_rate'],
                           temperature=cfg['kaam']['temperature'],
                           kan=cfg['kaam']['kan']
                       ))
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device)
    model.eval()

    ys, probs = [], []
    with torch.no_grad():
        for (mlo, cc, y, _) in tqdm(loader, desc=f"[test]"):
            mlo, cc = mlo.to(device), cc.to(device)
            P1, P2 = model(mlo, cc)
            logits = P1 + P2
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
            ys.append(y.numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    metrics = compute_metrics(y_true, y_prob, num_classes=cfg['num_classes'])
    print(f"Test metrics: AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  "
          f"Prec={metrics['precision']:.4f}  F1={metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
