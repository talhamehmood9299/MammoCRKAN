
import os, argparse, yaml, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.datasets import DualViewPairs
from .models.mammo_crkan import MammoCRKAN
from .models.loss import FocalLoss
from .utils.metrics import compute_metrics
from .utils.train_utils import set_seed, build_optimizer, build_scheduler, save_checkpoint

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--num_classes', type=int, default=None, help='Override config')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    if args.num_classes is not None:
        cfg['num_classes'] = args.num_classes

    set_seed(cfg.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outdir, exist_ok=True)

    # Data
    train_ds = DualViewPairs(args.train_csv, image_size=cfg['image_size'],
                             channels=cfg['channels'], right_orient_flag=cfg['right_orient'],
                             num_classes=cfg['num_classes'])
    val_ds = DualViewPairs(args.val_csv, image_size=cfg['image_size'],
                           channels=cfg['channels'], right_orient_flag=cfg['right_orient'],
                           num_classes=cfg['num_classes'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
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
    model = model.to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=cfg['focal_loss']['alpha'], gamma=cfg['focal_loss']['gamma']).to(device)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)

    best_val_auc = -1.0
    patience = cfg.get('early_stop_patience', 12)
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, cfg['epochs']+1):
        model.train()
        train_losses = []
        for (mlo, cc, y, _) in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]"):
            mlo, cc, y = mlo.to(device), cc.to(device), y.to(device)
            optimizer.zero_grad()
            P1, P2 = model(mlo, cc)
            loss = criterion(P1, y) + criterion(P2, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            global_step += 1
        if scheduler is not None:
            scheduler.step()

        # Validation
        if epoch % cfg.get('val_every', 1) == 0:
            model.eval()
            ys, probs = [], []
            with torch.no_grad():
                for (mlo, cc, y, _) in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [val]"):
                    mlo, cc = mlo.to(device), cc.to(device)
                    P1, P2 = model(mlo, cc)
                    # TDE(I) = P1 + psi * P2 ; paper uses sum; we aggregate logits then softmax for metrics
                    logits = P1 + P2
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    probs.append(prob)
                    ys.append(y.numpy())
            y_true = np.concatenate(ys, axis=0)
            y_prob = np.concatenate(probs, axis=0)
            metrics = compute_metrics(y_true, y_prob, num_classes=cfg['num_classes'])
            val_auc = metrics['auc']
            print(f"Val metrics: AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  F1={metrics['f1']:.4f}")

            # Early stopping and checkpoint
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                save_path = os.path.join(args.outdir, 'best.ckpt')
                save_checkpoint({
                    'model': model.state_dict(),
                    'best_val_auc': best_val_auc,
                    'epoch': epoch,
                    'cfg': cfg
                }, save_path)
                print(f"Saved best checkpoint to: {save_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improve {epochs_no_improve} epochs).")
                    break

        # Save periodic checkpoints
        if epoch % cfg.get('save_every', 1) == 0:
            ckpt_path = os.path.join(args.outdir, f'epoch_{epoch}.ckpt')
            save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'cfg': cfg}, ckpt_path)

    print("Training complete. Best Val AUC:", best_val_auc)

if __name__ == "__main__":
    main()
