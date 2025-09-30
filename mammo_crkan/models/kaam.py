
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import FeatureExtractor, _get_resnet
from .kan import KANUnivariateAggregator

def _saliency_from_feats(Fm):
    # category-agnostic saliency as L2 over channels
    # Fm: (B, C, H, W)
    return torch.sqrt((Fm**2).sum(dim=1, keepdim=True) + 1e-6)  # (B,1,H,W)

def _patch_sums(map2d, patch_hw):
    """Compute sliding window sums using avg_pool scaled by area."""
    ph, pw = patch_hw
    sums = F.avg_pool2d(map2d, kernel_size=(ph, pw), stride=1, padding=0) * (ph * pw)
    return sums  # (B,1,H-ph+1, W-pw+1)

def _crop(img, top, left, h, w):
    return img[..., top:top+h, left:left+w]

class MCPSelector:
    """
    Iteratively select M patch coordinates to maximize coverage (Budgeted Maximum Coverage).
    After each selection, reduce the saliency map within the selected region by factor (1 - r).
    """
    def __init__(self, patch_size=(512,512), num_patches=6, cover_rate=0.3):
        self.patch_h, self.patch_w = patch_size
        self.M = num_patches
        self.r = cover_rate

    def select(self, saliency):
        """
        saliency: (B,1,H,W) score map at feature resolution.
        Returns: list of coords [(top,left), ...] per batch element.
        """
        B, _, H, W = saliency.shape
        ph, pw = self.patch_h, self.patch_w
        coords_batch = []
        s = saliency.clone()
        # If feature map is too small, rescale patch size to fit
        scale_h = max(1, ph // 8)
        scale_w = max(1, pw // 8)
        # For selection, we operate at saliency resolution
        ph_f, pw_f = max(1, H // scale_h), max(1, W // scale_w)
        # limit patch to feature map size
        ph_f = min(H, max(2, ph_f))
        pw_f = min(W, max(2, pw_f))

        for b in range(B):
            sb = s[b:b+1]  # (1,1,H,W)
            coords = []
            for _ in range(self.M):
                sums = _patch_sums(sb, (ph_f, pw_f))  # (1,1,H-ph_f+1,W-pw_f+1)
                _, _, Hs, Ws = sums.shape
                if Hs <= 0 or Ws <= 0:
                    coords.append((0,0,ph_f,pw_f))
                    continue
                # argmax
                idx = torch.argmax(sums.view(-1)).item()
                ty = idx // Ws
                tx = idx % Ws
                top, left = int(ty), int(tx)
                coords.append((top, left, ph_f, pw_f))
                # attenuate inside selected region
                sb[:,:, top:top+ph_f, left:left+pw_f] *= (1.0 - self.r)
            coords_batch.append(coords)
        return coords_batch, (ph_f, pw_f)

class KAAM(nn.Module):
    def __init__(self, image_size=(2944,1920), in_ch=1, num_classes=3,
                 backbone_name='resnet18', patch_size=(512,512),
                 num_patches=6, cover_rate=0.3, temperature=0.5,
                 kan_cfg=None):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.selector = MCPSelector(patch_size=patch_size, num_patches=num_patches, cover_rate=cover_rate)

        # Backbone on patches
        self.backbone, out_dim = _get_resnet(backbone_name, pretrained=False, in_ch=in_ch)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # KAN-style aggregator to produce class logits per patch
        self.kan = KANUnivariateAggregator(out_dim, num_classes,
                                           num_bases=kan_cfg.get('num_bases', 10),
                                           degree=kan_cfg.get('degree', 3),
                                           init_range=tuple(kan_cfg.get('init_range', [-2.0, 2.0])))

    def extract_patch_features(self, patches):
        # patches: (B*M, C, H, W)
        feats = self.backbone(patches)
        feats = self.gap(feats).flatten(1)  # (B*M, D)
        return feats

    def forward(self, images_mlo, images_cc, feats_mlo, feats_cc):
        """
        images_mlo, images_cc: (B,1,H,W) original resized images
        feats_mlo, feats_cc:   (B,C,Hf,Wf) coarse features (used for MCP)
        returns: logits_P2: (B, num_classes)
        """
        B, _, H, W = images_mlo.shape

        # MCP selection on summed saliency across views
        sal_m = _saliency_from_feats(feats_mlo)
        sal_c = _saliency_from_feats(feats_cc)
        sal = sal_m + sal_c  # (B,1,Hf,Wf)
        coords_batch, (ph_f, pw_f) = self.selector.select(sal)

        # Map feature coords to image coords
        _, _, Hf, Wf = sal.shape
        scale_y = H / Hf
        scale_x = W / Wf
        patches = []
        for b, coords in enumerate(coords_batch):
            for (top, left, h_f, w_f) in coords:
                # feature-space to image-space box
                top_i = int(top * scale_y)
                left_i = int(left * scale_x)
                h_i = int(h_f * scale_y)
                w_i = int(w_f * scale_x)
                # crop both views, sum ("add") as in paper
                pm = _crop(images_mlo[b:b+1], top_i, left_i, h_i, w_i)
                pc = _crop(images_cc[b:b+1], top_i, left_i, h_i, w_i)
                # ensure same size; if small, pad
                Hc, Wc = pm.shape[-2:]
                if Hc < 8 or Wc < 8:
                    # too small, skip by upsampling smallest tile
                    pm = F.interpolate(pm, size=(32,32), mode='bilinear', align_corners=True)
                    pc = F.interpolate(pc, size=(32,32), mode='bilinear', align_corners=True)
                # add (average) two views, still (1,1,h,w)
                p = 0.5 * (pm + pc)
                # resnet expects 3 channels; replicate
                p = p.repeat(1, 3, 1, 1)
                # resize to a moderately sized patch for speed
                p = F.interpolate(p, size=(224,224), mode='bilinear', align_corners=True)
                patches.append(p)
        if len(patches) == 0:
            # fallback: global image as one patch
            p = 0.5 * (images_mlo + images_cc)
            p = p.repeat(1,3,1,1)
            p = F.interpolate(p, size=(224,224), mode='bilinear', align_corners=True)
            patches = [p[i:i+1] for i in range(B)]
        patches = torch.cat(patches, dim=0)  # (B*M,3,224,224)

        # KAN aggregator
        feat_vec = self.extract_patch_features(patches)  # (B*M, D)
        # reshape to (B, M, D)
        M = len(coords_batch[0]) if len(coords_batch) else 1
        feat_vec = feat_vec.view(B, M, -1)
        # compute h_j(F_j) per patch -> (B, M, C)
        h_vals = []
        for j in range(M):
            h_vals.append(self.kan(feat_vec[:, j, :]))
        h_vals = torch.stack(h_vals, dim=1)  # (B, M, C)
        # context gate: sigma(sum_l h_l)
        gate = torch.sigmoid(h_vals.sum(dim=1, keepdim=True))   # (B,1,C)
        p_j = gate * h_vals                                     # (B,M,C)

        # softmax weights ψ over patches using temperature τ
        logits_for_softmax = p_j / self.temperature
        psi = torch.softmax(logits_for_softmax.mean(dim=2), dim=1).unsqueeze(-1)  # (B,M,1)

        # aggregate to P2 logits
        P2 = (psi * p_j).sum(dim=1)  # (B, C)
        return P2
