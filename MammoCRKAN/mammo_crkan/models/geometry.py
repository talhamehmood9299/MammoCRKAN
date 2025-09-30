
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def spherical_grid(H, W, device):
    """
    Create spherical coordinate grid alpha in [-pi, pi], beta in [-pi/2, pi/2].
    """
    y = torch.linspace(0, H-1, H, device=device)
    x = torch.linspace(0, W-1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    alpha = (xx - (W-1)/2.0) / W * 2.0 * math.pi  # [-pi, pi]
    beta  = -(yy - (H-1)/2.0) / H * math.pi       # [-pi/2, pi/2]
    return alpha, beta

def spherical_features(alpha, beta):
    # G(α, β) = (sin α, cos α, sin β, cos β, cos α cos β)
    sa, ca = torch.sin(alpha), torch.cos(alpha)
    sb, cb = torch.sin(beta),  torch.cos(beta)
    cab = torch.cos(alpha) * torch.cos(beta)
    G = torch.stack([sa, ca, sb, cb, cab], dim=0)  # (5, H, W)
    return G

class SphericalSampleModule(nn.Module):
    """
    Implements SSM: learn offsets (Δα, Δβ) over spherical features and resample.
    """
    def __init__(self, in_ch, offset_channels=64, offset_scale=0.10):
        super().__init__()
        self.offset_scale = offset_scale
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + 5, offset_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(offset_channels, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, Fm):
        """
        Fm: (B, C, H, W) feature map
        returns warped features with spherical alignment.
        """
        B, C, H, W = Fm.shape
        device = Fm.device
        alpha, beta = spherical_grid(H, W, device)              # (H, W)
        G = spherical_features(alpha, beta).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,5,H,W)
        x = torch.cat([Fm, G], dim=1)                           # (B,C+5,H,W)
        offsets = self.conv(x) * self.offset_scale              # (B,2,H,W) in normalized units

        # Build base grid in normalized coords for grid_sample: x,y in [-1,1]
        # Map spherical (alpha,beta) to normalized grid: gx = alpha/pi, gy = 2*beta/pi
        gx = (alpha / math.pi).expand(B, 1, H, W)
        gy = ((2.0 * beta / math.pi)).expand(B, 1, H, W)
        base_grid = torch.cat([gx, gy], dim=1)                  # (B,2,H,W)
        grid = (base_grid + offsets).permute(0,2,3,1)           # (B,H,W,2)

        Fo = F.grid_sample(Fm, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return Fo
