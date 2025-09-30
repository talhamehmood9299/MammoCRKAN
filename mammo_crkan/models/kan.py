
import torch
import torch.nn as nn
import torch.nn.functional as F

def bspline_basis(x, knots, degree=3):
    """
    Compute B-spline basis functions for each x given uniform knots.
    x: (B, D) arbitrary real values
    knots: (K,) sorted knot positions (1D tensor)
    degree: spline degree (default: cubic)
    Returns: (B, D, K+degree-1) basis values per (x,d).
    Implementation uses Cox–de Boor recursion.
    """
    # Ensure proper shapes
    B, D = x.shape
    K = knots.numel()
    # Extend knots to open uniform knot vector for degree d
    d = degree
    # Pad first and last knots
    t = F.pad(knots, (d, d), mode='replicate')  # (K + 2d)
    # basis for degree 0
    # For numeric stability, we use piecewise indicator with epsilon
    eps = 1e-8
    x_exp = x.unsqueeze(-1)  # (B, D, 1)
    # intervals [t[i], t[i+1])
    B0_list = []
    for i in range(K + d - 1):
        left  = t[i]
        right = t[i+1]
        Bi = ((x_exp >= (left - eps)) & (x_exp < (right + eps))).float()
        B0_list.append(Bi)
    Bk = torch.stack(B0_list, dim=-1)  # (B, D, K+d-1)

    # Recursion for higher degrees
    def cdb(Bprev, k):
        # k: current degree
        size = Bprev.shape[-1] - 1  # number of segments - 1
        out = []
        for i in range(size):
            denom1 = (t[i+k] - t[i]) + eps
            denom2 = (t[i+k+1] - t[i+1]) + eps
            a = ((x_exp - t[i]) / denom1) * Bprev[..., i]
            b = ((t[i+k+1] - x_exp) / denom2) * Bprev[..., i+1]
            out.append(a + b)
        return torch.stack(out, dim=-1)

    Bcur = Bk
    for deg in range(1, d+1):
        Bcur = cdb(Bcur, deg)
    return Bcur  # (B, D, K+d-1 - d) = (B, D, K-1)  ~ K-1 effective bases

class KANUnivariateAggregator(nn.Module):
    """
    Map feature vectors to class logits via per-dimension B-spline bases
    and linear combination (Kolmogorov–Arnold style: sum of univariates).
    """
    def __init__(self, in_dim, out_dim, num_bases=10, degree=3, init_range=(-2.0, 2.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bases = num_bases
        self.degree = degree
        self.knots = nn.Parameter(torch.linspace(init_range[0], init_range[1], num_bases))
        # linear weights per (dim, basis) to out_dim
        self.weight = nn.Parameter(torch.randn(in_dim, num_bases-1, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        """
        x: (B, D) feature vector
        returns: (B, out_dim) logits
        """
        # Normalize x roughly into knot range
        xm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        bases = bspline_basis(xm, self.knots, degree=self.degree)  # (B, D, K-1)
        # Weighted sum over (D, K-1)
        # out[b, c] = sum_d sum_k  bases[b, d, k] * W[d, k, c] + b[c]
        B, D, K1 = bases.shape
        W = self.weight  # (D, K1, C)
        # ensure shapes match
        if W.shape[0] != D or W.shape[1] != K1:
            # re-init if dimension changed dynamically
            W = nn.Parameter(torch.randn(D, K1, self.out_dim) * 0.01).to(bases.device)
            self.weight = W
        out = torch.einsum('bdk,dkc->bc', bases, self.weight) + self.bias
        return out
