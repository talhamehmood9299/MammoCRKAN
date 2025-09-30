
import torch
import torch.nn as nn
from .geometry import SphericalSampleModule

class SSMAligner(nn.Module):
    """
    Given two feature maps (MLO and CC), apply SSM to each and return aligned features.
    """
    def __init__(self, in_ch, offset_channels=64, offset_scale=0.10):
        super().__init__()
        self.ssm_mlo = SphericalSampleModule(in_ch, offset_channels, offset_scale)
        self.ssm_cc  = SphericalSampleModule(in_ch, offset_channels, offset_scale)

    def forward(self, Fmlo, Fcc):
        Fo_mlo = self.ssm_mlo(Fmlo)
        Fo_cc  = self.ssm_cc(Fcc)
        return Fo_mlo, Fo_cc
