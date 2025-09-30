
import torch
import torch.nn as nn
from .backbone import FeatureExtractor
from .ssm import SSMAligner
from .heads import ClassifierHead
from .kaam import KAAM

class MammoCRKAN(nn.Module):
    def __init__(self, image_size=(2944,1920), in_ch=1, num_classes=3,
                 backbone1='resnet50',
                 ssm_cfg=None, kaam_cfg=None):
        super().__init__()
        self.extractor = FeatureExtractor(backbone1, in_ch=in_ch)
        self.ssm = SSMAligner(self.extractor.out_dim,
                              offset_channels=ssm_cfg.get('offset_channels', 64),
                              offset_scale=ssm_cfg.get('offset_scale', 0.10))
        # concat aligned features -> classifier
        self.cls = ClassifierHead(self.extractor.out_dim * 2, num_classes)

        # KAAM branch
        self.kaam = KAAM(image_size=image_size, in_ch=in_ch, num_classes=num_classes,
                         backbone_name=kaam_cfg.get('backbone', 'resnet18'),
                         patch_size=tuple(kaam_cfg.get('patch_size', (512,512))),
                         num_patches=int(kaam_cfg.get('num_patches', 6)),
                         cover_rate=float(kaam_cfg.get('cover_rate', 0.3)),
                         temperature=float(kaam_cfg.get('temperature', 0.5)),
                         kan_cfg=kaam_cfg.get('kan', {}))

    def forward(self, img_mlo, img_cc):
        # Extract coarse features
        Fmlo = self.extractor(img_mlo)  # (B,C,Hf,Wf)
        Fcc  = self.extractor(img_cc)

        # SSM alignment for direct effect P1
        Fo_mlo, Fo_cc = self.ssm(Fmlo, Fcc)
        P1 = self.cls(torch.cat([Fo_mlo, Fo_cc], dim=1))

        # KAAM for mediation effect P2 (uses images + features)
        P2 = self.kaam(img_mlo, img_cc, Fmlo, Fcc)
        return P1, P2
