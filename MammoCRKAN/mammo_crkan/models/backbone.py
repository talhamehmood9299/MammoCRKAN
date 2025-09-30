
import torch
import torch.nn as nn
import torchvision.models as models

def _get_resnet(name: str, pretrained=False, in_ch=1):
    name = name.lower()
    if name == 'resnet18':
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        out_dim = 512
    elif name == 'resnet34':
        m = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
        out_dim = 512
    elif name == 'resnet50':
        m = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
        out_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone {name}")
    # change stem for 1-channel input
    if in_ch != 3:
        conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.conv1 = conv1
    # return feature extractor (drop avgpool/fc)
    modules = list(m.children())[:-2]  # keep spatial feature map
    feat = nn.Sequential(*modules)
    return feat, out_dim

class FeatureExtractor(nn.Module):
    def __init__(self, name='resnet50', in_ch=1):
        super().__init__()
        self.backbone, self.out_dim = _get_resnet(name, pretrained=False, in_ch=in_ch)

    def forward(self, x):
        return self.backbone(x)  # (B, C, H', W')
