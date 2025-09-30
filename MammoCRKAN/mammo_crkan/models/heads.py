
import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_dim, num_classes)
        )
    def forward(self, x):
        return self.head(x)
