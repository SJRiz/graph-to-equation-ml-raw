import torch.nn as nn
from torchvision import models

class CNNModel(nn.Module):
    def __init__(self, num_params=7, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        feat_dim = 512  # resnet18 final feature dim

        # small attention (SE-style)
        self.se = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//4),
            nn.ReLU(),
            nn.Linear(feat_dim//4, feat_dim),
            nn.Sigmoid()
        )

        # shared head
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_params)  # predict normalized params
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # backbone: output shape (B, 512, 1, 1)
        f = self.backbone(x)
        f = f.view(f.size(0), -1)  # (B, 512)
        attn = self.se(f)
        f = f * attn
        s = self.shared(f)
        return self.regressor(s)
