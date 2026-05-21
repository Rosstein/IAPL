import torch
import torch.nn as nn


class DGPDL_Condition_Module(nn.Module):
    def __init__(self, n_ctx=2, vision_width=1024, cond_dim=512, scale_init=1e-6):
        super().__init__()
        hidden_dim = max(cond_dim // 2, 32)
        self.n_ctx = n_ctx
        self.vision_width = vision_width

        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, cond_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cond_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.core = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.bias_head = nn.Linear(cond_dim, n_ctx * vision_width)
        self.sup_head = nn.Linear(cond_dim, 1)
        self.scale = nn.Parameter(torch.full((1, n_ctx, vision_width), scale_init))

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.encoder(x).flatten(1)
        z = self.core(z)

        bias = self.bias_head(z).view(batch_size, self.n_ctx, self.vision_width)
        bias = bias * self.scale
        pred_bias = self.sup_head(z)

        return bias, pred_bias
