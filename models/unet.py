import torch
import torch.nn as nn
import torch.nn.functional as F
from config import T

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = nn.Conv2d(3, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up2 = nn.Conv2d(64 + 64, 3, 3, padding=1)
        self.time_emb = nn.Embedding(T, 128)

    def forward(self, x, t):
        t_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1)
        x1 = F.relu(self.down1(x))
        x2 = F.max_pool2d(F.relu(self.down2(x1)), 2)
        x_up = F.relu(self.up1(x2))
        x_up = torch.cat([x_up, x1[:, :, :x_up.shape[2], :x_up.shape[3]]], dim=1)
        return self.up2(x_up + t_emb)