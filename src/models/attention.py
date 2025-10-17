import torch, torch.nn as nn, torch.nn.functional as F
from mne.channels import make_standard_montage

class SpatialMHA(torch.nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()
        self.mha=torch.nn.MultiheadAttention(emb_dim, heads, batch_first=True)
        
        pos = torch.load("electrode_positions.pt")
        dist = torch.cdist(pos,pos)
        self.register_buffer('spatial_bias', torch.exp(-(dist**2)/(2*(dist.median(**2)))))

    def forward(self,x):
        h, _ = self.mha(x, x, x, attn_mask=self.spatial_bias)
        return x + h
    
class STMHA(torch.nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, emb_dim=128, n_heads=4):
        super().__init__()

        self.proj=torch.nn.Linear(n_timepoints, emb_dim)
        self.attn=SpatialMHA(emb_dim, n_heads)
        self.norm=torch.nn.LayerNorm(emb_dim)
        self.ffn=torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4*emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*emb_dim, emb_dim))
        self.head=torch.nn.Linear(2*emb_dim, n_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(self.attn(x))
        x = self.ffn(x)
        x = torch.cat([x.mean(dim=1), x.amax(dim=1)])
        return self.head(x)