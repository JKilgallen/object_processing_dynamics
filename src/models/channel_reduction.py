from torch.nn import functional as F
from torch import nn
import torch
import math
from functools import partial

class ChannelSelector(nn.Module):
    def __init__(self, channels_in, channels_out, tau = 1e-8, scheduler_kwargs={}):
        super().__init__()
        self.channels_in, self.channels_out = channels_in, channels_out
        self.Wk = nn.Parameter(torch.empty(channels_in, channels_in))
        self.Wq = nn.Parameter(torch.empty(channels_in, channels_in))
        nn.init.normal_(self.Wk, mean=0.0, std=0.01); nn.init.normal_(self.Wq, mean=0.0, std=0.01)
        self.step = self.get_scheduler(tau, **scheduler_kwargs)

    def _step(self, *, tau_start, steps, eta_min):
        self._t = min(self._t + 1, steps)
        r = self._t / max(1, steps)
        self.tau = eta_min + 0.5 * (tau_start - eta_min) * (1 + math.cos(math.pi * r))
        return self.tau

    def get_scheduler(self, tau=1e-8, steps=100, eta_min=1e-8):
        self.tau, self._t = tau, 0
        return partial(self._step, tau_start=tau, steps=steps, eta_min=eta_min)
    
    def get_selections(self):
        sel_idxs = torch.topk(self.selection_logits, self.channels_out).indices
        return sel_idxs

    def forward(self, x):
        B, C, _ = x.shape
        s = x.mean(dim=2).unsqueeze(-1)
        Q = s * self.Wq
        K = s * self.Wk
        att = (Q @ K.transpose(-1, -2)) / math.sqrt(self.Wq.size(1))
        att = torch.softmax(att, dim=-1)
        scores = att.mean(dim=1)
        g = -torch.log(-torch.log(torch.rand_like(scores)))
        noisy = scores + self.tau * g
        idx = noisy.topk(self.channels_out, dim=-1).indices
        hard_mask = torch.zeros_like(scores).scatter(-1, idx, 1.0).view(B, C, 1)
        y_hard = x * hard_mask
        soft_mask = torch.softmax(scores / max(self.tau, 1e-8), dim=-1).view(B, C, 1)
        y_soft = x * soft_mask
        return y_hard + (y_soft - y_soft.detach())


    
class LogisticRegression(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, max_channels=None, tau=1e-8, scheduler_kwargs={}):
        super().__init__()
        max_channels = n_channels if max_channels is None else max_channels
        self.max_channels = max_channels
        self.selector = ChannelSelector(n_channels, max_channels, tau, scheduler_kwargs)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * n_timepoints, n_classes)
        )

    def forward(self, x):
        x = self.selector(x)
        return self.head(x)

class MLP(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, proj_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels*n_timepoints, proj_dim),
            nn.SELU(),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, hidden_dim),
                nn.SELU(),
                nn.Linear(hidden_dim, proj_dim),
                nn.SELU(),
                nn.AlphaDropout(0.1)) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(proj_dim, n_classes),
        )
    
    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = (x + layer(x)) * (2 ** -0.5)
        x = self.head(x)
        return x
    
class STConvNet(nn.Module):
    def __init__(self, n_classes, C, T, F, K, P, H, D=1, E=1, bidirectional=True, self_normalizing=False):
        super().__init__()      
        self.T = T; self.F = F; self.D=D; self.P = P          
        activation = nn.SELU() if self_normalizing else nn.ReLU()
        dropout = nn.FeatureAlphaDropout(0.05) if self_normalizing else nn.Dropout2d(0.5)
        
        dilations = [1, 2, 3, 5, 7, 11, 13]
        padding = [(d*(K-1))//2 for d in dilations[:D]]
        
        self.temporal_convs = nn.ModuleList([nn.Conv2d(1, F, (1, K), dilation = d, padding = (0, p)) for d, p in zip(dilations, padding)])
        self.activation = activation
        self.dropout = dropout

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F*D, F*D*P, (C, 1), groups=F*D),
            activation,
            dropout
        )
        
        self.filter_conv = nn.Sequential(
            nn.Conv2d(P, P*E, (F*D, 1)),
            activation,
            dropout
        )
        # self.rnn = nn.GRU(P*E, H, bidirectional=bidirectional, batch_first=True)
        # self.lstm = nn.LSTM(P,H,1,batch_first=True)
        self.head = nn.Sequential(
            activation,
            nn.Flatten(),
            nn.Linear(P*E*T, n_classes)
            # nn.Linear(2 * (2*H if bidirectional else H), n_classes)
        )

        def lecun_init(layer):            
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x):
        x = x.unsqueeze(1).to(memory_format=torch.channels_last)
        x = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        x = self.dropout(self.activation(x))
        x = self.spatial_conv(x)
        x = self.filter_conv(x.view(-1, self.P, self.F*self.D, self.T))
        # x, h = self.rnn(x.squeeze(2).mT)
        # h, _ = self.lstm(x.squeeze(2).mT)
        # out = torch.cat([torch.mean(x, dim=1), torch.amax(x, dim=1)], dim=1)
        return self.head(x)

