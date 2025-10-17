import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from einops.layers.torch import Rearrange
from einops import rearrange
from pywt import DiscreteContinuousWavelet
from scipy.fft import next_fast_len
import numpy as np 
from matplotlib import pyplot as plt
import os

import torch.utils.data.dataloader
OCCIPITAL_ELECTRODES = {'HGSN_128': [65, 66, 67, 68, 69, 70, 71,
                                     72, 73, 74, 75, 76, 77, 81,
                                     82, 83, 84, 88, 89, 90, 94],
                        'BIOSEMI_96': [i - 1 for i in [83, 82, 81, 80, 79, 
                                                       85, 86, 87, 88, 89, 
                                                       94, 93, 92, 90, 
                                                       95, 96, 91]],
                        None: []}

def bands_to_scales(bands, sampling_freq, central_freq, freqs_per_band = 5):
    freqs = torch.cat([
        torch.logspace(np.log2(low_f), np.log2(high_f), freqs_per_band, base=2) for (low_f, high_f) in bands.values()
    ])
    return torch.flip(sampling_freq * central_freq / freqs, dims=(0, )), freqs

class CWT(nn.Module):
    def __init__(self, n_timepoints, sampling_freq, wavelet='cmor', central_freq=1.0, bandwidth=0.5, bands={'full_spectrum': (5, 25)}, freqs_per_band=32, device='cpu'):
        super().__init__()
        wavelet = f'{wavelet}{bandwidth}-{central_freq}'
        self.sampling_freq = sampling_freq
        self.freqs_per_band = freqs_per_band
        self.device = device

        kernel_path = os.path.join("data", 'misc', "cwt_kernels", f"Full@{sampling_freq}Hz_{freqs_per_band}", f'{wavelet}.pt')
        if os.path.exists(kernel_path):
            kernels, scales, freqs = torch.load(kernel_path, map_location=device).values()
            self.N = kernels.shape[-1]
        else:
            psi, x, = DiscreteContinuousWavelet(wavelet).wavefun(10)
            int_psi = torch.conj(torch.cumsum(torch.tensor(psi, dtype=torch.complex64, device=device), dim=0) * (x[1] - x[0]))

            scales, freqs = bands_to_scales(bands, sampling_freq, central_freq, freqs_per_band)
            scales = scales.to(device); freqs = freqs.to(device)
            scales_idxs = [(torch.arange(scale * (x[-1] - x[0]) + 1, device=device)/ (scale * (x[1] - x[0]))).long() for scale in scales]

            kernels = [torch.flip(int_psi[s[s<len(int_psi)]], dims=(0,)) for s in scales_idxs]
            kernel_lengths = torch.tensor([k.numel() for k in kernels], dtype=torch.long, device=device)
            
            self.N = next_fast_len(n_timepoints + kernel_lengths.max() - 1)
            kernels = torch.fft.fft(torch.stack([F.pad(k, (0, kernel_lengths.max() - k.numel())) for k in kernels], dim=0), n=self.N)
            tdx = (kernel_lengths// 2 - 1)[:, None] + (torch.arange(n_timepoints+1, device=device)[None, :])
            tdx = torch.einsum('n, lt -> ltn', torch.arange(self.N, device=device), tdx)
            kernels = kernels[:, None, :] * (torch.exp(2j * torch.pi * tdx / self.N))

            kernels = kernels.reshape(-1, self.N)
            os.makedirs(os.path.dirname(kernel_path), exist_ok=True)
            # torch.save({'kernels': kernels, 
            #             'scales': scales,
            #             'freqs': freqs}, kernel_path)

        self.register_buffer('kernels', kernels)
        self.register_buffer('scales', scales ** 0.5)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        b,c,t=x.shape        
        x_fft = torch.fft.fft(x, n=self.N).to(torch.complex64)
        
        out = x_fft.view(-1, self.N) @ self.kernels.T
        
        out = out.view(b, c, -1, t+1)/self.N
        out = self.scales[None, None, :, None] * torch.diff(out, dim=-1)

        return torch.log(out.abs())

class CWTNet(nn.Module):
    def __init__(self, n_classes, C, T, S, Hz, K, Fch, Fsp, P, cwt=True):
        super().__init__()
        self.C = C; self.T = T

        self.proj = nn.Identity()
        # self.proj = nn.Sequential(
        #     nn.Conv2d(Fch, Fch, kernel_size=1, groups=Fch, bias=True),
        #     nn.SELU()
        # ) 

        if cwt:
            self.cwt = CWT(T, Hz, freqs_per_band=S)
            self.spectral_conv = nn.Sequential(
                # nn.Conv2d(C, C*Fch, kernel_size=(K,K), groups=C, padding='same'),
                # nn.SELU(),
                # nn.Conv2d(C*Fch, C*Fch, kernel_size=(S,1), groups=C*Fch, padding='valid'),        
                nn.Conv2d(C, C*Fch, kernel_size=(S,1), groups=C, padding='valid'),                
                nn.SELU(),
            )
        else:
            self.cwt = None
            
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, Fch, kernel_size=(1, K), padding='valid'),
            nn.SELU()
        )        
        
        self.scale = 2**-0.5
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(Fch, Fch*Fsp, (C, K), stride=K//2, groups=Fch),
            nn.SELU(),
            # nn.Conv2d(Fch*Fsp, Fch*Fsp, (1, K), stride=(1, K//2)),
            # nn.SELU()
        )
        P = ((T - K + 1) - K)//(K//2) + 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(Fch*Fsp*P, n_classes),
        )

    def forward(self, x):
        x_temporal = self.temporal_conv(x.unsqueeze(1)).contiguous(memory_format=torch.channels_last)#bfct
        if self.cwt is not None:
            x_spectral = self.cwt(x).contiguous(memory_format=torch.channels_last)#bcst->b(cf)1t            
            x_spectral = rearrange(self.spectral_conv(x_spectral), 'b (c f) 1 t -> b f c t', c=self.C)        
            x_temporal = (self.proj(x_temporal) + self.proj(x_spectral)) * self.scale
        
        x_fused = self.spatial_conv(x_temporal) #b(fs)1t
        out = self.fc(x_fused)
        return out

class ConvBlock(nn.Module):
    def __init__(self, C, T, K, Fch, Fsp, D, self_normalizing=True):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, Fch, kernel_size=(1, K), padding='same', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm2d(Fch))

        dilations = [2*(ddx+1)+1 for ddx in range(D)]
        padding = [(0, ((K-1)//2)*d) for d in dilations]
        self.multiscale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(Fch, Fch, kernel_size=(1, K), dilation=d, groups=Fch, padding=p),
                nn.SELU() if self_normalizing else nn.BatchNorm2d(Fch),)             
            for d,p in zip(dilations, padding)])

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(Fch, Fch * Fsp, kernel_size=(C, 1), groups=Fch, bias=False), 
            nn.SELU() if self_normalizing else nn.BatchNorm2d(Fch*Fsp),
            nn.AlphaDropout(0.1) if self_normalizing else nn.Dropout(0.5),)
        
        self.residual_scale = (2**-0.5) if self_normalizing else 1.0        
        # self.pool = 
            
    def forward(self, x):
        x = self.temporal_conv(x)     
        for conv in self.multiscale_convs:
            x = (x + conv(x))*self.residual_scale   
        x = self.spatial_conv(x)        
        return x.squeeze(2)

class SelfNormAttention(nn.Module):
    def __init__(self, emb_dim, heads=1, p=0.1):
        super().__init__()
        self.H, self.D = heads, emb_dim//heads
        self.scale = (self.D)**-0.5
        
        # self.q_proj = nn.Conv1d(emb_dim, emb_dim, 1, bias=False)
        # self.k_proj = nn.Conv1d(emb_dim, emb_dim, 1, bias=False)
        # self.v_proj = nn.Conv1d(emb_dim, emb_dim, 1, bias=False)        
        
        self.out = nn.Conv1d(emb_dim, emb_dim, 1, bias=False)
        self.act = nn.SELU()
        self.drop = nn.AlphaDropout(p)

    def forward(self, q, k, v):
        # q = self.act(self.q_proj(q))
        # k = self.act(self.k_proj(k))
        # v = self.act(self.v_proj(v))
        
        B, C, Tq = q.shape
        _, _, Tk = k.shape
        q = q.view(B, self.H, self.D, Tq).permute(0, 1, 3, 2)
        k = k.view(B, self.H, self.D, Tk).permute(0, 1, 2, 3)
        v = v.view(B, self.H, self.D, Tk).permute(0, 1, 3, 2)
        
        a = (q @ k) * self.scale
        y = (F.softmax(a, dim=-1) @ v).permute(0,1,3,2).reshape(B,C,Tq)
        return self.drop(self.act(self.out(y)))

class OCSNCNN(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, filters_per_channel, temporal_kernel_size, spatial_proj_per_filter, n_dilations, montage=None, occipital_stream=True, self_normalizing=True):
        super().__init__()
        
        C = n_channels
        T = n_timepoints
        K = temporal_kernel_size
        Fch = filters_per_channel 
        Fsp = spatial_proj_per_filter
        D = n_dilations
        P = 8

        self.residual_scale = (2 ** -0.5) if self_normalizing else 1.0

        
        self.global_stream = ConvBlock(C, T, K, Fch, Fsp, D, self_normalizing=self_normalizing)
        if occipital_stream:
            self.register_buffer('occipital_electrodes', torch.tensor(OCCIPITAL_ELECTRODES[montage]).long())            
            O = self.occipital_electrodes.size(0)
            # self.register_buffer('global_electrodes', torch.tensor([idx for idx in range(C) if idx not in self.occipital_electrodes]).long())
            # self.global_stream = ConvBlock(C-O, T, K, Fch, Fsp, D, self_normalizing=self_normalizing)
            self.occipital_stream = ConvBlock(O, T, K, Fch, Fsp, D, self_normalizing=self_normalizing)
            self.cross_attn = SelfNormAttention(Fch*Fsp, Fsp)
        else:
            self.occipital_stream is None
            

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(P),     
            nn.Flatten(),         
            nn.Linear(Fch*Fsp*P, n_classes),
        )

        def lecun_init(layer):            
            if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        if self_normalizing:
            self.apply(lecun_init)
        
    def forward(self, x, softmax=False):
        x = x.unsqueeze(1).contiguous(memory_format=torch.channels_last)
        if self.occipital_stream is not None:
            x_global = self.global_stream(x)#[:, :, self.global_electrodes])
            x_occipital = self.occipital_stream(x[:, :, self.occipital_electrodes])
                   
            x_global = self.cross_attn(x_occipital, x_global, x_global)
            x_global = (x_global + x_occipital) * self.residual_scale
            

        x_global = self.proj(x_global)
        if softmax:
            return F.softmax(x_global, dim=-1)
        else:
            return x_global

class CGWAN(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, n_heads):
        super().__init__()
        self.T=n_timepoints
        self.C=n_channels

        self.temporal_conv = nn.Conv1d(n_channels, n_channels, 5, groups=n_channels, padding='same')
        occipital_idxs = torch.tensor(OCCIPITAL_ELECTRODES['HGSN_128']).long()
        self.register_buffer('occipital_electrodes', occipital_idxs)
        
        self.spatial_attn = nn.MultiheadAttention(n_timepoints, n_heads, batch_first=True)
        spatial_attn_mask = torch.load("geodesic_attn_mask.pt").float()
        self.register_buffer("spatial_mask", spatial_attn_mask)

        self.temporal_attn = nn.MultiheadAttention(n_channels, n_heads, batch_first=True, kdim=len(occipital_idxs), vdim=len(occipital_idxs))
        temporal_attn_mask = torch.triu(torch.ones(n_timepoints, n_timepoints, dtype=torch.bool), diagonal=1)
        self.register_buffer("temporal_mask", temporal_attn_mask)

        self.cross_mha = nn.MultiheadAttention(n_timepoints, n_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((16,16)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(16*16, n_classes)
        )

    def forward(self, x):
        B, C, T = x.shape
        x = self.temporal_conv(x)

        # x_spatial = rearrange(x, 'b c f t -> b c (f t)') 
        
        # x_spatial = rearrange(x_spatial, 'b c (f t) -> b f c t', t=self.T)

        # x_temporal = x.transpose(1,2)
        # x_global, x_occipital =x_temporal, x[:, :, self.occipital_electrodes]
        # x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)#, attn_mask=self.temporal_mask)
        # x_temporal = x_temporal.transpose(1,2)
        # x_occipital = x[:, self.occipital_electrodes]
        h, _ = self.spatial_attn(x, x, x, attn_mask=self.spatial_mask)

        # h, _ = self.cross_mha(x_spatial, x_temporal, x_temporal)
        # x_temporal = rearrange(x_temporal, 'b t (c f) -> b f c t', c=self.C)
        
        # h = x_spatial + x_temporal
        # h = h
        h = self.fc(h)
        return h

class OccipitallyGuidedAttention(nn.Module):
    def __init__(self, n_global, n_occipital, d_model, n_heads, kernel_size, activation, dropout):
        super().__init__()  
        self.H = n_heads
        self.C = n_global
        self.q = nn.Conv1d(n_occipital, d_model, kernel_size, padding='same', bias=False)
        self.k = nn.Conv1d(n_global, n_global * d_model, kernel_size, groups=n_global, padding='same', bias=False)
        self.v = nn.Conv1d(n_global, n_global * d_model, kernel_size, groups=n_global, padding='same', bias=False)
    
        self.proj = nn.Conv1d(n_global * d_model, n_global, 1, groups=n_global, bias=False)
        self.scale = (d_model//n_heads) ** -0.5
        self.activation = activation()
        self.dropout=dropout(0.1)

        
    def forward(self, x_global, x_occipital):
        Q = self.activation(self.q(x_occipital))
        K = self.activation(self.k(x_global))
        V = self.activation(self.v(x_global))
        Qh = rearrange(Q, 'b (h d) t -> b h d t', h=self.H)
        Kh = rearrange(K, 'b (c h d) t -> b h c d t', c=self.C, h=self.H)
        Vh = rearrange(V, 'b (c h d) t -> b h c d t', c=self.C, h=self.H)
        gates = self.activation(torch.einsum('bhdt,bhcdt->bhct', Qh, Kh) * self.scale)
        out = torch.einsum('bhct,bhcdt->bhcdt', gates, Vh)
        out = rearrange(out, 'b h c d t -> b (c h d) t')
        out = self.activation(self.proj(out))
    
        return out


class OAT(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, d_model, n_heads, temporal_kernel_size, n_layers = 1, montage=None, self_normalizing=True):
        super().__init__()
        
        C = n_channels
        T = n_timepoints
        K = temporal_kernel_size
        D = d_model 
        H = n_heads
        self.L = n_layers

        self.register_buffer('occipital_electrodes', torch.tensor(OCCIPITAL_ELECTRODES[montage]).long())
        O = self.occipital_electrodes.size(0)

        activation = nn.SELU if self_normalizing else nn.ReLU
        dropout_layer = (lambda p: nn.AlphaDropout(p) if self_normalizing else nn.Dropout(p))
        self.attention = nn.ModuleList([OccipitallyGuidedAttention(C, O, D, H, K, activation, dropout_layer) for _ in range(n_layers)])
        self.residual_scale = (2 ** -0.5) if self_normalizing else 1.0

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C*T, n_classes)
        )

        def lecun_init(layer):            
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x):

        for l in range(self.L):
            x_occipital = x[:, self.occipital_electrodes].contiguous()
            x = (x + self.attention[l](x, x_occipital)) * self.residual_scale

        out = self.head(x)
        return out

class ProjLayer(nn.Module):
    def __init__(self, C_in, C_out, T_in, T_out, p_drop=0.1, self_normalizing=True):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size=1), 
            nn.SELU() if self_normalizing else nn.ReLU(), 
            nn.AlphaDropout(p_drop) if self_normalizing else nn.Dropout(p_drop)
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(T_in, T_out), 
            nn.SELU() if self_normalizing else nn.ReLU(), 
        )
    def forward(self, x):        
        return self.temporal_proj(self.spatial_proj(x))

class StreamAttention(nn.Module):
    def __init__(self, emb_dim, n_heads=1):
        super().__init__()
        self.q_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
            )
        self.k_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
            )
        self.v_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
            )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SELU(),
            nn.AlphaDropout(0.1)
            )
        self.emb_dim = emb_dim
        self.n_heads = n_heads 
        self.head_dim = emb_dim//n_heads
        self.attn_scale = (emb_dim**-0.5)
        self.res_scale = (2**-0.5)

    def _split(self, x): 
        return x.view(-1, x.shape[1], self.n_heads, self.head_dim).permute(0,2,1,3)

    def _merge(self, x):  
        return x.permute(0,2,1,3).contiguous().view(-1, x.shape[2], self.emb_dim)

    def forward(self, q, k, v):
        res = q
        q, k, v = self.q_emb(q.mT), self.k_emb(k.mT), self.v_emb(v.mT)
        # q, k, v = q.mT, k.mT, v.mT
        q, k, v = self._split(q), self._split(k), self._split(v)        
        attn = (q @ k.mT) * self.attn_scale
        out = self._merge(F.softmax(attn, dim=-1) @ v)
        out = (res + self.proj(out).mT)*self.res_scale
        return out

class DualStreamFFSNN(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, hidden_spatial_dim, hidden_temporal_dim, depth=2, n_heads=8, p_drop=0.1, montage=None, self_normalizing=True, dual_stream=True):
        super().__init__()
        C, T = n_channels, n_timepoints
        projC, projT = hidden_spatial_dim, hidden_temporal_dim
        self.n_streams = 2 if dual_stream else 1
        
        aux_idxs = torch.tensor(OCCIPITAL_ELECTRODES[montage if dual_stream else None]).long()              
        core_idxs = torch.tensor([idx for idx in range(C) if idx not in aux_idxs]).long()
        self.register_buffer("aux_idxs", core_idxs)  
        self.register_buffer("core_idxs", aux_idxs)


        self.proj_core = ProjLayer(len(self.core_idxs), projC, T, projT, p_drop=p_drop, self_normalizing=self_normalizing) 
        self.proj_aux = ProjLayer(len(self.aux_idxs), projC, T, projT, p_drop=p_drop, self_normalizing=self_normalizing) if dual_stream else nn.Identity()

        self.scale = (2**-0.5) if (dual_stream and self_normalizing) else 1.0

        # encoder/decoder stacks
        self.enc_core, self.enc_aux, self.enc_fuse = self._build_streams(
            projC, projT, depth, p_drop, encode=True, 
            self_normalizing=self_normalizing, dual_stream=dual_stream
        )
        self.bottleneck_fusion = StreamAttention(projC//(2**depth), n_heads=4)
        self.dec_core, self.dec_aux, self.dec_fuse = self._build_streams(
            projC, projT, depth, p_drop, encode=False,
            self_normalizing=self_normalizing, dual_stream=dual_stream
        )
        
        self.fusion = StreamAttention(projC, n_heads)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(projC//(2**depth) * projT//(2**depth), n_classes)
        )

        def lecun_init(layer):            
            if isinstance(layer, (nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        self.apply(lecun_init)

    def _build_streams(self, startC, startT, depth, p_drop, encode=True, self_normalizing=True, dual_stream=True):
        def get_progression(dim_in, depth, encode=True):
            steps = [dim_in//(2**idx) for idx in range(depth+1)]
            steps = steps if encode else list(reversed(steps))
            prog = [(steps[idx-1], steps[idx]) for idx in range(1, len(steps))]
            return prog

        C_dims = get_progression(startC, depth, encode=encode)
        T_dims = get_progression(startT, depth, encode=encode)
        
        core_blocks, aux_blocks, fuse_blocks = [], [], []
        for (inC, outC), (inT, outT) in zip(C_dims, T_dims):
            core_blocks.append(ProjLayer(inC, outC, inT, outT, p_drop, self_normalizing=self_normalizing))            
            aux_blocks.append(ProjLayer(inC, outC, inT, outT, p_drop, self_normalizing=self_normalizing)) if dual_stream else nn.Identity()            
            fuse_blocks.append(StreamAttention(inC))
            # fuse_blocks.append(nn.Identity())
            
        return nn.ModuleList(core_blocks), nn.ModuleList(aux_blocks), nn.ModuleList(fuse_blocks)

    def encode(self, x_core, x_aux):
        core_skips, aux_skips = [], []
        for blk_core, blk_aux, blk_fuse in zip(self.enc_core, self.enc_aux, self.enc_fuse):   
            # x_core = blk_fuse(x_core, x_aux.detach(), x_aux.detach())
            core_skips.append(x_core); aux_skips.append(x_aux)
            # x_core = torch.cat([x_core, x_aux], dim=1)
            x_core = blk_core(x_core); x_aux = blk_aux(x_aux)
        x_core = self.bottleneck_fusion(x_core, x_aux, x_aux)            
        core_skips, aux_skips = reversed(core_skips), reversed(aux_skips)
        return x_core, x_aux, core_skips, aux_skips

    def decode(self, x_core, x_aux, core_skips, aux_skips):
        for blk_core, blk_aux, blk_fuse, s_core, s_aux in zip(
            self.dec_core, self.dec_aux, self.dec_fuse, core_skips, aux_skips
        ):
            # x_core = torch.cat([x_core, x_aux], dim=1)
            # x_core = blk_fuse(x_core, x_aux, x_aux)
            x_core = blk_core(x_core); x_aux = blk_aux(x_aux)
            x_core = blk_fuse(x_core, s_core, s_core)
            # x_core = (x_core + s_core) * self.scale
            # x_aux = (x_aux + s_aux) * self.scale
            
        return x_core, x_aux

    def forward(self, x):
        x_core, x_aux = x[:, self.core_idxs, :], x[:, self.aux_idxs, :]
        x_core, x_aux = self.proj_core(x_core), self.proj_aux(x_aux)
        x_core, x_aux, core_skips, aux_skips = self.encode(x_core, x_aux)
        # x_core, x_aux = self.decode(x_core, x_aux, core_skips, aux_skips)
        # x = self.fusion(x_core, x_aux, x_aux)
        # x = torch.cat([x_core, x_aux], dim=1)
        return self.head(x_core)

def attach_activation_stats(model):
    stats = {}

    def hook(name):
        def fn(module, inputs, output):
            with torch.no_grad():
                mu, sigma = output.mean().item(), output.std().item()
                stats[name] = (mu, sigma)
        return fn

    for name, module in model.named_modules():
        if isinstance(module, nn.SELU):
            module.register_forward_hook(hook(name))
    return stats

if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    from torchinfo import summary
    from torch.utils.data import DataLoader, TensorDataset
    from h5py import File
    n_classes = 6
    B, C, T = 64, 124, 32
    Hz, S = 62.5, 16
    Fch, Fsp, K, D, P, montage = 64, 16, 5, 3, 8, 'HGSN_128'
    cdx = 97
    lr, weight_decay = 0.0001, 0.0001
    device='cuda:0'
    warmup_epochs=10; profile_epochs=20         

    with File('data/SUDB/processed/S6/EEG.hdf5', 'r') as f:
        X = torch.from_numpy(f['EEG'][:]).to(device)
    with File('data/SUDB/processed/S6/category.hdf5', 'r') as f:
        y = torch.from_numpy(f['category'][:]).to(device)
    hC, hT = 128, 32

    X = (X - (X.mean(dim=(0, 2), keepdim=True)))/X.std(dim=(0, 2), keepdim=True)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)

    net = DualStreamFFSNN(n_classes, n_channels=C, n_timepoints=T, 
                          hidden_spatial_dim=hC, hidden_temporal_dim=hT, 
                          montage=montage, depth=3,
                          self_normalizing=True).to(device)    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    print(summary(net, input_size=(1, C, T))); exit()
    stats = attach_activation_stats(net)
    for _ in range(warmup_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataloader)
        print(f"Loss: {running_loss:.4f}")
        bads = {layer: (mu, std) for layer, (mu, std) in stats.items() \
                if (abs(mu) > 0.05) or (abs(std - 1.0) > 0.25)}
        print(", ".join([f"{layer}: ({mu:.4f}, {std:.4f})" for layer, (mu, std) in bads.items()]))

    # Profile a few iterations
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=False,
    # ) as prof:
    #     for _ in range(profile_epochs):
    #         with record_function("model_inference"):
    #             _ = net(X)

    # # Write summary to file
    # with open('profile.txt', "w") as f:
    #     f.write(prof.key_averages(group_by_input_shape=True).table(
    #         sort_by="cuda_time_total",
    #         row_limit=100
    #     ))
