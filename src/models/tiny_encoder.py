import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from einops.layers.torch import Rearrange
import os
from pywt import DiscreteContinuousWavelet
from scipy.fft import next_fast_len

from src.models.basemodel import TorchEncoder, TorchModel
OCCIPITAL_ELECTRODES = {'HGSN_128': [65, 66, 67, 68, 69, 70, 71,
                                     72, 73, 74, 75, 76, 77, 81,
                                     82, 83, 84, 88, 89, 90, 94],
                        'BIOSEMI_96': [i - 1 for i in [83, 82, 81, 80, 79, 
                                                       85, 86, 87, 88, 89, 
                                                       94, 93, 92, 90, 
                                                       95, 96, 91]]}
class ConvBlock(nn.Module):
    def __init__(self, n_channels, temporal_kernel_size, n_temporal_filters, padding=0, activation=nn.ReLU, dropout = nn.Dropout):
        super().__init__()
        padding = (0, padding)
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_temporal_filters//16, 
                      kernel_size=(1, temporal_kernel_size), dilation = 1, padding='same', bias=False),
            activation(),
            nn.Conv2d(n_temporal_filters//16, n_temporal_filters//8, 
                      kernel_size=(1, temporal_kernel_size), dilation=1, padding='same', bias=False),
            activation(),
            dropout(0.1),
            nn.Conv2d(n_temporal_filters//8, n_temporal_filters//8, 
                      kernel_size=(n_channels, 1), padding='valid', #groups=n_temporal_filters//8,
                      bias=False),
            activation(),                        
            nn.Conv2d(n_temporal_filters//8, n_temporal_filters//4, 
                      kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),
            activation(),
            nn.Conv2d(n_temporal_filters//4, n_temporal_filters//2, 
                      kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),
            activation(),
            nn.Conv2d(n_temporal_filters//2, n_temporal_filters, 
                      kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),
            # activation(),      
            # nn.Conv2d(n_temporal_filters, n_temporal_filters, 
            #           kernel_size=1, padding='valid', bias=False),
            # activation(),       
        )

    def forward(self, x):
        return self.conv(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_heads, mask=None):
        super().__init__() 
        
        self.keys = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.queries = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.values = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)

        self.split_heads = Rearrange('b (h d) t -> b h t d', h=n_heads)
        self.merge_heads = Rearrange('b h t d -> b (h d) t')

        self.scale = (n_channels//n_heads) ** 0.5
        self.softmax = nn.Softmax(dim=-1)                

    def forward(self, x):
            # x = x.transpose(1,2)
        k = self.split_heads(self.keys(x))
        q = self.split_heads(self.queries(x))
        v = self.split_heads(self.values(x))
        
        energy = torch.einsum('bhtd,bhsd->bhts', q, k) 
        attn   = self.softmax(energy / self.scale)

        out = torch.einsum('bhts,bhsd->bhtd', attn, v)
        out = self.proj(self.merge_heads(out)).contiguous()#.transpose(1,2)
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, n_channels, feedforward_dim, activation, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, feedforward_dim, 
                      kernel_size=1, padding='valid', bias=False),
            activation(),
            dropout(0.1),
            nn.Conv1d(feedforward_dim, n_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class TransformerBlock(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_heads, feedforward_dim, dropout, activation, mask=None):
        super().__init__()
        self.mha = MultiHeadAttention(n_channels, n_timepoints, n_heads)
        self.ffn = FeedForwardNetwork(n_channels, feedforward_dim, activation=activation, dropout=dropout)
        
        self.scale = 2**-0.5

    def forward(self, x):
        x = (x + self.mha(x)) * self.scale
        x = (x + self.ffn(x)) * self.scale
        return x

class Transformer(nn.Module):
    def __init__(self, n_layers, n_channels, n_timepoints, n_heads, feedforward_dim, dropout, activation, mask):
        super().__init__()
        # self.class_token = nn.Parameter(torch.zeros(1, n_channels, 1)); n_timepoints+=1
        self.layers = nn.ModuleList([TransformerBlock(n_channels, n_timepoints, n_heads, feedforward_dim, dropout, activation, mask=mask) for _ in range(n_layers)])

    def forward(self, x):
        # b, _, t = x.shape

        # token = self.class_token.expand(b, -1, -1)
        # x = torch.cat([token, x], dim=2)

        for layer in self.layers:
            x = layer(x)
        return x#[:, :, 0]



class TinyDecoder(TorchModel):
    def __init__(self, n_classes, n_channels, n_timepoints, n_filters=512, temporal_kernel_size = 5, padding=1, self_normalizing=True, montage='HGSN_128', device = 'cpu'):
        super().__init__(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints, n_filters=n_filters, temporal_kernel_size=temporal_kernel_size, padding=padding, self_normalizing=self_normalizing, montage=montage, device=device)

    class ConvBlock(nn.Module):
        def __init__(self, n_channels, temporal_kernel_size, n_filters, padding=0, activation=nn.ReLU, dropout = nn.Dropout):
            super().__init__()
            padding = (0, padding)
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, n_filters//16, 
                        kernel_size=(1, temporal_kernel_size), dilation = 1, padding='same', bias=False),
                activation(),
                nn.Conv2d(n_filters//16, n_filters//8, 
                        kernel_size=(1, temporal_kernel_size), dilation=1, padding='same', bias=False),
                activation(),
                dropout(0.05),
                nn.Conv2d(n_filters//8, n_filters//8, 
                        kernel_size=(n_channels, 1), padding='valid', #groups=n_temporal_filters//8,
                        bias=False),
                activation(),                        
                nn.Conv2d(n_filters//8, n_filters//4, 
                        kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),
                activation(),
                nn.Conv2d(n_filters//4, n_filters//2, 
                        kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),
                activation(),
                nn.Conv2d(n_filters//2, n_filters, 
                        kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, bias=False),     
        )
            
        def forward(self, x):
            return self.conv_block(x)
        
    def set_model_parameters(self, n_classes, n_channels, n_timepoints, n_filters, temporal_kernel_size, padding, montage, self_normalizing):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        activation = nn.SELU if self_normalizing else nn.ReLU
        conv_dropout = nn.FeatureAlphaDropout if self_normalizing else nn.Dropout
        dense_dropout = nn.AlphaDropout if self_normalizing else nn.Dropout

        # self.register_buffer('mask', torch.zeros(1, self.n_channels, 1).bool())
        # self.occipital_electrodes = OCCIPITAL_ELECTRODES[montage]
        # self.mask[:, self.occipital_electrodes, :] = True
        
        self.conv_block = self.ConvBlock(n_channels, temporal_kernel_size, n_filters, padding, activation, conv_dropout)
        # self.masked_conv_block = self.ConvBlock(n_channels, temporal_kernel_size, n_filters, padding, activation, conv_dropout)

        sequence_length = n_timepoints - ((temporal_kernel_size*2) - padding*2)*3

        self.classifier_block = nn.Sequential(
            activation(),
            dense_dropout(0.1),
            nn.Flatten(),            
            nn.Linear(sequence_length*n_filters, self.n_classes),
        )

        def lecun_init(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        if self_normalizing:
            self.apply(lecun_init)
        
    def forward(self, x, softmax=False):

        x = x.unsqueeze(1)
        # x_masked = x.masked_fill(self.mask, 0)

        x = self.conv_block(x)        
        # x_masked = self.masked_conv_block(x_masked)
        
        # x = torch.cat((x, x_masked), 1)
        # x = (x + x_masked) * (2**-0.5)

        x = self.classifier_block(x)
        if softmax:
            x = nn.functional.softmax(x, dim=1)
        return x
    
class TinyEncoder(TorchEncoder):
    def __init__(self, n_channels, n_timepoints,  emb_dim, temporal_kernel_size=9, padding=2, n_filters=512, n_layers = 1, n_heads = 8, feedforward_dim = None, self_normalizing=False, device='cpu'):
        super().__init__(n_channels=n_channels, n_timepoints=n_timepoints, emb_dim=emb_dim, temporal_kernel_size=temporal_kernel_size, padding=padding, n_filters=n_filters, n_layers=n_layers, n_heads=n_heads, feedforward_dim=feedforward_dim, self_normalizing=self_normalizing, device=device)
    
    def set_model_parameters(self, n_channels, n_timepoints, emb_dim, n_filters, temporal_kernel_size, padding, n_layers, n_heads, feedforward_dim, self_normalizing=False):
        
        activation = nn.SELU if self_normalizing else nn.ReLU
        conv_dropout = nn.FeatureAlphaDropout if self_normalizing else nn.Dropout
        dense_dropout = nn.AlphaDropout if self_normalizing else nn.Dropout
        feedforward_dim = feedforward_dim or 2 * n_filters

        self.conv = ConvBlock(n_channels, temporal_kernel_size, n_filters, padding=padding, activation=activation, dropout=conv_dropout)

        sequence_length = n_timepoints - ((temporal_kernel_size - 1) - padding*2)*3
        # mask = get_selt_attention_mask(sequence_length, kind='local_band', bandwidth=3)
        mask = None
        self.transformer = Transformer(n_layers, n_filters, sequence_length, n_heads, feedforward_dim, dropout=dense_dropout, activation=activation, mask=mask)

        self.proj = nn.Sequential(
            nn.Flatten(),
            activation(),
            dense_dropout(0.1),
            # nn.Linear(n_filters, emb_dim),
            nn.Linear(n_filters*sequence_length, emb_dim),
            # nn.Linear(n_filters*(sequence_length+1), emb_dim),
        )

        def lecun_init(layer):
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Parameter)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x, normalize_embedding=True):
        x = self.conv(x.unsqueeze(1)).squeeze(2)

        x = self.transformer(x)
        h = self.proj(x)
        if normalize_embedding:
            return F.normalize(h, p=2, dim=1)
        else:
            return h

class TinyTransformer(TorchEncoder):
    def __init__(self, n_channels, n_timepoints,  emb_dim, n_layers = 1, n_heads = 8, feedforward_dim = 2048, self_normalizing=False, device='cpu'):
        super().__init__(n_channels=n_channels, n_timepoints=n_timepoints, emb_dim=emb_dim, n_layers=n_layers, n_heads=n_heads, feedforward_dim=feedforward_dim, self_normalizing=self_normalizing, device=device)
        
    class TinyTransformerBlock(nn.Module):
        class MultiHeadAttention(nn.Module):
            def __init__(self, n_channels, n_timepoints, n_heads):
                super().__init__()                 
                self.keys = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
                self.queries = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
                self.values = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
                self.proj = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)

                # self.keys = nn.Linear(n_channels, n_channels, bias=False)
                # self.queries = nn.Linear(n_channels, n_channels, bias=False)
                # self.values = nn.Linear(n_channels, n_channels, bias=False)
                # self.proj = nn.Linear(n_channels, n_channels, bias=False)

                self.split_heads = Rearrange('b (h d) t -> b h t d', h=n_heads)
                self.merge_heads = Rearrange('b h t d -> b (h d) t')

                self.scale = (n_channels//n_heads) ** 0.5
                self.softmax = nn.Softmax(dim=-1)                

            def forward(self, x):
                # x = x.transpose(1,2)
                k = self.split_heads(self.keys(x))
                q = self.split_heads(self.queries(x))
                v = self.split_heads(self.values(x))
                
                energy = torch.einsum('bhtd,bhsd->bhts', q, k) 
                attn   = self.softmax(energy / self.scale)

                out = torch.einsum('bhts,bhsd->bhtd', attn, v)
                out = self.proj(self.merge_heads(out)).contiguous()#.transpose(1,2)
                return out

        class FeedForwardNetwork(nn.Module):
            def __init__(self, n_channels, feedforward_dim, activation, dropout):
                super().__init__()
                self.ffn = nn.Sequential(
                    nn.Conv1d(n_channels, feedforward_dim, kernel_size=3, padding='same', bias=False),
                    activation(),
                    dropout(0.1),
                    nn.Conv1d(feedforward_dim, n_channels, kernel_size=1, bias=False),
                )

            def forward(self, x):
                x = x
                x = self.ffn(x)
                return x
        
        def __init__(self, n_channels, n_timepoints, n_heads, feedforward_dim, dropout, activation):
            super().__init__()
            self.mha = self.MultiHeadAttention(n_channels, n_timepoints, n_heads)            
            self.ffn = self.FeedForwardNetwork(n_channels, feedforward_dim, activation=activation, dropout=dropout)
        
            self.scale = 2**-0.5

        def forward(self, x):
            x = (x + self.mha(x)) * self.scale
            x = (x + self.ffn(x)) * self.scale
            return x
        
    def set_model_parameters(self, n_channels, n_timepoints, emb_dim, n_layers, n_heads, feedforward_dim, self_normalizing=False):
        
        activation = nn.SELU if self_normalizing else nn.ReLU
        dropout = nn.FeatureAlphaDropout if self_normalizing else nn.Dropout
        dense_dropout = nn.AlphaDropout

        self.layers = nn.ModuleList([self.TinyTransformerBlock(n_channels, n_timepoints, n_heads, feedforward_dim, dropout=dense_dropout, activation=activation) for _ in range(n_layers)])
        
        self.proj = nn.Sequential(
            nn.Flatten(),
            activation(),
            dense_dropout(0.1),
            nn.Linear(n_channels*n_timepoints, emb_dim),
        )

        def lecun_init(layer):
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Parameter)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x, normalize_embedding=True):
        for layer in self.layers:
            x = layer(x)
        h = self.proj(x)
        if normalize_embedding:
            return F.normalize(h, p=2, dim=1)
        else:
            return h

class SNCNDN(TorchModel):
    def __init__(self, n_classes, n_channels, n_timepoints, n_filters=512, temporal_kernel_size = 5, padding=1, self_normalizing=False, device = 'cpu'):
        super().__init__(emb_dim=n_classes, n_channels=n_channels, n_timepoints=n_timepoints, n_filters=n_filters, temporal_kernel_size=temporal_kernel_size, padding=padding, self_normalizing=self_normalizing, device=device)

    class ConvBlock(nn.Module):
        def __init__(self, n_channels, temporal_kernel_size, n_filters, padding=0, self_normalizing=False):
            super().__init__()
            padding = (0, padding)
            self.conv_block = nn.Sequential(
                nn.Conv2d(1, n_filters//32, kernel_size=(1, temporal_kernel_size), dilation=1, padding='same', bias=False),
                nn.SELU() if self_normalizing else (nn.BatchNorm2d(n_filters//32)),
                # nn.Conv2d(n_filters//32, n_filters//16, kernel_size=(1, temporal_kernel_size), padding='same', bias=False),
                # nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//16), 
                # nn.Conv2d(n_filters//16, n_filters//8, kernel_size=(1, temporal_kernel_size), padding='same', bias=False),
                # nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//8),    

                nn.Conv2d(n_filters//32, n_filters//8, kernel_size=(n_channels, 1), padding='valid',bias=False, groups=n_filters//32), 
                nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//8), 
                nn.AlphaDropout(0.1)  if self_normalizing else nn.Dropout(0.5),                
                
                nn.Conv2d(n_filters//8, n_filters//4, kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, groups=n_filters//8,bias=False),
                nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//4), 
                nn.Conv2d(n_filters//4, n_filters//2, kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, groups=n_filters//4, bias=False),
                nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//2), 
                nn.Conv2d(n_filters//2, n_filters, kernel_size=(1, temporal_kernel_size*2 + 1), padding=padding, groups = n_filters//2, bias=False), 
                nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters),    
        )
            
        def forward(self, x):
            return self.conv_block(x)
        
    def set_model_parameters(self, emb_dim, n_channels, n_timepoints, n_filters, temporal_kernel_size, padding, self_normalizing=False):
        
        self.register_buffer('occipital_electrodes', torch.tensor(OCCIPITAL_ELECTRODES['HGSN_128']).long())
                
        self.conv_block = self.ConvBlock(n_channels, temporal_kernel_size, n_filters, padding, self_normalizing)
        self.aux_conv_block = self.ConvBlock(self.occipital_electrodes.size(0), temporal_kernel_size, n_filters, padding, self_normalizing)
        sequence_length = n_timepoints - ((temporal_kernel_size*2) - padding*2)*3

        self.proj = nn.Sequential(            
            nn.Flatten(),         
            nn.AlphaDropout(0.1) if self_normalizing else nn.Dropout(0.5),  
            nn.Linear(sequence_length*n_filters, emb_dim),
        )

        def lecun_init(layer):            
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        if self_normalizing:
            print("Using LeCun normal initialization to facilitate self-normalization.")
            self.apply(lecun_init)
        
    def forward(self, x, softmax=False):
        x = x.unsqueeze(1)
        x_occipital = x[:, :, self.occipital_electrodes].contiguous()

        x = self.conv_block(x)
        x_occipital = self.aux_conv_block(x_occipital)
        x = (x + x_occipital) * (2**-0.5)
        
        x = self.proj(x)

        if softmax:
            return F.softmax(x, dim=-1)
        else:
            return x

def bands_to_scales(bands, sampling_freq, central_freq, freqs_per_band = 5):
    import numpy as np
    scales = torch.cat([
        torch.logspace(np.log2(low_f), np.log2(high_f), freqs_per_band, base=2) for (low_f, high_f) in bands.values()
    ])
    return torch.flip(sampling_freq * central_freq / scales, dims=(0, ))

class CWT(nn.Module):
    def __init__(self, n_timepoints, sampling_freq, wavelet='cmor', central_freq=1.0, bandwidth=1.0, bands={'full_spectrum': (5, 25)}, freqs_per_band=32, device='cpu'):
        super().__init__()
        wavelet = f'{wavelet}{bandwidth}-{central_freq}'
        self.sampling_freq = sampling_freq
        self.freqs_per_band = freqs_per_band
        self.device = device

        kernel_path = os.path.join("data", 'misc', "cwt_kernels", f"Full@{sampling_freq}Hz", f'{wavelet}.pt')
        if os.path.exists(kernel_path):
            kernels, scales = torch.load(kernel_path, map_location=device).values()
            self.N = kernels.shape[-1]
        else:
            psi, x, = DiscreteContinuousWavelet(wavelet).wavefun(10)
            int_psi = torch.conj(torch.cumsum(torch.tensor(psi, dtype=torch.complex64, device=device), dim=0) * (x[1] - x[0]))

            scales = bands_to_scales(bands, sampling_freq, central_freq, freqs_per_band).to(device)
            scales_idxs = [(torch.arange(scale * (x[-1] - x[0]) + 1, device=device)/ (scale * (x[1] - x[0]))).long() for scale in scales]

            kernels = [torch.flip(int_psi[s[s<len(int_psi)]], dims=(0,)) for s in scales_idxs]
            kernel_lengths = torch.tensor([k.numel() for k in kernels], dtype=torch.long, device=device)
            
            self.N = next_fast_len(64 + kernel_lengths.max() - 1)
            kernels = torch.fft.fft(torch.stack([F.pad(k, (0, kernel_lengths.max() - k.numel())) for k in kernels], dim=0), n=self.N)

            tdx = (kernel_lengths// 2 - 1)[:, None] + (torch.arange(n_timepoints+1, device=device)[None, :])
            tdx = torch.einsum('n, lt -> ltn', torch.arange(self.N, device=device), tdx)
            
            kernels = kernels[:, None, :] * (torch.exp(2j * torch.pi * tdx / self.N))
            kernels = kernels.to(device)
            os.makedirs(os.path.dirname(kernel_path), exist_ok=True)
            # torch.save({'kernels': kernels, 'scales': scales}, kernel_path)
        self.register_buffer('kernels', kernels)
        self.register_buffer('scales', scales)

    def forward(self, x):
        x = torch.fft.fft(x, n=self.N).to(torch.complex64)
        out = torch.einsum('bct,skt->bcsk', x, self.kernels) / self.N
        out = torch.sqrt(self.scales[None, None, :, None]) * torch.diff(out, dim=-1)
        return out.abs()

class SpectralDecoder(TorchModel):
    def __init__(self, n_classes, n_channels, n_timepoints, sampling_freq, bands={'full_spectrum': (5, 25)}, freqs_per_band=32, n_filters=512, temporal_kernel_size = 5, freq_kernel_size=5, padding=1, self_normalizing=True, device = 'cpu'):
            super().__init__(n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints, sampling_freq=sampling_freq, bands=bands, freqs_per_band=freqs_per_band, n_filters=n_filters, temporal_kernel_size=temporal_kernel_size, freq_kernel_size=freq_kernel_size, padding=padding, self_normalizing=self_normalizing, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_timepoints, sampling_freq, bands, freqs_per_band, n_filters, temporal_kernel_size, freq_kernel_size, padding, self_normalizing):
        self.cwt = CWT(wavelet='cmor', central_freq=1.0, bandwidth=1.0, sampling_freq=sampling_freq, n_timepoints=n_timepoints, 
                       bands=bands, freqs_per_band=freqs_per_band, device=self.device)
        
        seq_len = n_timepoints - (temporal_kernel_size - padding)*6
        freq_len = freqs_per_band - (freq_kernel_size - padding)*6
        proj_in = n_filters*seq_len*freq_len

        padding = (padding, padding)
        self.register_buffer('occipital_electrodes', torch.zeros(self.n_channels).long())
        self.occipital_electrodes[:] = OCCIPITAL_ELECTRODES['HGSN_128']
        
        self.filter_bank = nn.Sequential(
            nn.Conv3d(1, n_filters//16, kernel_size=(1, freq_kernel_size, temporal_kernel_size), padding='same', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//16),  
            nn.Conv3d(n_filters//16, n_filters//8, kernel_size=(1, freq_kernel_size, temporal_kernel_size), padding='same', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//8),                        
            nn.Conv3d(n_filters//8, n_filters//8, kernel_size=(n_channels, 1, 1), padding='valid', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//8),
            nn.AlphaDropout(0.1) if self_normalizing else nn.Dropout(0.5),
        )
        self.occipital_filter_bank = nn.Sequential(
            nn.Conv3d(1, n_filters//16, kernel_size=(1, freq_kernel_size, temporal_kernel_size), padding='same', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//16),  
            nn.Conv3d(n_filters//16, n_filters//8, kernel_size=(1, freq_kernel_size, temporal_kernel_size), padding='same', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//8),                        
            nn.Conv3d(n_filters//8, n_filters//8, kernel_size=(self.occipital_electrodes.size(), 1, 1), padding='valid', bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm3d(n_filters//8),
            nn.AlphaDropout(0.1) if self_normalizing else nn.Dropout(0.5),
        )
        self.spectro_temporal_conv = nn.Sequential(
            nn.Conv2d(n_filters//8, n_filters//4, kernel_size=(freq_kernel_size*2+1, temporal_kernel_size*2+1), padding=padding, bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//4),
            nn.Conv2d(n_filters//4, n_filters//2, kernel_size=(freq_kernel_size*2+1, temporal_kernel_size*2+1), padding=padding, bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters//2),
            nn.Conv2d(n_filters//2, n_filters, kernel_size=(freq_kernel_size*2+1, temporal_kernel_size*2+1), padding=padding, bias=False),
            nn.SELU() if self_normalizing else nn.BatchNorm2d(n_filters),
        )        

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.SELU() if self_normalizing else nn.ReLU(),            
            nn.Linear(proj_in, n_classes),
        )

        def lecun_init(layer):
            if isinstance(layer, (nn.Conv3d, nn.Linear, nn.Parameter)):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        
        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x, softmax=False):
        x = self.cwt(x).unsqueeze(1)        
        x = self.filter_bank(x).squeeze(2)
        x = self.spectro_temporal_conv(x)
        x = self.proj(x)

        if softmax:
            F.softmax(x, dim=-1)
        return x                             

def get_selt_attention_mask(sequence_length, kind='local_band', bandwidth=1):
    if kind == 'causal':
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
    else:
        i = torch.arange(sequence_length)[:, None]
        j = torch.arange(sequence_length)[None, :]
        mask = (j - i).abs() > bandwidth
    return mask

def get_embedding_metrics(h_train, y_train, z_train, h_test, y_test, z_test):
    d = 1 - (h_test @ h_train.t())
    intra_class = y_test[:, None] == y_train[None, :]
    inter_class = ~intra_class
    intra_stimulus = z_test[:, None] == z_train[None, :]
    intra_class_inter_stimulus = intra_class & ~intra_stimulus

    return d[inter_class].mean().item(), d[intra_class].mean().item(), d[intra_class_inter_stimulus].mean().item(),  d[intra_stimulus].mean().item()

def get_rdm(h_train, y_train, h_test, y_test):
    d  = 1 - h_test @ h_train.t()   
    classes = y_train.unique(sorted=True)                   
    
    T  = (y_test[:, None] == classes[None, :]).float()    
    S  = (y_train[:, None] == classes[None, :]).float()   
    R  = (T.T @ d @ S) / (T.sum(0).unsqueeze(1) * S.sum(0).unsqueeze(0))
    R  = R.masked_fill(classes[:, None].eq(classes[None, :]), 0)
    return R

if __name__ == "__main__":
    from tqdm.auto import tqdm
    from sklearn.manifold import TSNE, MDS
    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns
    from src.experiments.experiment_loader import load_experiment_cfg
    from src.experiments.utils import get_data_manager, apply_transforms

    def snn_quickcheck(model):
        stats, hooks = {}, []
        def make_hook(name):
            def hook(_, __, out):
                t = out.detach().float()
                stats[name] = (t.mean().item(), t.std(unbiased=False).item())
            return hook

        for name, m in model.named_modules():
            if isinstance(m, nn.SELU):
                hooks.append(m.register_forward_hook(make_hook(name)))

        return stats
    
    def get_data_loaders(exp_id, model_id, subject, fold_idx, nested_fold_idx, trainer, device='cpu'):
        cfg = load_experiment_cfg(exp_id)
        model_cfg = cfg['models'][model_id]
        data_manager = get_data_manager(cfg, model_id, subject, fold_idx, nested_fold_idx, slice(None), slice(None), slice(None), device=device)
        transforms = model_cfg['transforms'][0]
        t = {k: v['method'](**v.get('kwargs', {})) for k, v in  transforms.items()}

        train_data = data_manager.get_data_partition('train')
        train_data = apply_transforms(train_data, t, fit=True)
        
        test_partitions = cfg['scheme']['test_partitions'] if nested_fold_idx is None else cfg['scheme']['val_partitions']
        test_data = {partition:
                        apply_transforms(
                            data_manager.get_data_partition(partition),
                        t, fit=False)
                    for partition in test_partitions}
        
        loaders = {partition: trainer.build_dataloader(data, shuffle=False) for partition, data in test_data.items()}
        loaders['train'] = trainer.build_dataloader(train_data, shuffle=True)

        return loaders

    def silhouette(rdm, y):
        # Get average intra class dist for a point i
        classes, inv = torch.unique(y, return_inverse=True)
        n_classes = classes.numel()
        n_samples, _ = rdm.shape

        K = torch.zeros((n_samples, n_classes))
        K[torch.arange(n_samples), inv]=1

        cluster_sizes = K.sum(dim=0)
        sample_cluster_size = cluster_sizes[inv]

        K_d = rdm @ K

        a = K_d[torch.arange(n_samples), inv]/(sample_cluster_size - 1).clip(1, None)

        K_d = K_d / cluster_sizes.clip(1, None)
        K_d[torch.arange(n_samples), inv] = float("Inf")
        b, _ = torch.min(K_d, dim=1)

        score = (b - a)/torch.max(a, b)
        return score

    LABELS = {'category': {0: 'HB', 1: 'HF', 2: 'AB', 3: 'AF', 4: 'FV', 5: 'IO'},
          'human_face_vs_artificial_object': {0: 'HF', 1: 'IO'},
          'exemplar': {idx: f'figures/stimuli/stimulus{idx+1}.png' for idx in range(72)},
          'artificial_object': {idx: f'figures/stimuli/stimulus{idx+61}.png' for idx in range(12)},
          'human_face': {idx: f'figures/stimuli/stimulus{idx+13}.png' for idx in range(12)}}
    COLORS = {'category': {0: '#207aba', 1: '#ff7923', 2: '#38a446', 3: '#ea2f1c', 4: '#8566b4', 5: '#bcbe3c'}, 
          'human_face_vs_artificial_object': {0: '#ff7923', 1: '#bcbe3c'}, 
          'exemplar': {idx: ['#207aba','#ff7923','#38a446','#ea2f1c','#8566b4','#bcbe3c'][idx//12] for idx in range(72)},
          'artificial_object': {idx: '#bcbe3c' for idx in range(12)},
          'human_face': {idx: '#ff7923' for idx in range(12)}}

    exp_id = "SUDB_full_response_category_decoding"
    model_id = "TinyDecoder"
    n_classes = 6
    n_channels, n_timepoints = 124, 32
    sampling_freq = 62.5
    emb_dim, n_filters = 6, 4096
    temporal_kernel_size, freq_kernel_size = 5, 5
    padding=1
    freqs_per_band=16
    n_layers = 1
    feedforward_dim = 2*n_filters #16384
    n_heads = 8     
    self_normalizing=False
    lr = 1e-4
    weight_decay = 1e-8
    max_epochs = 50
    batch_size = 64
    margin=0.2
    subject = 'S6'
    fold_idx, nested_fold_idx = 5, None
    device='cuda:0'
    plot_every=5
    tol = 0.15
    
    optimizer = torch.optim.AdamW
    criterion = nn.CrossEntropyLoss
    # criterion = nn.TripletMarginWithDistanceLoss

    fig, ax = plt.subplots(4, max_epochs//plot_every, figsize=(20, 16))
    with torch.device(device):
        enc_net = SNCNDN(
            n_classes = n_classes, 
            n_channels=n_channels,
            n_timepoints=n_timepoints,            
            temporal_kernel_size=temporal_kernel_size,
            padding=padding,
            n_filters=n_filters, 
            self_normalizing=self_normalizing,
            device=device)
        # enc_net = SpectralDecoder(
        #     n_classes = n_classes, 
        #     n_channels=n_channels,
        #     n_timepoints=n_timepoints,
        #     sampling_freq=sampling_freq,
        #     temporal_kernel_size=temporal_kernel_size,
        #     freq_kernel_size=freq_kernel_size,
        #     freqs_per_band=freqs_per_band,
        #     padding=padding,
        #     n_filters=n_filters, 
        #     self_normalizing=self_normalizing,
        #     device=device)

        enc_trainer = enc_net.get_trainer(
                        optimizer=optimizer, 
                        criterion=criterion, 
                        lr=lr, 
                        weight_decay=weight_decay, 
                        batch_size=batch_size, 
                        max_epochs=max_epochs,
                        # margin=margin
                        )

        loaders = get_data_loaders(exp_id, model_id, subject, fold_idx, nested_fold_idx, enc_trainer, device=device)
        stats = snn_quickcheck(enc_net) 

        for epoch in tqdm(range(1, max_epochs+1), desc='Training encoder', total=max_epochs):
            
            loss = enc_trainer.train(loaders['train'])
            bad = {n:(mu, sd) for n,(mu,sd) in stats.items() if abs(mu)>0.05 or abs(sd-1)>0.2}
            if ((epoch-1) % plot_every == 0) and self_normalizing:
                print(f"Self-normalizing failures: ", bad.items())
            for idx, (partition, loader) in enumerate(loaders.items()):
                if epoch % plot_every == 0:
                    enc_net.eval()
                    with torch.inference_mode():
                        # metrics = enc_trainer.predict_proba(loader)
                        y_true, y_pred, loss = enc_trainer.predict_proba(loader)
                        acc = (y_true == y_pred.argmax(axis=-1)).mean().item()
                        print(f"Epoch: {epoch}, {partition}: accuracy: {acc*100:.2f}%, loss: {loss}")
                        # print(f"kNN. Epoch: {epoch}, {partition}: accuracy: {metrics['accuracy']*100:.2f}%, accuracy@{metrics['opt_k']}: {metrics['opt_acc']*100:.2f}, TSR: {metrics['TSR']:.4f}, nDCG@10: {metrics['nDCG@10']:.4f}, mixing-ratio: {metrics['mixing-ratio']:.4f}.")
                       
                        