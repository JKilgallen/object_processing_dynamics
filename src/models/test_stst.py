import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, cwt
import numpy as np
from matplotlib import pyplot as plt
from time import time
from scipy.fft import next_fast_len
from src.models.basemodel import TorchModel

BANDS = {
    "delta": (2, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 50)
}

def bands_to_scales(bands, sampling_freq, central_freq, freq_resolution = 5):
    scales = torch.cat([
        torch.linspace(low_f, high_f, freq_resolution+1)[:freq_resolution] for (low_f, high_f) in bands.values()
    ])
    return torch.flip(sampling_freq * central_freq / scales, dims=(0, ))

class CWT(nn.Module):
    def __init__(self, wavelet, central_freq, bandwidth, bands, sampling_freq, n_time_points, freq_resolution, device='cpu'):
        
        super().__init__()

        wavelet = f'{wavelet}{bandwidth}-{central_freq}'
        psi, x, = DiscreteContinuousWavelet(wavelet).wavefun(10)
        int_psi = torch.conj(torch.cumsum(torch.tensor(psi, dtype=torch.complex64).to(device), dim=0) * (x[1] - x[0]))

        self.scales = bands_to_scales(bands, sampling_freq, central_freq, freq_resolution).to(device)
        self.n_scales = len(self.scales)
        scales_idxs = [(torch.arange(scale * (x[-1] - x[0]) + 1, device=device)/ (scale * (x[1] - x[0]))).long() for scale in self.scales]
        self.kernels = [torch.flip(int_psi[s[s<len(int_psi)]], dims=(0,)) for s in scales_idxs]


        self.kernel_lengths = torch.tensor([k.numel() for k in self.kernels], dtype=torch.long, device=device)
        self.N = next_fast_len(64 + self.kernel_lengths.max() - 1)
        T = torch.arange(self.N, device=device)
        tdx   = ((self.kernel_lengths -1)// 2)[:, None] + (torch.arange(n_time_points+1, device=device)[None, :])
        # self.N = torch.max(self.kernel_lengths) + 64
        self.kernels = torch.fft.fft(torch.stack([F.pad(k, (0, self.kernel_lengths.max() - k.numel())) for k in self.kernels], dim=0), n=self.N)
        self.K = self.kernels[:, None, :] * (torch.exp(2j * np.pi * T[None, None, :] * tdx[:, :, None] / self.N))
        torch.save(self.K, f"data/CVPR2021_REPROCESSED/processed/CWT_{wavelet}_{sampling_freq}Hz.pt")

    
    def forward(self, x):
        b, c, t = x.shape
        
        x = torch.fft.fft(x, n=self.N).to(torch.complex64)  # (B, C, N)

        print(f"K shape: {self.K.shape}, x shape: {x.shape}")
        out = torch.einsum('bct,skt->bcsk', x, self.K) / self.N
        out = torch.sqrt(self.scales[None, None, :, None]) * torch.diff(out, dim=-1).abs()

        return out

if __name__ == "__main__":
    print(bands_to_scales(BANDS, 128, 1.0, 5))
    x = torch.load("data/CVPR2021_REPROCESSED/processed/EEG_0.5-128Hz.pt")
    import torch.profiler as prof
    spectral_transform = CWT("cmor", 1.0, 1.0, BANDS, 128, 64, 5, device='cuda')
    with prof.profile(
    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
    ) as prof:
        with torch.no_grad():
            
            x_spectral = spectral_transform(x[:32, :, :].to('cuda'))
            print(x_spectral)

    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", row_limit=50))
    # spectral_transform = CWT("cmor", 1.0, 1.0, BANDS, 128, 5)
    # # get summary of spectral_transform
    # start = time()
    # x_spectral = spectral_transform(x[:2, :, :])
    # print(f"Time taken: {time() - start:.2f} seconds")
    # x_spectral = x_spectral.cpu().detach()
    # x_spectral_2, _ = cwt(x[3, 86, :].numpy(), scales=bands_to_scales(BANDS, 128, 1.0, 5).numpy(), wavelet='cmor1.0-1.0', sampling_period=1/128, method='fft')
    # x_spectral_2 = abs(x_spectral_2)
    # print(x_spectral.shape, x_spectral_2.shape)
    # fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    # vmin = min(np.percentile(x_spectral[3, 86, :, :].numpy(), 1), np.percentile(x_spectral_2[:, :], 1))
    # vmax = max(np.percentile(x_spectral[3, 86, :, :].detach().numpy(), 99), np.percentile(x_spectral_2[:, :], 99))
    # ax[0].pcolormesh(x_spectral[3, 86, :, :].detach().numpy(), vmin=vmin, vmax=vmax)
    # ax[1].pcolormesh(x_spectral_2[:, :], vmin=vmin, vmax=vmax)
    # ax[2].pcolormesh(x_spectral[3, 86, :, :].detach().numpy() - abs(x_spectral_2[:, :]), vmin=vmin, vmax=vmax)
    # plt.show()