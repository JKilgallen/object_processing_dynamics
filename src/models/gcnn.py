import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.data import Data
import torch_geometric.transforms as T
from pywt import DiscreteContinuousWavelet
from scipy.fft import next_fast_len
import numpy as np
import os
from mne.channels import make_standard_montage

def build_adj(coords, aux_idx=None, k=8, aux_boost=2.0):
    x = coords.float()
    N, D = x.shape
    if D == 3:
        u = x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        Dm = torch.cdist(u, u)
        ang = ((u @ u.t()).clamp(-1, 1) + 1) * 0.5
    else:
        Dm = torch.cdist(x, x); ang = 1.0
    knn = Dm.topk(k=k+1, largest=False).indices[:, 1:]
    sig = Dm.gather(1, knn).median().clamp_min(1e-6)
    W = torch.zeros_like(Dm)
    rbf = torch.exp(-(Dm / sig) ** 2)
    rows = torch.arange(N).unsqueeze(1).expand_as(knn)
    W[rows, knn] = rbf[rows, knn]
    W = torch.maximum(W, W.t()) * ang
    if aux_idx is not None and len(aux_idx):
        a = aux_idx.long()
        W[:, a] *= aux_boost; W[a, :] *= aux_boost
    W.fill_diagonal_(1.0)
    deg = W.sum(-1).clamp_min(1e-8); d = deg.pow(-0.5)
    return d.unsqueeze(1) * W * d.unsqueeze(0)  # sym-normalized


class EEGSensorNet(nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints,
                 f1=256, f2=64, c1=256, c2=32):
        super().__init__()
        self.n_channels, self.n_timepoints = n_channels, n_timepoints

        # grow features (f1 -> f2), shrink nodes (N -> c1 -> c2)
        self.gcn_z1 = gnn.DenseGCNConv(n_timepoints, f1, improved=True)
        self.gcn_s1 = gnn.DenseGCNConv(n_timepoints, c1, improved=True)

        self.gcn_z2 = gnn.DenseGCNConv(f1, f2, improved=True)
        self.gcn_s2 = gnn.DenseGCNConv(f1, c2, improved=True)

        self.act = nn.ELU(inplace=True)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(c2 * f2, n_classes))

        self.make_graph()  # sets self.adj_norm

    def make_graph(self):
        montage = make_standard_montage("GSN-HydroCel-129")
        pos = torch.tensor([p for _, p in montage.get_positions()['ch_pos'].items()][:124], dtype=torch.float32)
        data = T.Compose([T.KNNGraph(k=15, loop=False)])(Data(pos=pos))
        N = int(data.edge_index.max()) + 1
        A = torch.zeros((N, N), dtype=torch.float32)
        src, dst = data.edge_index
        w = 1/arc_dist(pos)
        A[src, dst] = w[src, dst]
        aux = torch.tensor([65,66,67,68,69,70,71,72,73,74,75,76,77,81,82,83,84,88,89,90,94])
        A[aux, aux] = w[aux, aux]
        # A[aux, :] = w[aux, :]
        A.fill_diagonal_(0.0) 
        # deg = A.sum(-1).clamp_min(1.0); d = deg.pow(-0.5)
        self.register_buffer('adj_norm', A)

    def forward(self, x):  # x: [B,N,T] (already normalized upstream)
        A = self.adj_norm

        # level 1: N -> c1, features N_timepoints -> f1
        z1 = self.act(self.gcn_z1(x, A))                 # [B,N,f1]
        s1 = torch.softmax(self.gcn_s1(x, A), dim=-1)    # [B,N,c1]
        x1, A1, l1, e1 = gnn.dense_diff_pool(z1, A, s1)  # [B,c1,f1]

        # level 2: c1 -> c2, features f1 -> f2
        z2 = self.act(self.gcn_z2(x1, A1))               # [B,c1,f2]
        s2 = torch.softmax(self.gcn_s2(x1, A1), dim=-1)  # [B,c1,c2]
        x2, A2, l2, e2 = gnn.dense_diff_pool(z2, A1, s2) # [B,c2,f2]

        reg = 1e-2 * (l1 + e1 + l2 + e2)
        return self.head(x2), reg     

def bands_to_scales(bands, sampling_freq, central_freq, freqs_per_band = 5):
    freqs = torch.cat([
        torch.logspace(np.log2(low_f), np.log2(high_f), freqs_per_band, base=2) for (low_f, high_f) in bands.values()
    ])
    return torch.flip(sampling_freq * central_freq / freqs, dims=(0, )), freqs

class CWT(nn.Module):
    def __init__(self, n_timepoints, sampling_freq, wavelet='cmor', central_freq=1.0, bandwidth=0.5, bands={'full_spectrum': (2, 25)}, freqs_per_band=32, device='cpu'):
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

        return out.abs()

class SpatialMHA(torch.nn.Module):
    def __init__(self, n_channels, emb_dim, n_heads, spatial_bias=True):
        super().__init__()
        self.C = n_channels; self.H = n_heads; self.D = emb_dim//n_heads
        self.scale = (self.D)**-0.5
        
        self.q_proj = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.SELU()
        )
        self.k_proj = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.SELU()
        )
        self.v_proj = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.SELU()
        )

        self.proj_out = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 1, bias=False),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            # nn.Linear(emb_dim*2, emb_dim, bias=False),
            # nn.SELU(),
        )    
        
        pos = torch.load("electrode_positions.pt")
        dist = arc_dist(pos)
        sigma = torch.tril(dist, diagonal=1).median()
        attn_bias = -(dist**2)/(2*(sigma**2) + 1e-8)
        attn_bias = attn_bias - attn_bias.mean(dim=1, keepdim=True)
        attn_bias = attn_bias if spatial_bias else torch.zeros_like(attn_bias)
        self.register_buffer('attn_bias', attn_bias)

        disp = pos.unsqueeze(0) - pos.unsqueeze(1) 
        disp = disp / (disp.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        disp[torch.arange(self.C), torch.arange(self.C)] = 0
        self.register_buffer('displacement', disp if spatial_bias else torch.zeros_like(disp))
        self.dir_vec = nn.Parameter(torch.randn(n_heads, pos.size(0))) if spatial_bias else torch.zeros(n_heads, pos.size(0))

        k = min(21, self.C)
        knn_idx = torch.topk(dist, k=k, largest=False).indices      # [C,k]
        knn_mask = torch.zeros_like(attn_bias).bool()
        knn_mask.scatter_(1, knn_idx, True)                         # row-wise k nearest
        self.register_buffer('knn_mask', knn_mask[None, :])

    def forward(self,x):
        q = self.q_proj(x).view(-1, self.D, self.C)
        k = self.k_proj(x).view(-1, self.D, self.C)
        v = self.v_proj(x).view(-1, self.D, self.C)
        
        scores = (q.mT @ k) * self.scale 
        scores += self.attn_bias.view(1, self.C, self.C)

        scores = scores.view(-1, self.H, self.C, self.C)
        dir_bias = (self.displacement[None, :, :, :] * F.normalize(self.dir_vec, dim=-1)[:, None, :, None]).mean(dim=-1)
        dir_bias = 0.5*(dir_bias - dir_bias.mT)
        scores += dir_bias - dir_bias.mean(dim=-1, keepdim=True)
        scores = scores.view(-1, self.C, self.C)
        scores = scores.masked_fill(~self.knn_mask, float('-inf'))

        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn = (F.softmax(scores, dim=-1) @ v.mT).mT.contiguous()
        out = self.proj_out(attn.view(-1, self.H*self.D, self.C))
        return out
    
class STMHA(torch.nn.Module):
    def __init__(self, n_classes, n_channels, n_timepoints, hidden_spatial_dim=16, hidden_temporal_dim=64, n_heads=4, spatial_bias=True, self_normalizing=True):
        super().__init__()
        self.C=n_channels; self.T=n_timepoints; self.F=hidden_temporal_dim
        cwt_dim = 32
        self.cwt = CWT(n_timepoints, 62.5, freqs_per_band=cwt_dim)
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(1, hidden_temporal_dim, (1, 5), padding=(0, 2)),
            nn.SELU(),
        )
        self.spectral_fusion = nn.Sequential(
            nn.Conv2d(hidden_temporal_dim+cwt_dim, hidden_temporal_dim, (1, 1)),
            nn.SELU(),
            nn.Conv2d(hidden_temporal_dim, hidden_temporal_dim, (1, n_timepoints), groups=hidden_temporal_dim),
            nn.SELU(),
            nn.Conv2d(hidden_temporal_dim, hidden_temporal_dim, (1, 1)),
            nn.SELU(),
        )

        self.attn = SpatialMHA(n_channels, hidden_temporal_dim, n_heads, spatial_bias=spatial_bias)

        self.spatial_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_spatial_dim),
            nn.SELU(),
        )
        self.head=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_temporal_dim*hidden_spatial_dim, n_classes),
        )

        self.scale = (2**-0.5) if self_normalizing else 1
        def lecun_init(layer):            
            if isinstance(layer, (nn.Parameter, nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        if self_normalizing:
            self.apply(lecun_init)

    def forward(self, x):
        x_cwt = (self.cwt(x).permute(0, 2, 1, 3) - 0.5433)/0.4020
        x = self.temporal_proj(x.unsqueeze(1))
        x = torch.cat([x, x_cwt], dim=1)
        x = self.spectral_fusion(x).squeeze(-1)
        x = (x + self.attn(x)) * self.scale
        x = self.spatial_proj(x)
        return self.head(x)

def arc_dist(pos):
    radii = pos.norm(dim=1, keepdim=True)
    pos = pos / radii
    angular_dist = torch.arccos((pos @ pos.T).clamp(-1, 1))
    arcs = radii.mean() * angular_dist
    return arcs
    
if __name__ == "__main__":
    from src.experiments.utils import get_data_manager, apply_transforms
    from src.experiments.experiment_loader import load_experiment_cfg
    from src.models.basemodel import TorchGeometricModel
    from torch_geometric.data import HeteroData


    device='cuda'
    fold_idx, nested_fold_idx = 0, None
    params = {'max_epochs': 100, 
              'batch_size': 64, 
              'lr': 0.0001, 
              'weight_decay': 0.00001}
    
    with torch.device(device):
        exp_cfg = load_experiment_cfg("dev_SUDB_full_response_category_decoding")
        data_manager = get_data_manager(exp_cfg, "STMHA", "S1", 0, None, slice(None), slice(None), slice(None), device=device)
        model_cfg = exp_cfg['models']["STMHA"]
        model = model_cfg['class'](**model_cfg['arch_config'], device=device)
        
        transforms = model_cfg['transforms']
        t = {k: v['method'](**v.get('kwargs', {})) for k, v in  transforms.items()}

        train_data = data_manager.get_data_partition('train')
        train_data = apply_transforms(train_data, t, fit=True)
        
        test_partitions = exp_cfg['scheme']['test_partitions'] if nested_fold_idx is None else exp_cfg['scheme']['val_partitions']
        test_data = {partition:
                        apply_transforms(
                            data_manager.get_data_partition(partition),
                        t, fit=False)
                    for partition in test_partitions}

        trainer = model.get_trainer(**model_cfg['trainer_config'], **params)
        train_loader = trainer.build_dataloader(train_data, shuffle=True)
        test_loaders = {partition: trainer.build_dataloader(data, shuffle=False) for partition, data in test_data.items()}
        
        for epoch in range(params['max_epochs']):
            loss = trainer.train(train_loader)
            for partition, loader in test_loaders.items():
                labels, logits, loss = trainer.predict_proba(loader)
                accuracy = (labels == logits.argmax(axis=1)).mean()                       
                print(f"Epoch {epoch+1}-{partition}: {accuracy*100:.2f}%, {loss:.4f}.")