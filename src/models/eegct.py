import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from src.models.basemodel import TorchModel

torch.backends.cudnn.benchmark = True        
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True       
torch.set_float32_matmul_precision('high')

class EEGConvTransformer(TorchModel):

    def __init__(self, C, T, H, E, F, n_classes=6, device='cpu'):
        super().__init__(C=C, T=T, H=H, E=E, F=F, n_classes=n_classes, device=device)

    def set_model_parameters(self, C, T, H, E, F, n_classes):
        self.C = C
        self.T = T
        self.H = H
        self.E = E
        self.F = F

        self.lfe = LocalFeatureExtractor(C)

        self.ct1 = ConvolutionalTransformer(C, T, H, E)
        self.ct2 = ConvolutionalTransformer(C, T, H, E)
        
        P = ((32 - 8)//4 + 1) ** 2
        self.ce = ConvolutionalEncoder(C, P, F)

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F * self.T, 500),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x, softmax=False):
        x = self.lfe(x)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ce(x)
        x = self.clf(x)
        if softmax:
            x = nn.functional.softmax(x, dim=1)
        return x

class LocalFeatureExtractor(nn.Module):
    def __init__(self, C):
        super(LocalFeatureExtractor, self).__init__()
        self.C = C
        self.neighbor_conv = nn.Conv3d(1, C//2, kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(0, 0, 1))
        self.region_conv = nn.Conv3d(1, C//2, kernel_size=(8, 8, 5), stride=(4, 4, 1), padding=(0, 0, 2))
        self.batch_norm = nn.BatchNorm3d(C)
        self.elu = nn.ELU()


    def forward(self, x):
        # x shape: [B, M1, M2, T]
        b, _, _, t = x.shape
        x = x.unsqueeze(1)

        neighor_patches = self.neighbor_conv(x)
        region_patches = self.region_conv(x)
        patches = torch.cat((neighor_patches, region_patches), dim=1)

        out = self.elu(self.batch_norm(patches))

        return out.view(b, self.C, -1, t)

class MultiHeadAttention(nn.Module):
    def __init__(self, C, T, H):
        super(MultiHeadAttention, self).__init__()

        assert C % H == 0, "Number of channels must be divisible by the number of heads."

        self.keys = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.queries = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.values = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.split_heads = Rearrange('b (h d) p t -> b h d p t', h=H)
        self.merge_heads = Rearrange('b h d p t -> b (h d) p t')

        self.scale = ((C//H) * T) ** 0.5
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: [B, C, P, T]
        k = self.split_heads(self.keys(x)) # [B, H, D, P, T]
        q = self.split_heads(self.queries(x)) # [B, H, D, P, T]
        v = self.split_heads(self.values(x)) # [B, H, D, P, T]

        energy = torch.einsum('bhdpt, bhdqt -> bhpq', k, q)
        attn = self.softmax(energy/self.scale)
        out = self.merge_heads(torch.einsum('bhpq, bhdpt -> bhdqt', attn, v))

        return out

class ConvolutionalFeatureExpander(nn.Module):
    def __init__(self, C, E):
        super(ConvolutionalFeatureExpander, self).__init__()

        self.conv1 = nn.Conv2d(C, E//2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(C, E//2, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2))
        self.batch_norm = nn.BatchNorm2d(E)
        self.elu = nn.ELU()
        self.conv3 = nn.Conv2d(E, C, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = torch.cat((self.conv1(x), self.conv2(x)), dim=1)
        x = self.elu(self.batch_norm(x))
        x = self.conv3(x)
        return x
    
class ConvolutionalTransformer(nn.Module):
    def __init__(self, C, T, H, E):
        super(ConvolutionalTransformer, self).__init__()

        self.mha = MultiHeadAttention(C, T, H)

        self.bn1 = nn.BatchNorm2d(C)
        self.convolutional_feature_expander = ConvolutionalFeatureExpander(C, E)
        self.bn2 = nn.BatchNorm2d(C)

    def forward(self, x):
        x = self.mha(x) + x
        x = self.bn1(x)

        x = self.convolutional_feature_expander(x) + x
        x = self.bn2(x)

        return x

class ConvolutionalEncoder(nn.Module):
    def __init__(self, C, P, F):
        super(ConvolutionalEncoder, self).__init__()
        self.C = C
        self.F = F

        self.conv1 = nn.Conv2d(P, F//2, kernel_size=(C, 3), stride=(1, 1), padding=(0, 1))
        self.conv2 = nn.Conv2d(P, F//2, kernel_size=(C, 5), stride=(1, 1), padding=(0, 2))
        self.batch_norm = nn.BatchNorm2d(F)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = torch.cat((self.conv1(x), self.conv2(x)), dim=1)
        x = self.elu(self.batch_norm(x))
        return x
    
if __name__ == '__main__':
    import time
    n_iter = 10
    x = torch.randn(64, 32, 32, 32).to("cuda")
    with torch.no_grad():
        t0 = time.time()
        model = EEGConvTransformer(C=72, T=32, H=12, E=432, F=768, n_classes=6).to("cuda", non_blocking=True)                      # large copy
        torch.cuda.synchronize(); t1 = time.time()
        avg_time_copy = 0.0
        for t in range(n_iter):
            torch.cuda.synchronize(); start = time.time()
            out = model(x)                                 # forward pass
            torch.cuda.synchronize(); end = time.time()
            avg_time_copy += (end - start)
        avg_time_copy /= n_iter

        t2 = time.time()
        with torch.device("cuda"):
            model = EEGConvTransformer(C=72, T=32, H=12, E=432, F=768, n_classes=6, device="cuda")
        torch.cuda.synchronize(); t3 = time.time()
        avg_time_direct = 0.0
        for t in range(n_iter):
            torch.cuda.synchronize(); start = time.time()
            out = model(x)                                 
            torch.cuda.synchronize(); end = time.time()
            avg_time_direct += (end - start)
        avg_time_direct /= n_iter

        t4 = time.time()
        with torch.device("cuda"):
            model = EEGConvTransformer(C=72, T=32, H=12, E=432, F=768, n_classes=6, device="cuda")
            model = torch.compile(model, fullgraph=True, dynamic=False, mode="reduce-overhead")
        avg_time_compile = 0.0
        torch.cuda.synchronize(); t5 = time.time()
        for t in range(n_iter):
            torch.cuda.synchronize(); start = time.time()
            out = model(x)                                 
            torch.cuda.synchronize(); end = time.time()
            avg_time_direct += (end - start)
        avg_time_compile /= n_iter

    print("cpu->cuda:", t1 - t0, "forward:", avg_time_direct)
    print("cuda->cuda:", t3 - t2, "forward:", avg_time_direct)
    print("cuda->cuda compile:", t5 - t4, "forward:", avg_time_direct)