import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops.layers.torch import Rearrange
from src.models import EEGCT

class EEGConvTransformer(nn.Module):
    def __init__(self, C, P, T, H, E, F, n_classes=6):
        super(EEGConvTransformer, self).__init__()
        self.C = C
        self.P = P 
        self.T = T
        self.H = H
        self.E = E
        self.F = F

        self.local_feature_extractor = LocalFeatureExtractor(C)

        self.ct1 = ConvolutionalTransformer(C, T, H, E)
        self.ct2 = ConvolutionalTransformer(C, T, H, E)

        self.convolutional_encoder = ConvolutionalEncoder(C, P, F)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.F * self.T, 500),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.local_feature_extractor(x)
        x = self.ct1(x)
        x = self.ct2(x)
        x = self.convolutional_encoder(x)
        x = self.classifier(x)
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
        self.C = C
        self.T = T
        self.H = H

        assert C % H == 0, "Number of channels must be divisible by the number of heads."
        self.D = C // H

        self.keys = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1))
        self.queries = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1))
        self.values = nn.Conv2d(C, C, kernel_size=(1, 1), stride=(1, 1))

        self.split_heads = Rearrange('b (h d) p t -> b h d p t', h=H)
        self.merge_heads = Rearrange('b h d p t -> b (h d) p t')

        self.scale = (self.D * T) ** 0.5
        

    def forward(self, x):
        # x shape: [B, C, P, T]
        k = self.split_heads(self.keys(x)) # [B, H, D, P, T]
        q = self.split_heads(self.queries(x)) # [B, H, D, P, T]
        v = self.split_heads(self.values(x)) # [B, H, D, P, T]

        energy = torch.einsum('bhdpt, bhdqt -> bhpq', k, q)
        attn = F.softmax(energy/self.scale, dim=-1)
        out = self.merge_heads(torch.einsum('bhpq, bhdpt -> bhdqt', attn, v))

        return out

class ConvolutionalFeatureExpander(nn.Module):
    def __init__(self, C, E):
        super(ConvolutionalFeatureExpander, self).__init__()
        self.C = C
        self.E = E

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
        self.C = C
        self.T = T
        self.H = H
        self.E = E

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

nn.Softmax()
device = 'cuda'
subject = 'S6'
fold_idx = 0
data_dir = '/home/jkilgallen/Projects/repeated_stimulus_confound_alt/data/SUDB'

eeg = torch.load(f'{data_dir}/processed/{subject}/AEP.pt', map_location=device)
category = torch.load(f'{data_dir}/processed/{subject}/category.pt', map_location=device)
fold = torch.load(f'{data_dir}/cross_validation/{subject}/task/category_decoding.pth', map_location=device)[fold_idx]

# data_loader = DataLoader(TensorDataset(torch.rand((256, 32, 32, 32)), torch.randint(0, 6, (256,))), batch_size=64, shuffle=True)
train_loader = DataLoader(TensorDataset(eeg[fold['train_val']], 
                                        category[fold['train_val']]), 
                          batch_size=64, shuffle=True)

variants = {'CT-Slim': {'C': 8, 'P': 49, 'T': 32, 'H': 4, 'E': 16, 'F': 256},
            'CT-Fit': {'C': 32, 'P': 49, 'T': 32, 'H': 8, 'E': 128, 'F': 512},
            'CT-Wide': {'C': 72, 'P': 49, 'T': 32, 'H': 12, 'E': 432, 'F': 768}}

net = EEGConvTransformer(C=8, P=49, T= 32, H=4, E=16, F=256).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
max_epochs = 35

# get the number of parameters in the model
for var, params in variants.items():
    net = EEGConvTransformer(**params)
    print(f"Variant: {var}, Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
exit()
for epoch in range(max_epochs):
    running_loss = 0.0
    for x, y in train_loader:
        out = net(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{max_epochs}, Loss: {running_loss:.4f}")

test_loader = DataLoader(TensorDataset(eeg[fold['confounded_test']],
                                       category[fold['confounded_test']]), 
                          batch_size=64, shuffle=False)

labels = []
preds = []
for x, y in test_loader:
    out = net(x)
    labels.append(y), preds.append(out.argmax(dim=1))
labels, preds = torch.cat(labels), torch.cat(preds)
accuracy = (labels == preds).float().mean().item()
print(f"Test Accuracy: {accuracy:.4f}")

