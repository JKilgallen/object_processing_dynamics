from torch import nn
from einops.layers.torch import Rearrange
from src.models.basemodel import TorchModel

class DeepConvNet(TorchModel):
    def __init__(self, n_classes=6, n_channels=124, n_time_points=32, n_temporal_filters = 25, temporal_kernel_size = 3, n_spatial_filters = 25, pool_method='max', pool_size = 3, pool_stride = 3, pool_padding = 1, device = 'cpu'):

        super().__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, n_temporal_filters=n_temporal_filters, temporal_kernel_size=temporal_kernel_size, n_spatial_filters=n_spatial_filters, pool_method=pool_method, pool_size=pool_size, pool_stride=pool_stride, pool_padding=pool_padding, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, n_temporal_filters, temporal_kernel_size, n_spatial_filters, pool_method, pool_size, pool_stride, pool_padding):
        self.N = n_classes
        self.C = n_channels
        self.T = n_time_points

        self.F = n_temporal_filters
        self.F_size = temporal_kernel_size

        self.D = n_spatial_filters

        self.P = pool_method
        self.P_size = pool_size
        self.P_stride = pool_stride
        self.P_padding = pool_padding

        self.block1 = nn.Sequential(
            nn.Conv1d(self.C, self.C*self.F, self.F_size, groups=self.C, padding='same', bias=False), # (C, T)
            nn.BatchNorm1d(self.C * self.F),
            nn.Dropout(0.5),
            nn.Conv1d(self.C * self.F, self.D, 1, bias=False),
            nn.BatchNorm1d(self.D),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(self.P_size, self.P_stride, self.P_padding), # (D, C, T)
        )

        self.block2 = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, self.D*2, (self.D, self.F_size), bias=False, padding=(0, 3)),
            Rearrange('b c 1 t -> b (c 1) t'),
            nn.BatchNorm1d(self.D*2),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(self.P_size, self.P_stride, self.P_padding)
        )

        self.block3 = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, self.D*4, (self.D*2, self.F_size), bias=False, padding=(0, 3)),
            Rearrange('b c 1 t -> b (c 1) t'),
            nn.BatchNorm1d(self.D*4),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(self.P_size, self.P_stride, self.P_padding) # (D, C, T)
        )

        self.block4 = nn.Sequential(
            Rearrange('b c t -> b 1 c t'),
            nn.Conv2d(1, self.D*8, (self.D*4, self.F_size), bias=False, padding=(0, 3)), 
            Rearrange('b c 1 t -> b (c 1) t'),
            nn.BatchNorm1d(self.D*8),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(self.P_size, self.P_stride), # (D, C, T)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, self.N),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)

        return x