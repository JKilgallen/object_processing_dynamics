import torch
from torch import nn
from src.models.basemodel import TorchModel

class ShallowConvNet(TorchModel):

    def __init__(self, n_classes=6, n_channels=124, n_time_points=32, n_temporal_filters = 16, temporal_kernel_size = 5, n_spatial_filters = 16, pool_method = "avg", pool_size = (1, 5), pool_stride = (1, 4), device = 'cpu'):
        super().__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, n_temporal_filters=n_temporal_filters, temporal_kernel_size=temporal_kernel_size, n_spatial_filters=n_spatial_filters, pool_method=pool_method, pool_size=pool_size, pool_stride=pool_stride, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, n_temporal_filters, temporal_kernel_size, n_spatial_filters, pool_method, pool_size, pool_stride):
        self.N = n_classes
        self.C = n_channels
        self.T = n_time_points

        self.F = n_temporal_filters
        self.F_size = temporal_kernel_size

        self.D = n_spatial_filters

        self.P = pool_method
        self.P_size = pool_size
        self.P_stride = pool_stride

        self.temporal_conv_block = nn.Sequential(
            nn.Conv1d(self.C, self.C*self.F, self.F_size, groups=self.C, bias=False),
            nn.BatchNorm1d(self.C*self.F),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.spatial_conv_block = nn.Sequential(
            nn.Conv1d(self.C*self.F, self.D, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.D),
            nn.ELU(),
            nn.Dropout(0.5)
        )

        self.avg_pool = nn.AvgPool2d(self.P_size, self.P_stride)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, self.N),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        
        x = self.temporal_conv_block(x)
        x = self.spatial_conv_block(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(x + 1e-6)
        x = self.classifier(x)
        
        return x
    