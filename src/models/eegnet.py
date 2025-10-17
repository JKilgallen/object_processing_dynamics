from torch import nn
from src.models.basemodel import TorchModel

class EEGNet(TorchModel):


    def __init__(self, n_classes=6, n_channels=124, n_time_points=32, n_temporal_filters = 16, temporal_kernel_size = 64, n_spatial_filters = 64, n_pointwise_filters=256, pointwise_kernel_size = 16, pool_kernel_1_size = 4, pool_kernel_2_size = 8, dropout_rate = 0.5,  device = 'cpu'):

        super().__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, n_temporal_filters=n_temporal_filters, temporal_kernel_size=temporal_kernel_size, n_spatial_filters=n_spatial_filters, n_pointwise_filters=n_pointwise_filters, pointwise_kernel_size=pointwise_kernel_size, pool_kernel_1_size=pool_kernel_1_size, pool_kernel_2_size=pool_kernel_2_size, dropout_rate=dropout_rate, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, n_temporal_filters, temporal_kernel_size, n_spatial_filters, n_pointwise_filters, pointwise_kernel_size, pool_kernel_1_size, pool_kernel_2_size, dropout_rate):
        self.N = n_classes
        self.C = n_channels
        self.T = n_time_points

        self.F1 = n_temporal_filters
        self.F1_size = temporal_kernel_size

        self.D = n_spatial_filters

        self.F2 = n_pointwise_filters
        self.F2_size = pointwise_kernel_size

        self.P1 = pool_kernel_1_size
        self.P2 = pool_kernel_2_size
        self.dropout = dropout_rate 

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.F1_size), padding="same", bias=False), # (F1, C, T)
            nn.BatchNorm2d(self.F1), # (F1, C, T)
            nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), groups=self.F1, padding="valid", bias=False), # (D * F1, 1, T)
            nn.BatchNorm2d(self.D * self.F1), # (D * F1, 1, T)
            nn.ELU(), # (D * F1, 1, T)
            nn.AvgPool2d((1, self.P1)), # (D * F1, 1, T // P)
            nn.Dropout(self.dropout)) # (D * F1, 1, T // P)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, kernel_size=(1, self.F2_size), groups = self.D * self.F1, padding="same", bias=False), # (D * F1, 1, T // P)
            nn.Conv2d(self.D * self.F1, self.F2, kernel_size=1, padding="same", bias=False), # (F2, 1, T // P)
            nn.BatchNorm2d(self.F2), # (F2, 1, T // P)
            nn.ELU(), # (F2, 1, T // P)
            nn.AvgPool2d((1, self.P2)), # (F2, 1, T // (P1 * P2))
            nn.Dropout(self.dropout))
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # (F2 * T // (P1 * P2))
            nn.Linear(self.F2 * self.T // (self.P1 * self.P2), self.N),
            nn.Softmax(dim=1))
        
    def forward(self, x):
        
        x = self.block1(x.unsqueeze(1))
        x = self.block2(x)
        x = self.classifier(x)

        return x
    
