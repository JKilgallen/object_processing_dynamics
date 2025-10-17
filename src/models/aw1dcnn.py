import torch.nn as nn
from src.models.basemodel import TorchModel

class AW1DCNN(TorchModel):
    def __init__(self, n_classes, n_channels, n_filters=512, hidden_dim_1=50, hidden_dim_2=100, kernel_size=5, device='cpu'):

        super().__init__(n_classes=n_classes, n_channels=n_channels, n_filters=n_filters, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, kernel_size=kernel_size, device=device)

    def set_model_parameters(self, n_classes=None, n_channels=None, n_filters=None, hidden_dim_1=None, hidden_dim_2=None, kernel_size=None):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.kernel_size = kernel_size

        self.l_conv = nn.Sequential(
            nn.Conv1d(self.n_channels, self.n_filters, self.kernel_size, padding='same'),
            nn.BatchNorm1d(self.n_filters),
            nn.ELU()
        )

        self.l_res1 = self.ResidualConnectionBlock(self.n_filters, self.kernel_size)
        self.l_res2 = self.ResidualConnectionBlock(self.n_filters, self.kernel_size)

        self.l_cls = nn.Sequential(nn.Flatten(),
                                    nn.Linear(16384, self.hidden_dim_1),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim_2, self.n_classes),
                                    # nn.Softmax(dim=1)
        )

    def forward(self, x, softmax=False):
        x = self.l_conv(x)
        x = self.l_res1(x)
        x = self.l_res2(x)
        x = self.l_cls(x)
        if softmax:
            x = nn.functional.softmax(x, dim=1)
        return x

    class ResidualConnectionBlock(nn.Module):
        def __init__(self, n_channels, kernel_size):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_channels, n_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(n_channels)
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(n_channels, n_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(n_channels)
            )

            self.elu = nn.ELU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.elu(x)
            res = x
            x = self.conv2(x)
            x += res
            x = self.elu(x)
            
            return x