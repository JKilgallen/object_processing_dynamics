import torch
from torch import nn
from src.models.basemodel import TorchModel

OCCIPITAL_ELECTRODES = {'HGSN_128': [65, 66, 67, 68, 69, 70, 71,
                                     72, 73, 74, 75, 76, 77, 81,
                                     82, 83, 84, 88, 89, 90, 94],
                        'BIOSEMI_96': [i - 1 for i in [83, 82, 81, 80, 79, 
                                                       85, 86, 87, 88, 89, 
                                                       94, 93, 92, 90, 
                                                       95, 96, 91]]}

class ADCNN(TorchModel):
    def __init__(self, n_classes, n_channels, n_time_points, montage, device = 'cpu'):
        super().__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, montage=montage, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, montage):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_time_points
        self.montage = montage

        self.register_buffer('mask', torch.zeros(1, 1, self.n_channels, 1))
        self.occipital_electrodes = OCCIPITAL_ELECTRODES[montage]
        self.mask[:, :, self.occipital_electrodes, :] = 1
        
        self.cnn_block_1 = self.CNNBlock(n_channels=n_channels)
        self.cnn_block_2 = self.CNNBlock(n_channels=n_channels)

        self.classifier_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2400, self.n_classes),
        )
        
    def forward(self, x, softmax=False):
        """
        Performs a forward pass through the ADCNN model.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 124, 32).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 6).
        """

        x = x.unsqueeze(1) # (batch_size, 1, 124, 32)

        x_masked = self.mask * x
        
        x1 = self.cnn_block_1(x)
        x2 = self.cnn_block_2(x_masked)

        out = torch.cat((x1, x2), 1)
        out = self.classifier_block(out)
        if softmax:
            out = nn.functional.softmax(out, dim=1)
        return out
    
    class CNNBlock(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            
            self.cnn_block = nn.Sequential(
                nn.Conv2d(1, 20, (1, 5), padding='valid'),
                nn.BatchNorm2d(20),
                nn.Conv2d(20, 20, (n_channels, 1), padding='valid'),
                nn.BatchNorm2d(20),
                nn.Dropout(0.5),
                nn.Conv2d(20, 40, (1, 5), padding='valid'),
                nn.BatchNorm2d(40),
                nn.Conv2d(40, 100, (1, 10), padding='valid'),
                nn.BatchNorm2d(100),
                nn.Conv2d(100, 200, (1, 10), padding='valid'),
                nn.BatchNorm2d(200)
            )

        def forward(self, x):
            x = self.cnn_block(x)
            return x
    
