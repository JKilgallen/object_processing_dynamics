import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from src.models.basemodel import TorchGeometricModel

class TSCNN(TorchGeometricModel):
    
    def __init__(self, n_classes, n_channels, n_time_points, hidden_dim_1, hidden_dim_2, time_kernel, pool_kernel, device='cpu'):
        super(TSCNN, self).__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, time_kernel=time_kernel, pool_kernel=pool_kernel, device=device)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, hidden_dim_1, hidden_dim_2, time_kernel, pool_kernel):
        from torch_geometric.nn import GraphConv

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_time_points = n_time_points
        self.hidden_dim_1 = hidden_dim_1 
        self.hidden_dim_2 = hidden_dim_2
        self.time_kernel = time_kernel
        self.pool_kernel = pool_kernel
        
        #Graph Convolutional Neural Network (GCNN)
        self.gconv1 = GraphConv(self.n_time_points, self.hidden_dim_1)
        self.gconv2 = GraphConv(self.hidden_dim_1, self.hidden_dim_1)
        self.act = nn.ELU()
        self.rearrange = Rearrange('(b n) f -> b n f', n=self.n_channels, f=self.n_time_points)
        self.avgpool = nn.AvgPool2d(kernel_size=(1,5))

        # One-tream Convolutional Neural Network (OSCNN)
        self.oscnn = nn.Sequential(Rearrange('(b n) f -> b 1 n f', n=self.n_channels, f=self.n_time_points),
                                   nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, self.time_kernel), padding=0),
                                   nn.BatchNorm2d(16),
                                   nn.ELU(True), 
                                   nn.Conv2d(16, 16, (self.n_channels, 1), stride=1, padding=0),
                                   nn.BatchNorm2d(16),
                                   nn.Dropout2d(p=0.25),
                                   nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, self.time_kernel), padding=0),
                                   nn.BatchNorm2d(32),
                                   nn.ELU(True),
                                   nn.MaxPool2d(kernel_size=(1, self.pool_kernel), stride=(1, self.pool_kernel), padding=0),
                                   nn.Dropout2d(p=0.25),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, self.time_kernel), padding=0),
                                   nn.BatchNorm2d(64),
                                   nn.ELU(True),
                                   nn.MaxPool2d(kernel_size=(1, self.pool_kernel), stride=(1, self.pool_kernel), padding=0))
        
        self.lin = nn.Linear(self.hidden_dim_2, self.n_classes)

    def forward(self, x, edge_index, edge_attr, softmax=False):
        ## Graph Convolutional Neural Network (GCNN)
        x_gcnn = self.gconv1(x, edge_index, edge_attr)
        x_gcnn = self.act(x_gcnn)
        x_gcnn = self.gconv2(x_gcnn, edge_index, edge_attr)
        x_gcnn = self.act(x_gcnn)
        x_gcnn = self.rearrange(x_gcnn)
        x_gcnn = self.avgpool(x_gcnn)

        x_gcnn = x_gcnn.view(x_gcnn.size(0), -1)

        # One Stream Convolutional Neural Network (OSCNN)
        x_oscnn = self.oscnn(x)
        x_oscnn = x_oscnn.view(x_oscnn.size(0), -1)

        # Concatenate GCNN and OSCNN outputs and pass through linear layer
        x_out = torch.cat((x_gcnn, x_oscnn), 1)
        x_out = self.lin(x_out)
        if softmax:
            x_out = nn.functional.softmax(x_out, dim=1)
        return x_out

#     def get_trainer(self, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, gamma=None, milestones=None, transforms=None):
#         trainer = BaseTrainer(self, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, gamma=None, milestones=None, transforms=None)
#         trainer.extract_inputs = extract_inputs.__get__(trainer)
#         trainer.build_dataloader = build_dataloader.__get__(trainer)
#         return trainer
    
# def extract_inputs(self, data):
#     *inputs, labels = data.x, data.edge_index, None, data.y
#     return inputs, labels

# def build_dataloader(self, data, shuffle=True):
#     from torch_geometric.data import Data
#     from torch_geometric.loader import DataLoader
#     EEG, edge_index, labels = data.values()
#     data = [Data(x=EEG[i], edge_index=edge_index[i], y=labels[i]) for i in range(len(EEG))]
#     return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)