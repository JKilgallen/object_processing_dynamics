import torch
from torch import nn
from src.models.basemodel import TorchModel


class LogisticRegression(TorchModel):
    def __init__(self, n_classes=6, n_features = 124*32,  device = 'cpu'):

        super().__init__(n_classes=n_classes, n_features=n_features, device=device)

    def set_model_parameters(self, n_classes, n_features):
        self.n_classes = n_classes
        self.n_features = n_features

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.n_features, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
