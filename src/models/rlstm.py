import torch
from torch import nn
from src.models.basemodel import TorchModel
from einops.layers.torch import Rearrange
# class RLSTM(TorchModel):
#     def __init__(self, n_classes, n_channels, n_time_points, hidden_dim=16, device='cpu'):
#         super(RLSTM, self).__init__(n_classes=n_classes, n_channels=n_channels, n_time_points=n_time_points, hidden_dim=hidden_dim, device=device)
    
#     def set_model_parameters(self, n_classes, n_channels, n_time_points, hidden_dim):
#         self.n_classes = n_classes
#         self.n_channels = n_channels
#         self.n_time_points = n_time_points
#         self.hidden_dim = hidden_dim
    
#         self.channel_weights = nn.Parameter(torch.ones(1, self.n_channels, 1))

#         self.batch_channnel = Rearrange('b c t -> (b c) t')
#         self.unbatch_channel = Rearrange('(b c) t -> b c t', c=self.n_channels)

#         self.tfe = nn.ModuleDict({
#             'lstm': nn.LSTM(1, self.hidden_dim, batch_first = True),
#             'dropout': nn.Dropout(0.5),
#             # 'mha': self.MultiHeadAttention(1, 1)
#         })

#         self.ffe = nn.Sequential(
#             nn.Conv1d(1, 2, kernel_size = 4, stride = 2, padding = 1),
#             nn.Conv1d(2, 4, kernel_size = 4, stride = 2, padding = 1),
#             nn.Conv1d(4, 8, kernel_size = 4, stride = 2, padding = 1),
#             nn.Conv1d(8, 16, kernel_size = 4, stride = 1, padding = 0),
#             nn.LeakyReLU()
#         )

#         self.lstm = nn.LSTM(self.n_channels*2, 64, batch_first = True)
#         self.dropout = nn.Dropout(0.5)

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(16*64, 100*self.n_classes),
#             nn.LeakyReLU(),
#             nn.Linear(100*self.n_classes, self.n_classes),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         """
#         Forward pass of the model.
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, n_channels, n_time_points).
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, n_classes).
#         """

#         # x is of shape (batch_size, n_channels, n_time_points)


#         x = x * self.channel_weights # (batch_size, n_channels, n_time_points)

#         # Reshape the input so that each channel is a separate sequence
#         x = self.batch_channnel(x) # (batch_size * n_channels, n_time_points)
#         x1 = x.unsqueeze(-1) # (batch_size * n_channels, n_time_points, 1)
#         x2 = x.unsqueeze(1) # (batch_size * n_channels, 1, n_time_points)
        
#         # TEMPORAL FEATURE EXTRACTION
#         # ---------------------------
#         # The paper says "This study used LSTM as the method to extract time feature. 
#         # First, the single channel EEG signal passes through an LSTM layer with a hidden size of 16. 
#         # After that, the reduced dimension EEG signal is input to the Multi-Head attention. 
#         # The final result is regarded as the time feature extracted from a single channel EEG signal."
#         #
#         # Our interpretation of this is that the LSTM takes the sequence of EEG data for each channel,
#         # and considers it to be a sequence of 1-dimensional data points. The LSTM then outputs a hidden
#         # state of size 16 for each time point in the sequence. We then take the hidden state at the last
#         # time point as the output of the LSTM for each channel. To pass this output to the Multi-Head
#         # Attention layer, we then need to either consider it as a sequence of length 1 with 16 features,
#         # or as a sequence of 16 features with length 1. Choosing the former would mean that the Multi-Head
#         # Attention layer only has one time point to consider, which seems to defeat the purpose of using
#         # an attention mechanism. However, choosing the latter would mean that the Multi-Head Attention layer
#         # would consider have and embeded dimension of 1, which also seems to defeat the purpose of using
#         # a Multi-Head Attention layer as it would only be able to use a single attention head. 
#         # On balance we have chosen the former approach as the Multi-Head Attention layer may have been chosen
#         # due to the fact that pytorch lacks a single-head attention mechanism with learnable transformations, 
#         # while there is no explanation of why an attention mechanism would be used if the sequence was of length 1.
#         # This would also explain why no number of heads is specified in the paper as if the number of features
#         # is 1, then the number of heads must also be 1.

#         x1, _ = self.tfe['lstm'](x1) # (batch_size * n_channels, n_time_points, 16)
#         x1 = x1[:, -1, :].unsqueeze(-1) # (batch_size * n_channels, 16, 1)
#         x1 = self.tfe['dropout'](x1) # (batch_size * n_channels, 16, 1)
#         # x1 = self.tfe['mha'](x1).squeeze(-1) # (batch_size * n_channels, 16)
#         x1 = nn.functional.scaled_dot_product_attention(x1, x1, x1).squeeze(-1) # (batch_size * n_channels, 16)
#         x1 = self.unbatch_channel(x1) # (batch_size, n_channels, 16)

#         # FREQUENCY FEATURE EXTRACTION
#         # ----------------------------
#         x2 = self.ffe(x2).squeeze(-1) # (batch_size * n_channels, 16, 1)
#         x2 = self.unbatch_channel(x2) # (batch_size, n_channels, 16)

#         # FUSION
#         # ------
#         # The paper says:
#         # "After combining the features extracted by LSTM and one-dimensional convolution, the final number of features input
#         # into the classifier is 16*2*K. The LSTM layer in the classifier is used to reduce the large number of channels ... 
#         # The classifier consists of one layer of LSTM and two layers of FC, the LSTM layer in the classifier is used to reduce 
#         # the large number of channels. The number of features of 16*2*K input is reduced to 16 * 64."
#         # 
#         # This means the LSTM layer has a hidden size of 64, and our input is of shape (2*K, 16). And since we are told the LSTM
#         # layer in the classifier is used to reduce the large number of channels, we can infer that the input to the LSTM layer
#         # is a sequence of length 16 with 2*K features, and the output is a sequence of length 16 with 64 features as stated in the
#         # paper.
        
#         out = torch.cat((x1, x2), 1) # (batch_size, n_channels * 2, 16)
#         out = out.permute(0, 2, 1) # (batch_size, 16, n_channels * 2)
#         out, _ = self.lstm(out)
#         out = self.dropout(out)

#         out = self.classifier(out)

#         return out

from einops import rearrange  
class RLSTM(TorchModel):
    def __init__(self, n_channels, device='cpu', **kwargs):
        super().__init__(n_channels=n_channels, device=device, **kwargs)

    def set_model_parameters(self, n_classes, n_channels, n_time_points, **kwargs):
        self.n_channels = n_channels
        self.channel_weights = nn.Parameter(torch.ones(1, self.n_channels, 1))

        self.batch_channnel = Rearrange('b c t -> (b c) t')
        self.unbatch_channel = Rearrange('(b c) t -> b t c', c=self.n_channels)

        self.channel_lstm = nn.LSTM(1, 16, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.mha = nn.MultiheadAttention(16, 4, batch_first=True)

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=4, stride=2),
            nn.Conv1d(2, 4, kernel_size=4, stride=1),
            nn.Conv1d(4, 8, kernel_size=4, stride=2),
            nn.Conv1d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(self.n_channels*2, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*64, 600),
            nn.ReLU(),
            nn.Linear(600, n_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x, softmax=False):
        x = x * self.channel_weights
        x = self.batch_channnel(x) 

        x_temp = x.unsqueeze(-1) # (batch_size * n_channels, n_time_points, 1)
        x_temp = self.channel_lstm(x_temp)[0] # (batch_size * n_channels, n_time_points, 16)
        x_temp = self.dropout1(x_temp) # (batch_size * n_channels, n_time_points, 16)
        x_temp = self.mha(x_temp, x_temp, x_temp)[0][:, -1, :] # (batch_size * n_channels, 16)
        x_temp = self.unbatch_channel(x_temp)

        x_spatial = x.unsqueeze(1) # (batch_size * n_channels, 1, n_time_points)
        x_spatial = self.spatial_conv(x_spatial).squeeze(-1) # (batch_size * n_channels, 16)
        x_spatial = self.unbatch_channel(x_spatial) # (batch_size, n_channels, 16)

        out = torch.cat((x_temp, x_spatial), dim=2)
        out = self.lstm(out)[0]
        out = self.dropout2(out)  # (batch_size, n_time_points, 64)

        out = self.classifier(out)
        if softmax:
            out = nn.functional.softmax(out, dim=1)
        return out

# import torch

# if __name__ == "__main__":
#     net = RLSTM(n_classes=6, n_channels=124, n_time_points=32, hidden_dim=16).to('cuda')

#     eeg= torch.load("/home/jkilgallen/Projects/eeg_decoding/data/SUDB/processed/S6/EEG.pt", map_location='cuda')
#     category = torch.load("/home/jkilgallen/Projects/eeg_decoding/data/SUDB/processed/S6/category.pt", map_location='cuda')
#     exemplar = torch.load("/home/jkilgallen/Projects/eeg_decoding/data/SUDB/processed/S6/exemplar.pt", map_location='cuda')
#     folds = torch.load("/home/jkilgallen/Projects/imagenet_SUCKS/data/SUDB/category_decoding.pth", map_location='cuda')

#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(eeg[folds[0][0]['train']], 
#                                     category[folds[0][0]['train']]),
#         batch_size=64,
#         shuffle=True,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(
#             eeg[folds[0][0]['confounded_val']], 
#             category[folds[0][0]['confounded_val']]),
#         batch_size=64,
#         shuffle=False,
#     )

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)
#     max_epochs = 30
#     for epoch in range(max_epochs):
#         train_loss, val_loss = 0.0, 0.0
#         for inputs, labels in train_loader:
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         for inputs, labels in val_loader:
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#         train_loss/= len(train_loader)
#         val_loss /= len(val_loader)
#         print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

#     test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(eeg[folds[0][0]['confounded_test']], category[folds[0][0]['confounded_test']]),
#         batch_size=64,
#         shuffle=False,
#     )

#     y_true, y_pred = [], []
#     for inputs, labels in test_loader:
#         outputs = net(inputs)
#         y_true.append(labels.cpu())
#         y_pred.append(outputs.argmax(dim=1).cpu())
#     y_true = torch.cat(y_true)
#     y_pred = torch.cat(y_pred)

#     print(f'Test Accuracy: {torch.sum(y_true == y_pred) / len(y_true):.4f}')