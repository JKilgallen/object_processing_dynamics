from src.models.basemodel import TorchModel, TorchTrainer, BaseTrainer
import torch
from torch.nn import functional as F 
import numpy as np

class kNNTrainer(TorchTrainer):

    def __init__(self, model, criterion, batch_size, k=None, transforms=None):
        super(TorchTrainer, self).__init__(model=model, criterion=criterion, k=k, batch_size=batch_size, transforms=transforms)

        self.batch_size = batch_size
        self.criterion = criterion()
        if k:
            self.model.set_k(k)

    def extract_inputs(self, data):
        inputs, labels = data
        inputs = [inputs.reshape(inputs.shape[0], -1)]  # Flatten the inputs
        return inputs, labels

    def fit(self, train_data, val_data = None):
        inputs, labels = train_data.values()
        inputs, labels = self.extract_inputs((inputs, labels))
        self.model.set_feature_vectors(inputs[0], labels)
        
        train_loader = self.build_dataloader(train_data, shuffle=True)
        val_loader = self.build_dataloader(val_data, shuffle=False) if val_data else None

        train_loss = self.validate(train_loader)
        self.train_loss_history = np.array([train_loss])

        if val_loader:
            val_loss = self.validate(val_loader)
            self.val_loss_history = np.array([val_loss])

        if val_loader:
            return self.train_loss_history, self.val_loss_history
        else:
            return self.train_loss_history  
    
    def save_train_data(self, result_dir):
        return super(TorchTrainer, self).save_train_data(result_dir)

class kNN(TorchModel):

    def __init__(self, n_classes, k=None, metric = "euclidian", device = 'cpu'):
        super().__init__(n_classes=n_classes, k=k, metric=metric, device=device)
        self.samples = None
        self.labels = None
    
    def set_model_parameters(self, n_classes, k=None, metric = "euclidean"):
        self.n_classes = n_classes

        self.k = k
        self.metric = metric

        self.max_samples_per_iter = None
        return
    
    def set_k(self, k):
        self.k = k

    def get_trainer(self, criterion, batch_size, k=None, transforms=None, **kwargs):
        
        return kNNTrainer(model=self, criterion=criterion, batch_size=batch_size, k=k, transforms=transforms)
    
    def euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2)
    
    def angular_distance(self, x, y):
        cosine_similarities = (x / torch.norm(x, dim=1, keepdim=True)) @ (y / torch.norm(y, dim=1, keepdim=True)).T
        return torch.arccos(cosine_similarities)/torch.pi

    def set_feature_vectors(self, x_train, y_train):

        self.samples = x_train
        self.labels = y_train

        self.is_trained = True

    def forward(self, x):

        if self.samples is None or self.labels is None:
            raise ValueError("Model not trained")
        elif self.k is None:
            raise ValueError("k not set")

        if self.metric == "euclidean":
            dist_func = self.euclidean_distance
        elif self.metric == "angular":
            dist_func = self.angular_distance
        else:
            raise ValueError("Invalid metric")

        # Need to find:
        #   - The indices of the k nearest neighbors (shape: (n_queries, k))
        #   - The labels of the k nearest neighbors (shape: (n_queries, k))
        #   - The distances of the k nearest neighbors to the query points (shape: (n_queries, k))
        knn_idxs = torch.tensor([], dtype=torch.long, device=x.device)
        max_samples_per_iter = self.max_samples_per_iter if self.max_samples_per_iter else len(self.samples)
        for start_idx in range(0, len(self.samples) - max_samples_per_iter + 1, max_samples_per_iter):
            end_idx = start_idx + max_samples_per_iter

            # Since we update the indices, labels and distances in each iteration, we need to keep track of the original indices of the samples
            # For the idxs tensor we need the unique indices of the current k nearest neighbors and the indices of the current samples.
            # We also need the labels of the current k nearest neighbors and the labels of the current samples.
            # We then need to update the distances tensor with the distances of the current samples to the query points.
            candidate_idxs = torch.cat([knn_idxs.unique(), torch.arange(start_idx, end_idx, device=x.device)])
            candidate_distances = dist_func(x, self.samples[candidate_idxs])

            _, knn_idxs = candidate_distances.topk(self.k, dim=1, largest=False)
            knn_idxs = candidate_idxs[knn_idxs]
        out = F.one_hot(self.labels[knn_idxs], num_classes=self.n_classes).float().sum(dim=1) / self.k
        return out
