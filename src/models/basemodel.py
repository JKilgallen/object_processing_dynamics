from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.amp import autocast
import os
from sklearn import config_context
from contextlib import nullcontext
import numpy as np

torch.backends.cudnn.benchmark = True        
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True       
torch.set_float32_matmul_precision('high')

SKLEARN_GPU_ENABLED = True

class BaseModel:
    def get_trainer(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the get_trainer method.")

class BaseTrainer:
    def __init__(self, model, **kwargs):
        """Initializes the trainer. To be extended by subclasses."""
        self.model = model
        self.is_trained = False

        self.train_loss_history = []
        self.val_loss_history = []

        self.hyperparameters = kwargs
        
    def train(self, train_data):
        """Fits the model to the training data. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the fit method.")
    
    def predict_proba(self, test_data):
        """Predicts probabilities for the test data. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the predict_proba method.")
    
    def save_test_data(self, f, partition_id, epoch, labels, logits, loss):
        """Saves the predictions to a file."""

        if 'epoch' not in f:
            f.attrs['hyperparameters'] = str(self.hyperparameters)
            f.create_group('epoch')
        if f'{epoch+1}' not in f['epoch']:
            f['epoch'].create_group(f'{epoch+1}')

        partition = f['epoch'][f'{epoch+1}'].create_group(partition_id)
        if 'test' in partition_id:
            partition.create_dataset('logits', data=logits)
            partition.create_dataset('labels', data=labels)
        partition.attrs['loss'] = loss
        partition.attrs['accuracy'] = (labels == logits.argmax(axis=1)).mean()
    
    def build_dataloader(self, data, shuffle=True):
        labels = data['target']
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels.argsort()]
        self.label_map = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long).to(labels.device)
        self.label_map[unique_labels] = torch.arange(len(unique_labels)).to(labels.device)
        self.label_map_inv = unique_labels.to(labels.device)
        
        data['target'] = self.label_map[labels]
        return data
    
class TorchModel(BaseModel, nn.Module):

    def __init__(self, model=None, device='cpu', memory_format=torch.contiguous_format, **kwargs):
        super().__init__()
        self.device = device

        if model is not None:
            self.core = model(**kwargs).to(device, non_blocking=True)
        else:
            self.set_model_parameters(**kwargs)
            self.to(device, non_blocking=True, memory_format=memory_format)

    def set_device(self, device, non_blocking=False):
        self.device = device
        self.to(device, non_blocking=non_blocking)

    def set_model_parameters(self):
        """Sets the model parameters. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement the set_model_parameters method.")

    def forward(self, x, softmax=False, **kwargs):
        if hasattr(self, "core"):
            out = self.core(x, **kwargs)
            if softmax:
                out = F.softmax(out, dim=-1)
            return out
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def get_trainer(self, **kwargs):
        return TorchTrainer(self, **kwargs)

class GPUDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, transform=None, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

        self._n_samples = len(dataset)
        self._n_batches = (self._n_samples + self.batch_size - 1) // self.batch_size

        if self.shuffle:
            self._generator = torch.Generator(device=device)
        else:
            self._idxs = torch.arange(self._n_samples)

    def __len__(self):
        return self._n_batches

    def _shuffle(self):
        self._idxs = torch.randperm(self._n_samples, generator=self._generator)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        for offset in range(0, self._n_samples, self.batch_size):
            if self.transform:
                yield self.transform([self.dataset[i] for i in self._idxs[slice(offset, offset + self.batch_size)]])
            else:
                yield self.dataset[self._idxs[slice(offset, offset + self.batch_size)]]
        # if self.shuffle:
        #     self._shuffle()  # ready for the next epoch
        # raise StopIteration
class DummyScheduler: step = lambda self: None

class TorchTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, lr, weight_decay, batch_size, criterion_kwargs = {}, max_epochs=None, compile=False,mixed_precision=False, scheduler=None, scheduler_kwargs={}, transforms=None):
        super().__init__(model=model, 
                         optimizer=optimizer, 
                         criterion=criterion, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         batch_size=batch_size, 
                         max_epochs=max_epochs, 
                         mixed_precision=False,
                         scheduler=scheduler, 
                         transforms=transforms)

        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = criterion(**criterion_kwargs)

        self.batch_size = batch_size
        self.max_epochs = max_epochs        
        self.mixed_precision = mixed_precision

        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer, **scheduler_kwargs)
        else:            
            self.scheduler = DummyScheduler()

        self.epochs_trained = 0

        self.label_map = None

    def extract_inputs(self, data):
        """Extracts inputs and labels from the data."""
        *inputs, labels = [_.to(self.model.device) for _ in data]
        return inputs, labels
    
    def extract_outputs(self, inputs, softmax=False):
        """Extracts outputs from the model given the inputs."""
        return self.model(*inputs, softmax=softmax)
    
    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = self.extract_inputs(data)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.model.device, dtype=torch.bfloat16, enabled=self.mixed_precision):
                outputs = self.extract_outputs(inputs)
                loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.scheduler.step()
        self.epochs_trained += 1
        train_loss = running_loss / len(train_loader)
        return train_loss
        
    def predict_proba(self, test_loader):
        
        y_true = []
        y_pred = []
        running_loss = 0.0
        self.model.eval()
        for data in test_loader:
            with torch.inference_mode(), autocast(self.model.device, dtype=torch.bfloat16, enabled=self.mixed_precision):
                inputs, labels = self.extract_inputs(data)
                logits = self.extract_outputs(inputs, softmax=True)
            loss = self.criterion(logits, labels)
            running_loss += loss.item()
            y_true.append(labels)
            y_pred.append(logits)
        
        running_loss /= len(test_loader)
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()

        return y_true, y_pred, running_loss
    
    def save_model(self, result_dir):
        model_path = os.path.join(result_dir, f'checkpoint.pth')
        os.makedirs(result_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
    
    def load_model(self, result_dir):
        model_path = os.path.join(result_dir, f'checkpoint.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Results, but no model found in {result_dir}. Unable to resume experiment.")
        self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
        self.model.to(self.model.device)

    def build_dataloader(self, data, shuffle = True):
        *features, labels = data.values()
        unique_labels = labels.unique(sorted=True)
        self.label_map = torch.zeros(unique_labels.max().item() + 1, dtype=torch.long).to(labels.device)
        self.label_map[unique_labels] = torch.arange(len(unique_labels)).to(labels.device)
        self.label_map_inv = unique_labels.to(labels.device)

        return GPUDataLoader(TensorDataset(*features, self.label_map[labels]), batch_size=self.batch_size, shuffle=shuffle, device=labels.device)

class ScikitModel(BaseModel):
    def __init__(self, clf, forward_method='decision_function', device='cpu', **kwargs):
        self.clf = clf(**kwargs)
        self.forward_method = forward_method
        self.device = device
    
    def __call__(self, x):
        """Performs a forward pass through the model."""
        if self.forward_method == 'decision_function':
            logits = self.clf.decision_function(x)
            logits = torch.as_tensor(logits, device=x.device)
            logits = logits.clip(-64, 64)
            if logits.ndim == 1:
                logits = torch.stack([torch.ones_like(logits), logits + 1], dim=-1)
                logits = torch.exp(logits)
            logits = logits / torch.sum(logits, dim=1, keepdims=True)
        elif self.forward_method == 'predict_proba':
            logits = self.clf.predict_proba(x)
            logits = torch.as_tensor(logits, device=x.device)
        else:
            raise ValueError("Invalid forward method.")
        
        return logits
    
    def get_trainer(self, **kwargs):
        return ScikitTrainer(self, **kwargs)
    
class ScikitTrainer(BaseTrainer):
    def __init__(self, model, criterion, n_components=None, transforms=None, **kwargs):
        super().__init__(model=model, n_components=n_components, transforms=transforms)
        
        self.model = model

        self.criterion = criterion()

        self.n_components = n_components
        self.transforms = transforms

        self.train_loss_history = []
        self.val_loss_history = []

    def validate(self, val_data):
        inputs, labels = val_data.values()

        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)

        with config_context(array_api_dispatch=True) if SKLEARN_GPU_ENABLED else nullcontext():
            inputs = inputs if SKLEARN_GPU_ENABLED else inputs.cpu().numpy()
            logits = self.model(inputs)
            
        loss = self.criterion(logits, labels).item()
        return loss
    
    def train(self, train_data, val_data=None):
        inputs, labels = train_data.values()
        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)
        
        # print(inputs.device, labels.device)
        inputs = inputs if SKLEARN_GPU_ENABLED else inputs.cpu().numpy()
        labels = labels if SKLEARN_GPU_ENABLED else labels.cpu().numpy()

        # print(inputs.device, labels.device)

        with config_context(array_api_dispatch=True) if SKLEARN_GPU_ENABLED else nullcontext():
            self.model.clf.fit(inputs+1e-8, labels)

        train_loss = self.validate(train_data)
        self.train_loss_history = np.array([train_loss])

        if val_data is not None:
            val_loss = self.validate(val_data)
            self.val_loss_history = np.array([val_loss])
            return self.train_loss_history, self.val_loss_history

        return self.train_loss_history

    def predict_proba(self, test_data):
        inputs, labels = test_data.values()
        if self.n_components:
            inputs = inputs(self.n_components)
        else:
            inputs = inputs.reshape(inputs.size(0), -1)

        inputs = inputs if SKLEARN_GPU_ENABLED else inputs.cpu().numpy()
        with config_context(array_api_dispatch=True) if SKLEARN_GPU_ENABLED else nullcontext():
            logits = self.model(inputs)
            # print(logits.shape, len(labels.unique()), torch.min(labels.unique()), torch.max(labels.unique()))
        loss = self.criterion(logits, labels).item()

        return labels.cpu().numpy(), logits.cpu().numpy(), loss
    

class TorchGeometricModel(TorchModel):
    def forward(self, data, softmax=False, **kwargs):
        if hasattr(self, "core"):
            out = self.core(data, **kwargs)
            if softmax:
                out = F.softmax(out, dim=-1)
            return out
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    def get_trainer(self, **kwargs):
        return TorchGeometricTrainer(self, **kwargs)
    
class TorchGeometricTrainer(TorchTrainer):
    def __init__(self, model, optimizer, criterion, lr, weight_decay, batch_size, max_epochs, scheduler=None, scheduler_kwargs={}):
        super().__init__(model=model, 
                         optimizer=optimizer, 
                         criterion=criterion, 
                         lr=lr, 
                         weight_decay=weight_decay, 
                         batch_size=batch_size, 
                         max_epochs=max_epochs,
                         scheduler=scheduler,
                         scheduler_kwargs=scheduler_kwargs)
        
    def extract_inputs(self, data):
        # data = data.to(self.model.device)
        return [data], data.y

    def build_dataloader(self, data, shuffle=True):
        from torch_geometric.data import Data, Batch
        # from torch_geometric.loader import DataLoader as GeometricDataLoader
        EEG, edge_index, labels = data.values()
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels.argsort()]
        self.label_map = torch.zeros(unique_labels.max() + 1, dtype=torch.long).to(labels.device)
        self.label_map[unique_labels] = torch.arange(len(unique_labels)).to(labels.device)
        self.label_map_inv = unique_labels.to(labels.device)
        data = [Data(x=EEG[i], edge_index=edge_index[i], y=self.label_map[labels[i]]) for i in range(len(EEG))]
        return GPUDataLoader(data, batch_size=self.batch_size, transform = Batch.from_data_list, shuffle=shuffle, device=labels.device)        

if __name__ == '__main__':
    from src.experiments.experiment_loader import load_experiment_cfg
    from src.experiments.utils import get_data_manager

    cfg = load_experiment_cfg("CVPR2021_object_decoding")
    subject = "S1"
    model = "ADCNN"
    fold_idx = 0
    classes = [0, 11]
    batch_size = 128
    device = 'cuda:0'

    with torch.device(device):
        data_manager = get_data_manager(cfg, model, subject, classes, fold_idx, device=device, non_blocking=False)
        train_data = data_manager.get_data_partition('train')
        print(train_data.keys())
        train_dataset = TensorDataset(*train_data.values())
        data_loader = GPUDataLoader(train_dataset, batch_size=batch_size, shuffle=True, device=device)

        print(data_loader._n_samples)
        for inputs, labels in data_loader:
            print(inputs.shape, inputs.device)
            print(labels.shape, labels.device, labels.unique(return_counts=True))

        print("Rerunning...")
        for inputs, labels in data_loader:
            print(inputs.shape, inputs.device)
            print(labels.shape, labels.device, labels.unique(return_counts=True))

