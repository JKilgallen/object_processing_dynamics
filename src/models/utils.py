import torch
import torch.profiler as profiler
from src.models.lda import LDA
from src.models.adcnn import ADCNN
from src.models.aw1dcnn import AW1DCNN
from src.models.eegct import EEGCT
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
import time

def profile_model(model, trainer_cfg, hyperparameters, n_classes, input_shape):    

    trainer = model.get_trainer(trainer_cfg, hyperparameters)
    train_data = [torch.rand(*input_shape).to(model.device), torch.randint(0, n_classes, (input_shape[0],)).to(model.device)]
    val_data = [torch.rand(*input_shape).to(model.device), torch.randint(0, n_classes, (input_shape[0],)).to(model.device)]
    test_data = [torch.rand(*input_shape).to(model.device), torch.randint(0, n_classes, (input_shape[0],)).to(model.device)]

    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        trainer.fit(train_data, val_data)
        torch.cuda.synchronize()
        trainer.predict(test_data)
        torch.cuda.synchronize()
    del train_data, val_data, test_data
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    torch.cuda.empty_cache()
    return prof

def run_model(model, trainer_cfg, hyperparameters, n_classes, input_shape):
    trainer = model.get_trainer(trainer_cfg, hyperparameters)
    data = [torch.rand(*input_shape).to(model.device), torch.randint(0, n_classes, (input_shape[0],)).to(model.device)]
    trainer.fit(data, data)
    trainer.predict(data)
    del data, model, trainer
    torch.cuda.empty_cache()

trainer_cfg = {'optimizer': Adam,
               'criterion': CrossEntropyLoss(),
               'max_epochs': 10,
               'patience': 10,
               'batch_size': 128}

hyperparameters = {'lr': 1e-3,
                   'weight_decay': 1e-3}

# print("ADCNN")
# adcnn = torch.compile(ADCNN(device='cuda'))
# profile_model(adcnn, trainer_cfg, hyperparameters, 6, (1000, 124, 32))

# print("AW1DCNN")
# aw1dcnn = torch.compile(AW1DCNN(6, 124, 512, 50, 100, 5, device='cuda'))
# profile_model(aw1dcnn, trainer_cfg, hyperparameters, 6, (1000, 124, 32))

# print("EEGCT")
# eegct = torch.compile(EEGCT(6, 72, 12, 432, 768, 49, 32, device='cuda'))
# profile_model(eegct, trainer_cfg, hyperparameters, 6, (1000, 32, 32, 32))

def evaluate_model_runtime(model_class, model_args, trainer_cfg, hyperparameters, n_classes, input_shape):

    start_time = time.time()
    model = model_class(**model_args)
    run_model(model, trainer_cfg, hyperparameters, n_classes, input_shape)
    end_time = time.time()
    print(f"Full precision runtime: {end_time - start_time} seconds")

    start_time = time.time()
    model = model_class(**model_args)
    trainer_cfg['mixed_precision'] = True
    run_model(model, trainer_cfg, hyperparameters, n_classes, input_shape)
    end_time = time.time()
    print(f"Mixed precision runtime: {end_time - start_time} seconds")




# Example usage
# evaluate_model_runtime(ADCNN, {'device': 'cuda'}, trainer_cfg, hyperparameters, 6, (1000, 124, 32))
# evaluate_model_runtime(AW1DCNN, {'n_classes': 6, 
#                                  'n_channels': 124, 
#                                  'n_filters': 512, 
#                                  'hidden_dim1': 50, 
#                                  'hidden_dim2': 100, 
#                                  'kernel_size': 5, 
#                                  'device': 'cuda'}
#                                  , trainer_cfg, hyperparameters, 6, (1000, 124, 32))
evaluate_model_runtime(EEGCT, {'n_classes': 6, 
                               'n_features': 72, 
                               'n_heads': 12, 
                               'hidden_dim_1': 432, 
                               'hidden_dim_2': 768, 
                               'n_patches': 49, 
                               'n_time_points': 32, 
                               'device': 'cuda'},
                       trainer_cfg, hyperparameters, 6, (1000, 32, 32, 32))

               