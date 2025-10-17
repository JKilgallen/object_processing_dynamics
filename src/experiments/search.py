
import numpy as np
from h5py import File
import os
from tqdm.auto import tqdm
# from src.experiments.utils import get_sub_result_dir, apply_transforms




# def load_sub_result(subresult_dir):
#     if os.path.exists(os.path.join(subresult_dir, 'data.hdf5')):
#         with File(os.path.join(subresult_dir, 'data.hdf5'), 'r') as f:
#             val_loss_history = f['val_loss_history'][:]
#         return val_loss_history
#     else:
#         return []

# def evaluate(sub_result_dir, model_cfg, hyperparameters, max_epochs, train_data, val_data, device='cpu'):
#     val_losses = load_sub_result(sub_result_dir)
#     if len(val_losses) >= max_epochs:
#         return val_losses[:max_epochs]
    
#     model = model_cfg['class'](**model_cfg['arch_config'], device=device)
#     trainer = model.get_trainer(**model_cfg['trainer_config'], **hyperparameters, max_epochs=max_epochs)
   
#     if len(val_losses) > 0:
#         trainer.load_train_data(sub_result_dir)

#     _, val_losses = trainer.fit(train_data, val_data)
#     trainer.save_train_data(sub_result_dir)

#     return val_losses

# def grid_search(experiment, subject, fold_idx, model_cfg, data_manager, device='cpu'):

#     n_nested_folds = data_manager.n_nested_folds
#     search_space = model_cfg['search_space'].copy()
#     max_epochs = search_space.pop('max_epochs', 1)
#     transforms = search_space.pop('transforms', [{}])

#     grid = build_grid(search_space)
#     scores = np.zeros((n_nested_folds, len(grid), len(transforms), max_epochs))

#     for idx in tqdm(range(n_nested_folds), desc="Evaluating nested folds", unit="fold", leave=False):
#         data_manager.set_fold_idx(nested_fold_idx = idx)
#         train_data = data_manager.get_train_data()
#         val_data = data_manager.get_val_data()
#         for tdx, transform in enumerate(transforms):
#             t = {k: v['method'](**v.get('params', {})) for k, v in transform.items()}
#             train = apply_transforms(train_data, t, fit=True)
#             val = apply_transforms(val_data, t)
#             for hdx, hyperparameters in enumerate(grid):
#                 if 'transforms' in model_cfg['search_space']:
#                     hyperparameters['transforms'] = transform
#                 sub_result_dir = get_sub_result_dir(experiment, subject, fold_idx, idx, model_cfg['id'], hyperparameters)
#                 scores[idx, hdx, tdx, :] = evaluate(sub_result_dir, model_cfg, hyperparameters, max_epochs, train, val, device=device)
        
#     scores = scores.mean(axis=0)
#     opt_pdx, opt_tdx, opt_epochs = np.unravel_index(np.argmin(scores), scores.shape)
#     opt_params = grid[opt_pdx]

#     if 'max_epochs' in model_cfg['search_space']:
#         opt_params['max_epochs'] = int(opt_epochs) + 1

#     if 'transforms' in model_cfg['search_space']:
#         opt_params['transforms'] = transforms[opt_tdx]
        
#     tqdm.write(f"Optimal hyperparameters: {opt_params} with score {scores[opt_pdx, opt_tdx, opt_epochs]}")

#     return opt_params



# def hierarchical_grid_search(experiment, subject, fold_idx, model_cfg, data_manager, device='cpu'):
#     """
#     Perform a hierarchical grid search for hyperparameter optimization.
#     """
#     n_nested_folds = data_manager.n_nested_folds
#     search_space = model_cfg['search_space'].copy()
#     max_epochs = search_space.pop('max_epochs', 1)

#     opt_params = {k: v[0] for k, v in search_space.items()}  

#     for key, values in tqdm(search_space.items(), desc='Selecting optimal hyperparameters', unit='parameter'):
#         scores = np.zeros((n_nested_folds, len(values), max_epochs))
#         for idx in tqdm(range(n_nested_folds), desc="Evaluating nested folds", unit="fold", leave=False):
#             data_manager.set_fold_idx(nested_fold_idx=idx)
#             if key != 'transforms':
#                 transforms = {k: v['method'](**v.get('params', {})) for k, v in opt_params.get('transforms', {}).items()}
#                 train = apply_transforms(data_manager.get_train_data(), transforms, fit=True)
#                 val = apply_transforms(data_manager.get_val_data(), transforms)

#             for vdx, v in enumerate(values):
#                 hyperparameters = opt_params.copy()
#                 hyperparameters[key] = v
#                 if key == 'transforms':
#                     transforms = {k: v['method'](**v.get('params', {})) for k, v in v.items()}
#                     train = apply_transforms(data_manager.get_train_data(), transforms, fit=True)
#                     val = apply_transforms(data_manager.get_val_data(), transforms)
                    
#                 sub_result_dir = get_sub_result_dir(experiment, subject, fold_idx, idx, model_cfg['id'], hyperparameters)
#                 scores[idx, vdx, :] = evaluate(sub_result_dir, model_cfg, hyperparameters, max_epochs, train, val, device=device)
#             del train, val

#         scores = scores.mean(axis=0)
#         opt_vdx, opt_epochs = np.unravel_index(np.argmin(scores), scores.shape)
#         opt_params[key] = values[opt_vdx]

#     if 'max_epochs' in model_cfg['search_space']:
#         opt_params['max_epochs'] = int(opt_epochs) + 1

#     tqdm.write(f"Optimal hyperparameters: {opt_params} with score {scores[opt_vdx, opt_epochs]}")

#     return opt_params