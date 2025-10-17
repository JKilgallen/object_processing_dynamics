import os
import torch
from src.data.feature_engineering.utils import load_feature

def save_cv_folds(folds, scheme_id, subject, dataset):
    folds_path = get_cv_folds_path(scheme_id, subject, dataset=dataset)
    os.makedirs(os.path.dirname(folds_path), exist_ok=True)
    torch.save(folds, folds_path)

def load_cv_folds(scheme_id, subject, dataset_cfg, device = 'cpu'):
    folds_path = get_cv_folds_path(scheme_id, subject, dataset_cfg['name'])
    folds = torch.load(folds_path, map_location=device, weights_only = False)
    return folds

def get_cv_folds_path(scheme_id, subject, dataset):
    path = ['data',
            dataset,
            'cross_validation',
            subject,
            'task',
            scheme_id, 
            f'folds.pth']
    
    return os.path.join(*path)

def process_cv_folds(cross_val_method, n_folds, n_nested_folds, subject, dataset_cfg, scheme_id, requires={}, trial_subset=slice(None)):
    if not os.path.exists(get_cv_folds_path(scheme_id, subject, dataset_cfg['name'])):
        requires = {k: load_feature(v, subject, dataset_cfg=dataset_cfg, trial_subset=trial_subset) for k, v in requires.items()}
        folds = cross_val_method(**requires, n_folds=n_folds, n_nested_folds=n_nested_folds)
        save_cv_folds(folds, scheme_id, subject, dataset=dataset_cfg['name'])
        return folds
    