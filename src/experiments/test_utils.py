import os
from hashlib import sha256
from base64 import b32encode
import numpy as np
import pandas as pd
from h5py import File, ExternalLink
from tqdm import tqdm
import torch
import pyarrow
from pyarrow import Table, parquet
# from fastparquet import write as write_table
from itertools import chain
import pickle as pkl
import multiprocessing as mp
from multiprocessing import Pool
from src.data.data_manager import DataManager
from src.data.feature_engineering.utils import load_feature
from src.model_selection import load_cv_folds
from src.experiments.search import build_grid
from src.experiments.experiment_loader import load_experiment_cfg

def get_result_dir(experiment, subject, fold_idx, nested_fold_idx, model_id, hyperparameters):
    params = hyperparameters.copy()
    params.pop('max_epochs', None)
    params_key = generate_hash(params)
    path = os.path.join('outputs', experiment, subject, f'fold_{fold_idx}', 'nested_folds', f'nested_fold_{nested_fold_idx}', model_id, params_key)
    return path

def epochs_completed(result_dir):
    if not os.path.isdir(result_dir):
        return 0
    
    with File(os.path.join(result_dir, 'data.hdf5'), 'a') as f:
        epoch = 0
        while True:
            if f'epoch/{epoch + 1}' not in f:
                return epoch
            epoch += 1

def get_data_manager(cfg, model_id, subject, fold_idx, nested_fold_idx, device):
    features = cfg['models'][model_id]['features']
    target = cfg['scheme']['target']

    features = {f: load_feature(subject, f, device) for f in features}
    features['target'] = load_feature(subject, target, device)
    folds = load_cv_folds(subject, target, device)
    
    data_manager = DataManager(features, folds)
    data_manager.set_fold_idx(fold_idx, nested_fold_idx=nested_fold_idx)
    return data_manager

def apply_transforms(data, transforms, fit=False):
    if fit:
        out = {k: transforms[k].fit_transform(v) if k in transforms else v for k, v in data.items()}
    else:
        out = {k: transforms[k].transform(v) if k in transforms else v for k, v in data.items()}

    return out

def get_exp_key(exp_id, subject_id, fold_idx, nested_fold_idx, model_id, hyperparameters):
    key = b32encode(sha256(str(hyperparameters).encode()).digest()).decode('utf-8').rstrip('=')
    return '/'.join([exp_id, subject_id, f'fold_{fold_idx}', f'nested_fold_{nested_fold_idx}', model_id, key])

def update_accuracy(exp_id, model_id, subject, fold_idx, nested_fold_idx, params):
    max_epochs = params.pop('max_epochs', None)
    path = os.path.join(get_result_dir(exp_id, subject, fold_idx, nested_fold_idx, model_id, params), 'data.hdf5')
    with File(path, 'a') as f:
        for epoch in f['epoch']:
            if max_epochs:
                params['epoch'] = epoch
            epoch_data = f[f'epoch/{epoch}']
            epoch_data.attrs['hyperparameters'] = str(params)
            for partition_id in epoch_data:
                partition_data = epoch_data[partition_id]
                predicted = np.argmax(partition_data['logits'][:], axis=-1)
                partition_data.attrs['accuracy'] = (predicted == partition_data['labels'][:]).mean()

def extract_results(job, index_path):
    exp_id, model_id, subject, fold_idx, nested_fold_idx, params, exp_key = pkl.loads(job.tobytes())
    with File(index_path, 'r') as index:
        results = []
        for epoch in index[exp_key]:
            params['epoch'] = epoch
            for partition_id in index[exp_key][epoch]:
                partition_data = index[exp_key][epoch][partition_id]
                try:
                    loss = partition_data.attrs['loss']
                    labels = partition_data['labels'][:]
                    predicted = np.argmax(partition_data['logits'][:], axis=-1)
                except:
                    print(exp_id, subject, fold_idx, nested_fold_idx, model_id, exp_key, partition_id)
                
                accuracy = (predicted == labels).mean()
                confusion_matrix = np.bincount((labels*6 + predicted), minlength=6**2).reshape((6,6))
                r = {'experiment': exp_id, 
                     'model_id': model_id, 
                     'subject': subject, 
                     'fold': fold_idx, 
                     'nested_fold': nested_fold_idx, 
                     'hyperparameters': str(params), 
                     'partition': partition_id, 
                     'loss': loss,
                     'accuracy': accuracy
                     }
                # logits = partition_data['logits'][:]
                r.update({f'confusion_{i}_{j}': confusion_matrix[i, j] for i in range(6) for j in range(6)})
                results.append(r)
    return results

def process_result(job, index_path):
    exp_id, model_id, subject, fold_idx, nested_fold_idx, params, exp_key = pkl.loads(job.tobytes())
    with File(index_path, 'r') as index:
        results = []
        for epoch in index[exp_key]:
            params['epoch'] = epoch
            for partition_id in index[exp_key][epoch]:
                partition_data = index[exp_key][epoch][partition_id]
                try:
                    loss = partition_data.attrs['loss']
                    labels = partition_data['labels'][:]
                    logits = partition_data['logits'][:]
                    logits = {f'logits_{i}': logits_i for i, logits_i in enumerate(logits.T)}
                except:
                    raise ValueError(f'Unable to extract data for {exp_id}, {subject}, {fold_idx}, {nested_fold_idx}, {model_id}, {exp_key}, {partition_id}')
                
                results.append({
                    'experiment': exp_id,
                    'model_id': model_id,
                    'subject': subject,
                    'fold': fold_idx,
                    'nested_fold': nested_fold_idx,
                    'param_key': params.keys(),
                    'param_value': [str(v) for v in params.values()],
                    'partition': partition_id,
                    'loss': loss,
                    'labels': labels,
                    **logits
                    })
        return results

def build_result_dataset(experiments, batch_size=1024, chunksize =64): 

    schema = pyarrow.schema([
    ('experiment', pyarrow.string()),
    ('model_id', pyarrow.string()),
    ('subject', pyarrow.string()),
    ('fold', pyarrow.int32()),
    ('nested_fold', pyarrow.int32()),
    ('param_key', pyarrow.string()),
    ('param_value', pyarrow.string()),
    ('partition', pyarrow.string()),
    ('loss', pyarrow.float32()),
    ('labels', pyarrow.list_(pyarrow.int32()))] + 
    [(f'logits_{i}', pyarrow.list_(pyarrow.float32())) for i in range(6)])
    
    pool = mp.Queue()
    for exp_id in experiments:
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(process_result, iter_experiments(exp_id, 'completed_experiments'), chunksize=chunksize)):
                results.extend(result)
                if i % batch_size == 0:
                    dataset_path = os.path.join('outputs', exp_id, f'part_{i // batch_size}')
                    table = Table.from_pylist(results, schema=schema)
                    parquet.pq.write_to_dataset(
                        table,
                        root_path=dataset_path,
                        partition_cols=['labels'] + [f'logits_{j}' for j in range(6)], 
    )

def initialize_index(cfg, index_path):
    gpu_jobs, cpu_jobs = [], []
    exp_id = cfg['exp_id']
    for model_id in cfg['models']:
        cpu_only = cfg['models'][model_id].get('cpu_only', False)
        search_space = cfg['models'][model_id]['search_space'].copy()
        parameter_grid = build_grid(search_space)
        for params in parameter_grid:
            for subject in cfg['dataset']['subjects']:
                for fold_idx in range(cfg['scheme']['n_folds']):
                    for nested_fold_idx in range(cfg['scheme']['n_nested_folds']):
                            results_dir = get_result_dir(exp_id, subject, fold_idx, nested_fold_idx, model_id, params)
                            job = np.void(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, params.copy(), results_dir]))
                            if cpu_only:
                                cpu_jobs.append(job)
                            else:
                                gpu_jobs.append(job)
    with File(index_path, 'a') as index:
        manifest = index.create_group('manifest')
        manifest.create_dataset('gpu_experiments', data = np.array(gpu_jobs, dtype='V1024'))
        manifest.create_dataset('cpu_experiments', data = np.array(cpu_jobs, dtype='V1024'))

def update_index(index_path):
    with File(index_path, 'a') as index:
        completed_experiments = []
        for exp_type in ['gpu_experiments', 'cpu_experiments']:
            incomplete_experiments = []
            manifest = index['manifest']
            for job in manifest[exp_type][:]:
                exp_id, model_id, subject, fold_idx, nested_fold_idx, params, results_dir = pkl.loads(job.tobytes())
                exp_key = get_exp_key(exp_id, subject, fold_idx, nested_fold_idx, model_id, params)
                if exp_key in index:
                    completed_experiments.append(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, params, exp_key]))
                    continue
                if epochs_completed(results_dir) >= params.get('max_epochs', 1):
                    result_path = os.path.join(results_dir, 'data.hdf5')
                    index[exp_key] = ExternalLink(result_path, 'epoch')
                    completed_experiments.append(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, params, exp_key]))
                else:
                    incomplete_experiments.append(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, params, results_dir]))
            del manifest[exp_type]
            manifest.create_dataset(exp_type, data = np.array(incomplete_experiments, dtype='V1024'))
        if 'completed_experiments' in index['manifest']:
            del manifest['completed_experiments']
        manifest.create_dataset('completed_experiments', data=np.array(completed_experiments, dtype='V1024'))

def generate_hash(cfg):
    return b32encode(sha256(str(cfg).encode()).digest()).decode('utf-8').rstrip('=')

def generate_experiment_manifests(experiments):
    manifests = []
    for exp_id in experiments:
        cfg = load_experiment_cfg(exp_id)
        exp_key = generate_hash(cfg)
        exp_dir = os.path.join('outputs', exp_id, exp_key)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        index_path = os.path.join(exp_dir, 'index.hdf5')
        if not os.path.exists(index_path):
            initialize_index(cfg, index_path)
        update_index(index_path)
        manifests.append(index_path)
    return manifests

def iter_experiments(manifests, exp_type):
    for manifest_path in manifests:
        with File(manifest_path, 'r') as m: 
            for job in m['manifest'][exp_type][:]:
                exp_id, model_id, subject, fold_idx, nested_fold_idx, params, loc = pkl.loads(job.tobytes())
                yield exp_id, model_id, subject, fold_idx, nested_fold_idx, params, loc


if __name__ == '__main__':
    manifests = generate_experiment_manifests(['category_decoding'])
    n_exp = 0
    for exp in iter_experiments(manifests, 'gpu_experiments'):
        n_exp += 1
    