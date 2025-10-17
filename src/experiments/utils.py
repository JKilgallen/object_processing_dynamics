import os
import itertools
from hashlib import sha256
from base64 import b32encode
import numpy as np
import pandas as pd
from h5py import File, ExternalLink
from tqdm import tqdm
import pyarrow
from pyarrow import Table, parquet
import pickle as pkl
from atexit import register
import yaml
import multiprocessing as mp
from contextlib import nullcontext
from src.data.data_manager import DataManager
from src.data.feature_engineering.utils import load_feature
from src.model_selection import load_cv_folds, process_cv_folds
from src.experiments.experiment_loader import load_experiment_cfg, load_dataset_cfg

def get_result_dir(experiment, subject, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, hyperparameters):
    params = hyperparameters.copy()
    params.pop('max_epochs', None)
    params_key = generate_hash(params)

    if spatial_subset == slice(None):
        spatial_subset = "all_electrodes"
    else:
        spatial_subset = f"electrode_{spatial_subset}"

    if temporal_subset == slice(None):
        temporal_subset = "all_timepoints"
    else:
        temporal_subset = f"timepoints_{temporal_subset.start}-{temporal_subset.stop}"
    if nested_fold_idx is None:
        path = os.path.join('outputs', experiment, spatial_subset, temporal_subset, subject, f'fold_{fold_idx}', model_id, params_key)
    else:
        path = os.path.join('outputs', experiment, spatial_subset, temporal_subset, subject, f'fold_{fold_idx}',  f'nested_fold_{nested_fold_idx}', model_id, params_key)
    
    return path

def epochs_completed(result_dir, max_epochs=1000):
    path = os.path.join(result_dir, 'data.hdf5')
    if not os.path.exists(path):
        return 0
    
    try:
        with File(path, 'r') as f:
            if f'epoch/{max_epochs}' in f:
                return max_epochs
            epoch = 0
            while True:
                if f'epoch/{epoch + 1}' not in f:
                    return epoch
                epoch += 1
    except:
        with open('corrupted.txt', 'a') as f:
            f.writelines(path)
        print(f'Unable to read epochs from {path}. Assuming 0 epochs completed.')
        raise ValueError(f'Unable to read epochs from {path}. Assuming 0 epochs completed.')

def get_data_manager(cfg, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, device='cpu', non_blocking=False):
    features = cfg['models'][model_id]['features']
    scheme_id = cfg['scheme']['scheme_id']
    target = cfg['scheme']['target']
    dataset_cfg = cfg['dataset']
    
    features = {f: load_feature(f, subject, trial_subset=trial_subset, spatial_subset=spatial_subset, temporal_subset=temporal_subset, dataset_cfg=dataset_cfg, device=device, non_blocking=non_blocking) for f in features}
    features['target'] = load_feature(target, subject, trial_subset=trial_subset, dataset_cfg=dataset_cfg, device=device, non_blocking=non_blocking)
    if nested_fold_idx is None:
        nested_fold_idx = -1
    fold = load_cv_folds(scheme_id, subject, dataset_cfg=dataset_cfg, device=device)[fold_idx][nested_fold_idx]
    data_manager = DataManager(features, fold)
    return data_manager

def apply_transforms(data, transforms, fit=False):
    if fit:
        out = {k: transforms[k].fit_transform(v) if k in transforms else v for k, v in data.items()}
    else:
        out = {k: transforms[k].transform(v) if k in transforms else v for k, v in data.items()}

    return out

def build_grid(search_space):
    atoms = [k for k, v in search_space.items() if type(v) != list]
    atoms = {k: search_space.pop(k) for k in atoms}
    grid = [dict(zip(search_space.keys(), c)) for c in itertools.product(*search_space.values())]
    for params in grid:
        params.update(atoms)
    return grid

def get_exp_key(exp_id, subject_id, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, hyperparameters):
    params = hyperparameters.copy()
    params.pop('max_epochs', None)

    key = b32encode(sha256(str(params).encode()).digest()).decode('utf-8').rstrip('=')
    if spatial_subset == slice(None):
        spatial_subset = "all_electrodes"
    else:
        spatial_subset = f"electrode_{spatial_subset}"

    if temporal_subset == slice(None):
        temporal_subset = "all_timepoints"
    else:
        temporal_subset = f"timepoints_{temporal_subset.start, temporal_subset.stop}"

    if nested_fold_idx is None:
        return '/'.join([exp_id, spatial_subset, temporal_subset, subject_id, f'fold_{fold_idx}', model_id, key])
    else:
        return '/'.join([exp_id, spatial_subset, temporal_subset, subject_id, f'fold_{fold_idx}', f'nested_fold_{nested_fold_idx}', model_id, key])

def process_result(job):
    exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, params, exp_key = job
    results = []
    max_epochs = params.pop('max_epochs', 1)
    try:    
        spatial_subset = "all_electrodes" if spatial_subset == slice(None) else f"electrode_{spatial_subset}"
        temporal_subset = "all_timepoints" if temporal_subset == slice(None) else f"timepoints_{temporal_subset.start}-{temporal_subset.stop}"
        for epoch in range(1, max_epochs + 1):
            p = params.copy()
            p['epoch'] = epoch
            for partition_id, partition_data in index[f'{exp_key}/epoch/{epoch}'].items():
                loss = partition_data.attrs['loss']
                accuracy = partition_data.attrs['accuracy']
                res = {
                    'experiment': exp_id,
                    'model_id': model_id,
                    'subject': subject,
                    'fold': fold_idx,
                    'nested_fold': nested_fold_idx,
                    'spatial_subset': spatial_subset, 
                    'temporal_subset': temporal_subset,
                    'hyperparameters': str(p),
                    'partition': partition_id,
                }

                export = 'test' in partition_id
                if export_accuracy:
                    res['accuracy']= accuracy
                if export_loss:
                    res['loss']= loss
                if export_logits:
                    res['logits'] = partition_data['logits'][:] if export else []
                if export_predicted:
                    res['predicted'] = np.argmax(res.get('logits', partition_data['logits'][:]), axis=1) if export else []
                if export_labels:
                    res['labels'] = partition_data['labels'][:] if export else []

                results.append(res)
    except:
        if os.path.exists('corrupted.txt'):
            with open("corrupted.txt", 'a') as f:
                f.write(f"{exp_key}\n")
                
    return results

def get_schema(accuracy=True, loss=True, logits=False, predicted=False, labels=False):
    schema = [
        ('experiment', pyarrow.string()),
        ('model_id', pyarrow.string()),
        ('subject', pyarrow.string()),
        ('fold', pyarrow.int32()),
        ('nested_fold', pyarrow.int32()),
        ('spatial_subset', pyarrow.string()),
        ('temporal_subset', pyarrow.string()),
        ('hyperparameters', pyarrow.string()),
        ('partition', pyarrow.string()),]
    
    if accuracy:
        schema.append(('accuracy', pyarrow.float32()))
    if loss:
        schema.append(('loss', pyarrow.float32()))
    if predicted:
        schema.append(('predicted', pyarrow.list_(pyarrow.int32())))
    if logits:
        schema.append(('logits', pyarrow.list_(pyarrow.list_(pyarrow.float32()))))
    if labels:
        schema.append(('labels', pyarrow.list_(pyarrow.int32())))

    return pyarrow.schema(schema)

def build_result_dataset(manifest_path, chunksize = 1, filter_by_fold=None):
    
    update_index(manifest_path, filter_by_fold=filter_by_fold)
    output_dir =os.path.join(os.path.dirname(manifest_path), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    with File(manifest_path, 'r') as index:
        data = index['manifest']
        exports = {key: data['exports'].attrs.get(key, False) for key in ['accuracy', 'loss', 'logits', 'predicted', 'labels']}
        schema = get_schema(**exports)
        
        n_folds = data.attrs['n_folds']
        for fold_idx in range(n_folds):
            if (filter_by_fold is not None) and (fold_idx != filter_by_fold):
                continue
            path = os.path.join(output_dir, f'fold_{fold_idx}_results.parquet')
            n_jobs = len(data['completed_experiments'][f'fold_{fold_idx}'])
            if not os.path.exists(path):
                with mp.Pool(processes=mp.cpu_count(), initializer=init_fetcher, initargs=(manifest_path, *exports.values())) as pool:   
                    results = []
                    for result in tqdm(pool.imap_unordered(process_result, iter_experiments(index, 'completed_experiments', filter_by_fold=fold_idx), chunksize=chunksize), desc='Writing results to dataset', total=n_jobs):
                        results.extend(result)
                results = Table.from_pylist(results, schema=schema)
                parquet.write_table(results, path)
                
    print(f"Finished writing results to file.")
    

def init_fetcher(index_path, accuracy=True, loss=True, logits=False, predicted=False, labels=True):
    global index, export_accuracy, export_loss, export_logits, export_predicted, export_labels
    index = File(index_path, 'r')
    export_accuracy = accuracy
    export_loss = loss
    export_logits = logits
    export_predicted=predicted
    export_labels=labels
    register(index.close) 

def iter_config(cfg, filter_by_fold=None):
    exp_id = cfg['exp_id']
    dataset_cfg = cfg['dataset']
    scheme_cfg = cfg['scheme']

    scheme_id = scheme_cfg['scheme_id']
    cross_val_method = scheme_cfg['cross_val']['method']
    cross_val_kwargs = scheme_cfg['cross_val']['kwargs']
    n_folds = scheme_cfg['cross_val']['n_folds']
    n_nested_folds = scheme_cfg['cross_val']['n_nested_folds']
    trial_subset = scheme_cfg.get('trial_subset', slice(None))
    spatial_subsets = scheme_cfg.get('spatial_subsets', [slice(None)])
    temporal_subsets = scheme_cfg.get('temporal_subsets', [slice(None)])

    for subject in cfg['dataset']['subjects']:
        process_cv_folds(cross_val_method, subject=subject, dataset_cfg=dataset_cfg, scheme_id=scheme_id, trial_subset=trial_subset, **cross_val_kwargs) 
        for model_id in cfg['models']:
            param_grid = build_grid(cfg['models'][model_id]['search_space'].copy())
            transforms = cfg['models'][model_id].get('transforms', {})
            for fold_idx in range(n_folds):
                if (filter_by_fold is not None) and (filter_by_fold != fold_idx):
                    continue
                for nested_fold_idx in list(range(n_nested_folds)) + [None]:
                    for spatial_subset in spatial_subsets:
                        for temporal_subset in temporal_subsets: 
                            job = np.void(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid]))
                            # print(exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid)
                            yield job

def iter_update(index, filter_by_fold=None):
    for job in iter_experiments(index, 'incomplete_experiments', filter_by_fold=filter_by_fold):
        exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid = job
        remaining_params = []
        for params in param_grid:
            try:
                p = params.copy()
                p['transforms'] = transforms
                exp_key = get_exp_key(exp_id, subject, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, p)
                if exp_key in index:
                    yield np.void(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, p, exp_key])), True
                else:
                    remaining_params.append(params)
            except:
                remaining_params.append(params)
        if len(remaining_params) != 0:
            yield np.void(pkl.dumps([exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, remaining_params])), False

def get_n_incomplete_jobs(index, filter_by_fold=None):
    n_jobs = 0
    for job in iter_experiments(index, 'incomplete_experiments', filter_by_fold=filter_by_fold):
        *_, param_grid = job
        n_jobs += len(param_grid)
    return n_jobs

def get_n_complete_jobs(index, filter_by_fold=None):
    n_jobs = 0
    for job in iter_experiments(index, 'completed_experiments', filter_by_fold=filter_by_fold):
        n_jobs += 1
    return n_jobs

def get_progress(index, filter_by_fold=None):
    n_complete = get_n_complete_jobs(index, filter_by_fold=filter_by_fold)
    n_incomplete = get_n_incomplete_jobs(index, filter_by_fold=filter_by_fold)
    return n_complete, n_complete + n_incomplete

def update_index(index_path, filter_by_fold=None):
    with File(index_path, 'a') as index:
        data = index[f'manifest']
        n_complete, n_incomplete = 0, 0
        for fold_idx in range(data.attrs['n_folds']):
            if (filter_by_fold is not None) and (filter_by_fold != fold_idx):
                continue
            jobs, mask = np.fromiter(tqdm(iter_update(index, filter_by_fold=filter_by_fold), desc=f"Updating completed experiments"), dtype=np.dtype((object, 2))).T
            jobs, mask = jobs.astype('V2048'), mask.astype('?')
            incomplete_experiments = jobs[~mask]
            completed_experiments = jobs[mask]

            data['incomplete_experiments'][f'fold_{fold_idx}'].resize(incomplete_experiments.shape)
            data['incomplete_experiments'][f'fold_{fold_idx}'][:] = incomplete_experiments
        
            if not 'completed_experiments' in data:    
                data.create_group('completed_experiments')
                
            if not f'fold_{fold_idx}' in data['completed_experiments']:
                data['completed_experiments'].create_dataset(f'fold_{fold_idx}', data=completed_experiments, maxshape=(None,), chunks=True)
            else:
                completed_experiments = np.concatenate([completed_experiments, data['completed_experiments'][f'fold_{fold_idx}'][:]])
                data['completed_experiments'][f'fold_{fold_idx}'].resize(completed_experiments.shape)
                data['completed_experiments'][f'fold_{fold_idx}'][:] = completed_experiments
            n_incomplete += get_n_incomplete_jobs(index, filter_by_fold=filter_by_fold)
            n_complete += len(completed_experiments)
    print(f'{n_complete}/{n_complete+n_incomplete} experiments have been completed.')
    return n_complete, n_incomplete + n_complete


def generate_hash(cfg):
    return b32encode(sha256(str(cfg).encode()).digest()).decode('utf-8').rstrip('=')

def restore_corrupted_index(exp_id, model_id = None, fold_idx=None):
    index_path = get_experiment_manifest(exp_id=exp_id, model_id=model_id)
    backup_path = f"{index_path}.bak"
    os.rename(index_path, backup_path)

    get_experiment_manifest(exp_id=exp_id, model_id=model_id)
    with File(index_path, 'a') as index, File(backup_path, 'r') as backup:
        for job in iter_experiments(index, 'incomplete_experiments', filter_by_fold=fold_idx):
            restore_experiment(job, index=index, backup=backup)
    
def restore_experiment(job, index, backup):
    exp_id, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid = job
    for params in param_grid:
        p = params.copy()
        p.pop('max_epochs', None)
        p['transforms'] = transforms
        exp_key = get_exp_key(exp_id, subject, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, p)
        
        try:
            backup.copy(backup[exp_key], index, exp_key)
        except:
            pass

def get_experiment_manifest(exp_id, model_id=None):
    safe_cfg = load_experiment_cfg(exp_id, safe=True)
    exp_key = generate_hash(safe_cfg)
    exp_dir = os.path.join('outputs', exp_id, exp_key)
    if model_id is not None:
        safe_cfg['models'] = {model_id: safe_cfg['models'][model_id]}
        exp_dir = os.path.join(exp_dir, 'by_model', model_id)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, f'{exp_id}.yaml'), 'w') as file:
            yaml.safe_dump(safe_cfg, file, default_flow_style=False, sort_keys=False)

    cfg = load_experiment_cfg(exp_id)
    if model_id is not None:
        cfg['models'] = {model_id: cfg['models'][model_id]}
    
    index_path = os.path.join(exp_dir, 'index.hdf5')
    if not os.path.exists(index_path):
        with File(index_path, 'w') as index:
            data = index.create_group('manifest')
            data.attrs['exp_id'] = exp_id
            data.create_dataset('model_ids', data = [model_id for model_id in cfg['models'].keys()])
            data.attrs['dataset'] = cfg['dataset']['name']
            data.attrs['scheme_id'] = cfg['scheme']['scheme_id']
            data.attrs['target'] = cfg['scheme']['target']
            data.attrs['cross_val_method'] = np.void(pkl.dumps(cfg['scheme']['cross_val']['method']))
            data.attrs['cross_val_kwargs'] = str(cfg['scheme']['cross_val']['kwargs'])
            data.attrs['n_folds'] = cfg['scheme']['cross_val']['n_folds']
            data.attrs['n_nested_folds'] = cfg['scheme']['cross_val']['n_nested_folds']
            data.attrs['spatial_subsets'] = str(cfg['scheme'].get('spatial_subsets', [slice(None)]))
            data.attrs['temporal_subsets'] = str(cfg['scheme'].get('temporal_subsets', [slice(None)]))

            incomplete_experiments = data.create_group('incomplete_experiments')
            for fold_idx in range(data.attrs['n_folds']):
                jobs = np.fromiter(iter_config(cfg, filter_by_fold=fold_idx), dtype=np.dtype(object)).astype('V2048')
                incomplete_experiments.create_dataset(f'fold_{fold_idx}', data =jobs, maxshape=(None,), chunks=True)

            exports = data.create_group('exports')
            export_cfg = cfg.get('exports', {'accuracy': True, 'loss': True})
            for key in ['accuracy', 'loss', 'labels', 'logits', 'predicted', 'labels']:
                exports.attrs[key] = export_cfg.get(key, False)

    return index_path

def iter_experiments(manifest, exp_type, filter_by_model=None, filter_by_fold=None, filter_by_window=None, filter_by_electrode=None):
    with File(manifest, 'r') if type(manifest) == str else nullcontext(manifest) as m:
        if f'manifest/{exp_type}' in m:
            data = m[f'manifest/{exp_type}']
            for fold_idx in range(m['manifest'].attrs['n_folds']):
                if (filter_by_fold is not None) and (fold_idx != filter_by_fold):
                    continue
                for job in data[f'fold_{fold_idx}'][:]:
                    yield pkl.loads(job.tobytes())