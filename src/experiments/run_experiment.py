# from sklearnex import patch_sklearn
# patch_sklearn()
from torch import multiprocessing as mp, device as default_device
from torch.cuda import empty_cache, current_stream
import numpy as np
from tqdm.auto import tqdm
from h5py import File, ExternalLink
import os
import gc
from src.experiments.experiment_loader import load_experiment_cfg
from src.experiments.utils import get_result_dir, epochs_completed, apply_transforms, get_data_manager, get_experiment_manifest, update_index, iter_experiments, build_result_dataset, get_exp_key, get_progress, restore_corrupted_index

def run_experiment(experiment, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid, device, lock, progress_queue):
    cfg = load_experiment_cfg(experiment)
    model_cfg = cfg['models'][model_id]
    data_manager = get_data_manager(cfg, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, device=device, non_blocking=False)
    
    t = {k: v['method'](**v.get('kwargs', {})) for k, v in  transforms.items()}

    train_data = data_manager.get_data_partition('train')
    train_data = apply_transforms(train_data, t, fit=True)
    
    test_partitions = cfg['scheme']['test_partitions'] if nested_fold_idx is None else cfg['scheme']['val_partitions']
    test_data = {partition:
                    apply_transforms(
                        data_manager.get_data_partition(partition),
                    t, fit=False)
                for partition in test_partitions}
    
    try:
        lock.acquire()
        for params in param_grid:
            model = model_cfg['class'](**model_cfg['arch_config'], device=device)
            trainer = model.get_trainer(**model_cfg['trainer_config'], **params)
            train_loader = trainer.build_dataloader(train_data, shuffle=True)
            test_loaders = {partition: trainer.build_dataloader(data, shuffle=False) for partition, data in test_data.items()}

            p = params.copy()
            p['transforms'] = transforms
            max_epochs = p.pop('max_epochs', 1)
            result_dir = get_result_dir(experiment, subject, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, p)
            
            try:   
                last_epoch = epochs_completed(result_dir, max_epochs=max_epochs)
                if last_epoch > 0: 
                    trainer.load_model(result_dir)
            except Exception as e:
                print(f'Unable to resume experiment due to missing or corrupt model file in {result_dir}. Creating backup and restarting experiment from epoch 0.')
                os.rename(os.path.join(result_dir, 'data.hdf5'), os.path.join(result_dir, 'data.hdf5.bak'))
                last_epoch = 0

            # os.makedirs(result_dir, exist_ok=True)
            # result_path = os.path.join(result_dir, 'data.hdf5')

            result_data = {}
            for epoch in range(last_epoch, max_epochs):
                result_data[epoch + 1] = {}
                if trainer.train(train_loader): 
                    for partition, loader in test_loaders.items():
                        labels, logits, loss = trainer.predict_proba(loader)
                        accuracy = (labels == logits.argmax(axis=1)).mean()
                        result_data[epoch + 1][partition] = {
                            'labels': labels,
                            'logits': logits,
                            'loss': loss,
                            'accuracy': accuracy
                        }
                        if (nested_fold_idx is None) and ((epoch+1) % 10) == 0:
                            tqdm.write(f"Epoch {epoch+1}, {model_id}, {subject}:{fold_idx}:{nested_fold_idx}-{partition}: {accuracy*100:.2f}%, {loss:.4f}.")
                else:
                    tqdm.write(f"Error training model {model_id} with parameters {p} on subject {subject} fold {fold_idx}, spatial subset {spatial_subset}, temporal subset {temporal_subset}.")
            # if max_epochs > 1:
            #     trainer.save_model(result_dir)
            progress_queue.put((experiment, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, p, result_data))
    finally:
        del model, trainer
        gc.collect()
        if device != 'cpu':
            empty_cache()
        lock.release()

def worker(job_queue, progress_queue, device_pool, device_locks, ):
    device = device_pool.get()
    lock = device_locks[device]
    # print(f"{device} was allocated to a process.")
    with default_device(device):
        while True:
            job = job_queue.get()
            if job is None:  # Sentinel value to signal end of jobs
                device_pool.put(device)  # Put the device back in the pool
                return
            
            experiment, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid = job
            run_experiment(experiment, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, transforms, param_grid, device, lock, progress_queue)

def get_lockable_devices(devices, n_active = 1, n_preloaded=1, ctx=None):
    # print("Generating locks to limit concurrent gpu access.")
    manager = ctx.Manager()
    pool = mp.Queue()
    locks = manager.dict({d: manager.BoundedSemaphore(n_active) for d in devices})
    # print(f"The following locks were created which allow at most {n_active} concurrent jobs per device.")
    # print(f"Initializing device pool...")
    for device in devices:
        for _ in range(n_active + n_preloaded):
            pool.put(device)
            # print(f"Successfully added {device} added to the device pool.")
        
    return pool, locks
    
def populate_job_queue(jobs, job_queue, n_procs):
    for job in jobs:
        job_queue.put(job)
    for _ in range(n_procs):
        job_queue.put(None)  # Sentinel value to signal end of jobs
    return job_queue

def stream_to_file(index, job):
    experiment, model_id, subject, fold_idx, nested_fold_idx, trial_subset, spatial_subset, temporal_subset, params, result_data = job
    exp_key = get_exp_key(experiment, subject, fold_idx, nested_fold_idx, spatial_subset, temporal_subset, model_id, params)

    if not f'{exp_key}/epoch' in index:
        index.create_group(f'{exp_key}/epoch')
        index[exp_key].attrs['hyperparameters'] = str(params)
    for epoch, epoch_data in result_data.items():
        if not f'{exp_key}/epoch/{epoch}' in index:
            index.create_group(f'{exp_key}/epoch/{epoch}')
        for partition, partition_data in epoch_data.items():
            if not f'{exp_key}/epoch/{epoch}/{partition}' in index:
                index.create_group(f'{exp_key}/epoch/{epoch}/{partition}')
            index[f'{exp_key}/epoch/{epoch}/{partition}'].attrs['accuracy'] = partition_data['accuracy']
            index[f'{exp_key}/epoch/{epoch}/{partition}'].attrs['loss'] = partition_data['loss']
            if 'test' in partition:
                index[f'{exp_key}/epoch/{epoch}/{partition}'].create_dataset('labels', data=partition_data['labels'])
                index[f'{exp_key}/epoch/{epoch}/{partition}'].create_dataset('logits', data=partition_data['logits'])

def run(exp_id, devices, max_active_jobs=None, max_preloaded_jobs=None, skip_models=[], skip_recheck=False, reset_progress=False, skip_analysis=False, restore=False):
    exp_cfg = load_experiment_cfg(exp_id)
    # primary_index_path = get_experiment_manifest(exp_id)
    models = exp_cfg['models']
    analysis_method = exp_cfg['scheme']['analysis']['method']
    n_folds = exp_cfg['scheme']['cross_val']['n_folds']

    for model_id in models:
        if model_id in skip_models:
            continue
        sub_index_path  = get_experiment_manifest(exp_id, model_id)
        if restore:
            restore_corrupted_index(exp_id, model_id)
            
        for fold_idx in range(n_folds):
            n_active = exp_cfg['models'][model_id].get('n_active_jobs_per_gpu', 1) if max_active_jobs is None else max_active_jobs
            n_preloaded = exp_cfg['models'][model_id].get('n_preloaded_jobs_per_gpu', 0) if max_preloaded_jobs is None else max_preloaded_jobs

            dest = os.path.join(os.path.dirname(sub_index_path), 'results', f'fold_{fold_idx}_results.parquet')
            if os.path.exists(dest):
                continue
            if skip_recheck:
                n_complete, total = get_progress(sub_index_path, filter_by_fold=fold_idx)
            else:
                n_complete, total = update_index(sub_index_path, filter_by_fold=fold_idx)

            if n_complete != total:

                job_queue = mp.Queue()
                progress_queue = mp.Queue()

                ctx = mp.get_context("spawn")
                device_pool, device_locks = get_lockable_devices(devices, n_active=n_active, n_preloaded=n_preloaded, ctx=ctx)
                n_concurrent = len(devices)*(n_active + n_preloaded)
                # print(f"Starting {n_concurrent} processes: {n_active} processes and {n_preloaded} preloaded processes per device.")
                processes = [ctx.Process(target=worker, args=(job_queue, progress_queue, device_pool, device_locks)) for _ in range(n_concurrent)]

                with tqdm(total=total, initial=n_complete, desc="Running experiments", unit="experiment") as pbar:
                    for process in processes:
                        process.start()

                    populate_job_queue(iter_experiments(sub_index_path, 'incomplete_experiments', filter_by_fold=fold_idx), job_queue, n_concurrent)
                    with File(sub_index_path, 'a') as sub_index:
                        for idx in range(n_complete, total):
                            stream_to_file(sub_index, progress_queue.get()); pbar.update(1)

                    for process in processes:
                        process.join()
            build_result_dataset(sub_index_path, filter_by_fold=fold_idx)
        if not skip_analysis:
            analysis_method(sub_index_path, exp_cfg)
    # analysis_method(primary_index_path)
    
    # return primary_index_path
