from torch import multiprocessing as mp
from torch.cuda import device_count
import argparse
from src.data import download_raw_dataset, process_dataset
from src.experiments.run_experiment import run
from src.experiments.experiment_loader import load_experiment_cfg

def main(experiment_ids, devices = ['cpu'], max_active_jobs=None, max_preloaded_jobs=None, skip_models=[], skip_download=False, skip_processing=False, skip_recheck=False, reset_progress=False, skip_analysis=False, restore=False):
    for exp_id in experiment_ids:
        exp_cfg = load_experiment_cfg(exp_id)
        dataset_cfg = exp_cfg['dataset']
        if not skip_download:
            download_raw_dataset(dataset_cfg=dataset_cfg)
        if not skip_processing:
            process_dataset(dataset_cfg=dataset_cfg)
        print(exp_id)
        run(exp_id=exp_id, 
            devices=devices, 
            max_active_jobs=max_active_jobs, 
            max_preloaded_jobs=max_preloaded_jobs, 
            skip_models=skip_models, 
            skip_recheck=skip_recheck, 
            reset_progress=reset_progress, 
            skip_analysis=skip_analysis, 
            restore=restore)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--set-max-active-jobs', default=None, type=int)
    parser.add_argument('--set-max-preloaded-jobs', default=None, type=int)
    parser.add_argument('--set-devices', nargs='*', default = [f'cuda:{i}' for i in range(device_count())])
    parser.add_argument('--skip-download', action = 'store_true')
    parser.add_argument('--skip-processing', action = 'store_true')
    parser.add_argument('--skip-models', nargs='+', default=[])
    parser.add_argument('--skip-recheck', action = 'store_true')
    parser.add_argument('--skip-analysis', action = 'store_true')
    parser.add_argument('--restore', action = 'store_true')
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    main(
        experiment_ids = args.experiments, 
        devices = args.set_devices, 
        max_active_jobs = args.set_max_active_jobs, 
        max_preloaded_jobs = args.set_max_preloaded_jobs, 
        skip_models = args.skip_models, 
        skip_download = args.skip_download, 
        skip_processing = args.skip_processing, 
        skip_recheck = args.skip_recheck, 
        skip_analysis = args.skip_analysis, 
        restore = args.restore)
