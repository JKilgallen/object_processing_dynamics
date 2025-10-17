import os
import pandas as pd
from pyarrow.dataset import dataset
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
from tqdm.auto import tqdm
import glob
import re
from matplotlib import pyplot as plt
from src.experiments.experiment_loader import load_experiment_cfg

LABELS = {'category': {0: 'HB', 1: 'HF', 2: 'AB', 3: 'AF', 4: 'FV', 5: 'IO'},
          'human_face_vs_artificial_object': {0: 'HF', 1: 'IO'},
          'exemplar': {idx: f'figures/stimuli/stimulus{idx+1}.png' for idx in range(72)},
          'artificial_object': {idx: f'figures/stimuli/stimulus{idx+61}.png' for idx in range(12)},
          'human_face': {idx: f'figures/stimuli/stimulus{idx+13}.png' for idx in range(12)}}
COLORS = {'category': {0: '#207aba', 1: '#ff7923', 2: '#38a446', 3: '#ea2f1c', 4: '#8566b4', 5: '#bcbe3c'}, 
          'human_face_vs_artificial_object': {0: '#ff7923', 1: '#bcbe3c'}, 
          'exemplar': {idx: ['#207aba','#ff7923','#38a446','#ea2f1c','#8566b4','#bcbe3c'][idx//12] for idx in range(72)},
          'artificial_object': {idx: '#bcbe3c' for idx in range(12)},
          'human_face': {idx: '#ff7923' for idx in range(12)}}

def summarize_results(results_df):
    results_df['accuracy'] *= 100
    results_df['fold'] = results_df['fold'].astype(str)
    results_df = pd.pivot_table(results_df,
                                index=['experiment', 'model_id', 'subject', 'fold', 'hyperparameters', 'spatial_subset', 'temporal_subset'],
                                values=['loss', 'accuracy'],
                                columns=['partition'],
                                aggfunc='mean').reset_index()
    results_df.columns = ['_'.join(reversed(col)).strip('_') for col in results_df.columns.values]
    return results_df

def select_models(results_df, metric='accuracy', stimulus_repetition=False, post_hoc=False):
    summary_df = summarize_results(results_df)    
    opt_partition = f"{'' if (stimulus_repetition and post_hoc) else 'un'}{'confounded_test' if post_hoc else 'val'}_{metric}"
    opt_df = summary_df.groupby(['experiment', 'model_id', 'subject', 'fold', 'hyperparameters']).mean(numeric_only=True)
    opt_df = opt_df.loc[opt_df.groupby(['experiment', 'subject', 'model_id', 'fold'])[opt_partition].idxmax()].reset_index()
    opt_df = results_df.merge(opt_df[['experiment', 'subject', 'model_id', 'fold', 'hyperparameters']], how='inner')
    # opt_df = opt_df.loc['test' in opt_df['partition']]
    return opt_df

def analyze_results(index_path, exp_cfg=None, alpha=0.05):
    result_dir = os.path.dirname(index_path)
    exp_id = result_dir.split('/')[1]
    post_hoc=True; stimulus_repetition=False
    files = glob.glob(f"{result_dir}/**/*.parquet", recursive=True)
    results_df = dataset(files, format="parquet").to_table().to_pandas()
    results_df = select_models(results_df, stimulus_repetition=stimulus_repetition, post_hoc=post_hoc)
    results_df.to_csv(os.path.join(result_dir, 'summary.csv'))

    print(results_df.groupby(['model_id', 'partition']).mean(numeric_only=True))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze experiment results.')
    parser.add_argument('index_path', type=str, help='Path to the index file containing experiment results.')
    args = parser.parse_args()
    analyze_results(args.index_path)