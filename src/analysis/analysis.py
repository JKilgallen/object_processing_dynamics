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
from src.analysis.visualization import plot_confusion_matrix, plot_embedding, plot_dendrogram, plot_topomap, plot_time_course, plot_temporally_resolved_confusion_matrices, plot_temporally_resolved_embedding, plot_temporally_resolved_dendrograms, plot_temporally_resolved_topomaps

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
TEMPORAL_SUBSETS = ["all_timepoints"] + [f"timepoints_{t.start}-{t.stop}" for t in list(slice(i, i+6) for i in range(0, 27, 3))]
SPATIAL_SUBSETS = ["all_electrodes"] + [f"electrode_{idx}" for idx in range(124)]
OPTIONS = {
    'category': {
        'full_response': {
            'calculate_bias': True,
            'confusion_matrix': {'label_style': 'pretty', 
                                 'colors':COLORS['category'],
                                 'labels': LABELS['category'],
                                 'plot_bias': True},
            'mds': {'marker_style': 'pretty', 
                    'colors':COLORS['category'],
                    'labels': LABELS['category']},
            'dendrogram': {'label_style': 'pretty', 
                           'colors':COLORS['category'],
                           'labels': LABELS['category']}
            },
        'spatial': {
            'calculate_bias': True,
            'topomap': {'plot_bias': True}
        },
        'temporal': {
            'calculate_bias': True,
            'time_course': {'colors': COLORS['category'],
                            'labels': LABELS['category'],
                            'legend': np.arange(0, 6),
                            'plot_bias': True},
            'confusion_matrix':  {'label_style': 'pretty', 
                                 'colors':COLORS['category'],
                                 'labels': LABELS['category'],
                                 'plot_bias': True},
            'mds': {'marker_style': 'pretty', 
                    'colors':COLORS['category'],
                    'labels': LABELS['category'],},
            'dendrogram':  {'label_style': 'pretty', 
                           'colors':COLORS['category'],
                           'labels': LABELS['category']}
        },
        'spatio_temporal': {
            'calculate_bias': True,
            'topomap': {'plot_bias': True}
        },
    },
    'exemplar': {
        'full_response': {
            'confusion_matrix': {'subtract_diagonal': True,
                                 'label_style': 'image', 
                                 'colors':COLORS['exemplar'],
                                 'labels': LABELS['exemplar'],
                                 'stagger': True,
                                 'zoom': 0.125,
                                 'offset': 0.03,
                                 'figsize': (12, 12)},
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['exemplar'],
                    'labels': LABELS['exemplar'],
                    'boxplot_hue': ('category', np.arange(len(LABELS['exemplar']))//12),
                    'boxplot_colors': COLORS['category'] ,
                    'n_components': 4,
                    'jointplot': True,
                    },
            'dendrogram': {'label_style': 'image', 
                           'colors':COLORS['exemplar'],
                           'labels': LABELS['exemplar'],
                           'zoom': 0.07,
                           'stagger': True,
                           'offset': 0.05,
                           'figsize': (12, 4)}
            },
        'spatial': {
            'topomap': {}
        },
        'temporal': {
            'time_course': {'colors': COLORS['exemplar'],
                            'labels': LABELS['category'],
                            'legend': np.arange(0, 72, 12),
                            'linewidth': 1},
            'confusion_matrix':  None,
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['exemplar'],
                    'labels': LABELS['exemplar']},
            'dendrogram':  None
        },
        'spatio_temporal': {
            'topomap': {}
        },
    },
    'artificial_object': {
        'full_response': {
            'confusion_matrix': {'label_style': 'image', 
                                 'colors':COLORS['artificial_object'],
                                 'labels': LABELS['artificial_object'],
                                 'zoom': 0.085,
                                 'offset': 0.05,},
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['artificial_object'],
                    'labels': LABELS['artificial_object'],
                    },
            'dendrogram': {'label_style': 'image', 
                           'colors':COLORS['artificial_object'],
                           'labels': LABELS['artificial_object'],
                           'zoom': 0.085,
                           'offset': 0.05,}
            },
        'spatial': {
            'topomap': {}
        },
        'temporal': {
            'time_course': {'colors': COLORS['artificial_object'], 'figsize': (4, 4)},
            'confusion_matrix':  None,
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['artificial_object'],
                    'labels': LABELS['artificial_object']},
            'dendrogram':  None
        },
        'spatio_temporal': {
            'topomap': {}
        },
    },
    'human_face': {
        'full_response': {
            'confusion_matrix': {'label_style': 'image', 
                                 'colors':COLORS['human_face'],
                                 'labels': LABELS['human_face'],
                                 'zoom': 0.085,
                                 'offset': 0.05,},
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['human_face'],
                    'labels': LABELS['human_face'],
                    },
            'dendrogram': {'label_style': 'image', 
                           'colors':COLORS['human_face'],
                           'labels': LABELS['human_face'],
                           'zoom': 0.085,
                           'offset': 0.05,}
            },
        'spatial': {
            'topomap': {}
        },
        'temporal': {
            'time_course': {'colors': COLORS['human_face'], 'figsize': (4, 4)},
            'confusion_matrix':  None,
            'mds': {'marker_style': 'image', 
                    'colors':COLORS['human_face'],
                    'labels': LABELS['human_face']},
            'dendrogram':  None
        },
        'spatio_temporal': {
            'topomap': {}
        },
    },
    'human_face_vs_artificial_object': {
        'full_response': {
            'calculate_bias': True,
            'confusion_matrix': {'label_style': 'pretty', 
                                 'colors':COLORS['human_face_vs_artificial_object'],
                                 'labels': LABELS['human_face_vs_artificial_object'],
                                 'plot_bias': True},
            'mds': {'marker_style': 'pretty', 
                    'colors':COLORS['human_face_vs_artificial_object'],
                    'labels': LABELS['human_face_vs_artificial_object']},
            'dendrogram': {'label_style': 'pretty', 
                           'colors':COLORS['human_face_vs_artificial_object'],
                           'labels': LABELS['human_face_vs_artificial_object']}
            },
        'spatial': {
            'calculate_bias': True,
            'topomap': {'plot_bias': True}
        },
        'temporal': {
            'time_course': {'colors': COLORS['human_face_vs_artificial_object'],
                            'labels': LABELS['human_face_vs_artificial_object'],
                            'legend': np.arange(0, 2), 'figsize': (4, 4),
                            'plot_bias': True},
            'confusion_matrix':  {'label_style': 'pretty', 
                                 'colors':COLORS['human_face_vs_artificial_object'],
                                 'labels': LABELS['human_face_vs_artificial_object'],
                                 'plot_bias': True},
            'mds': {'marker_style': 'pretty', 
                    'colors':COLORS['human_face_vs_artificial_object'],
                    'labels': LABELS['human_face_vs_artificial_object']},
            'dendrogram':  {'label_style': 'pretty', 
                           'colors':COLORS['human_face_vs_artificial_object'],
                           'labels': LABELS['human_face_vs_artificial_object']}
        },
        'spatio_temporal': {
            'calculate_bias': True,
            'topomap': {'plot_bias': True}
        },
    },
}

def get_exp_info(exp_cfg):
    fields = re.search(r"^(.+?)_(?:(?:full_response|single_electrode|temporal_window)|_)+_(.+?)_decoding$", exp_cfg['exp_id'])
    dataset, task = fields.group(1), fields.group(2)
    spatial_subset = exp_cfg['scheme'].get('spatial_subsets', None)
    temporal_subset = exp_cfg['scheme'].get('temporal_subsets', None)

    match (spatial_subset, temporal_subset):
        case (None, None):
            exp_type = "full_response"
        case (_, None):
            exp_type = "spatial"
        case (None, _):
            exp_type = "temporal"
        case (_, _):
            exp_type = "spatio_temporal"
    
    return dataset, task, exp_type

def get_channel(results_df):
    return results_df['spatial_subset'].str.extract(r'electrode_(\d+)')[0].astype(int)

def get_temporal_windows(results_df):
    results_df[['start_timepoint', 'end_timepoint']] = results_df['temporal_subset'].str.extract(r'timepoints_(\d+)-(\d+)').astype(int)
    results_df['end_timepoint'] = results_df['end_timepoint'] - 1
    results_df['mid_timepoint'] = (results_df['start_timepoint'] + results_df['end_timepoint'])/2
    results_df['temporal_window'] = results_df.apply(lambda row: f"{row['start_timepoint'] * 16}-{row['end_timepoint'] * 16} ms", axis=1)

    window_info = results_df.groupby(['temporal_subset']).agg(temporal_window=('temporal_window', 'first'), 
                                                                           order=('mid_timepoint', 'mean'))
    window_info['order'] = window_info['order'].rank(method='dense').astype(int)
    return results_df, window_info

def test_electrode_significance(results_df, task, alpha=0.05):
    # Use LME to check if each electrode is significantly different from chance.
    # Then use multiple tests to adjust for multiple comparisons
    return results_df.groupby(['experiment', 'model_id', 'channel', 'temporal_subset', 'partition']).agg(accuracy=('accuracy', 'mean'))
    chance = 100/len(LABELS[task])
    results_df = results_df.copy()
    results_df['accuracy'] -= chance
    test_df = {'model_id': [], 'experiment': [], 'model_id': [], 'channel': [], 'temporal_subset': [], 'partition': [], 'accuracy': [], 'p_value': []
    }
    for (exp_id, model_id, channel, temporal_subset, partition), channel_df in tqdm(results_df.groupby(level=['experiment', 'model_id', 'channel', 'temporal_subset', 'partition']), desc="Assessing spatio-temporal significance"):
        channel_df = channel_df.reset_index()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lme = smf.mixedlm("accuracy ~ 1", channel_df, groups=channel_df["subject"]).fit(method='bfgs')
        test_df['experiment'].append(exp_id)
        test_df['model_id'].append(model_id)
        test_df['channel'].append(channel)
        test_df['temporal_subset'].append(temporal_subset)
        test_df['partition'].append(partition)
        test_df['accuracy'].append(lme.params.loc['Intercept'] + chance)
        test_df['p_value'].append(lme.pvalues.loc['Intercept'])
    test_df = pd.DataFrame(test_df)
    reject, p_value_corrected, _, alpha_adjusted = multipletests(test_df['p_value'], alpha=alpha, method='bonferroni')
    test_df['reject'] = reject
    test_df.loc[~reject, 'accuracy'] = 0
    return test_df
    

def analyze_full_response_dynamics(results_df, task):
    kwargs = OPTIONS.get(task, {}).get('full_response', {})
    calculate_bias = kwargs.get('calculate_bias', False)
    cm_kwargs = kwargs.get('confusion_matrix', {})
    subtract_diagonal = cm_kwargs.pop('subtract_diagonal', False) if cm_kwargs is not None else False
    plot_cm_bias = kwargs['confusion_matrix'].pop('plot_bias', False) if kwargs['confusion_matrix'] is not None else False
    plot_mds_bias = kwargs['mds'].pop('plot_bias', False) if kwargs['mds'] is not None else False
    plot_dendrogram_bias = kwargs['dendrogram'].pop('plot_bias', False) if kwargs['dendrogram'] is not None else False

    cm_df = get_confusion_matrices(results_df, groups=['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'partition'])
    limits_df = cm_df.copy()
    if subtract_diagonal:
        limits_df = subtract_cm_diagonal(cm_df)
    vmin, vmax = np.min(limits_df.values), np.max(limits_df.values)
    if calculate_bias:
        cm_df = get_bias(cm_df)

    for (exp_id, model_id, spatial_subset, temporal_subset, partition), cm in cm_df.groupby(level=['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'partition']):
        fig_dir = os.path.join('figures', exp_id, model_id, spatial_subset, temporal_subset, partition)
        os.makedirs(fig_dir, exist_ok=True)
        rdm = get_rdm(cm)
        
        if (partition != 'bias') or (plot_mds_bias == True):
            plot_embedding(rdm.copy(), **kwargs['mds']).savefig(os.path.join(fig_dir, 'MDS.pdf'), bbox_inches='tight', format='pdf')
        if (partition != 'bias') or (plot_dendrogram_bias == True):
            plot_dendrogram(rdm.to_numpy(), **kwargs['dendrogram']).savefig(os.path.join(fig_dir, 'dendrogram.pdf'), bbox_inches='tight', format='pdf')
        
        if partition == 'bias':
            vmn, vmx = np.min(cm.values), np.max(cm.values)
            cmap = 'coolwarm'
        else:
            vmn, vmx = vmin, vmax
            cmap='turbo'
        
        if subtract_diagonal:
            cm = subtract_cm_diagonal(cm)
        if (partition != 'bias') or (plot_cm_bias == True):
            plot_confusion_matrix(cm, cmap=cmap, vmin=vmn, vmax=vmx, **kwargs['confusion_matrix']).savefig(os.path.join(fig_dir, 'confusion_matrix.pdf'), bbox_inches='tight', format='pdf')

def analyze_spatial_dynamics(results_df, task):
    kwargs = OPTIONS.get(task, {}).get('spatial', {})
    calculate_bias = kwargs.get('calculate_bias', False)
    plot_topomap_bias = kwargs['topomap'].pop('plot_bias', False) if kwargs['topomap'] is not None else False

    results_df['channel'] = get_channel(results_df)
    spatial_df = results_df.groupby(['experiment', 'model_id', 'channel', 'temporal_subset', 'partition']).agg(accuracy=('accuracy', 'mean'))
    vmin, vmax = np.min(spatial_df['accuracy'].values), np.max(spatial_df['accuracy'].values)
    # vmin = 100/len(LABELS[task])

    if calculate_bias:
        spatial_df = get_bias(spatial_df)

    for (exp_id, model_id, temporal_subset, partition), channel_df in spatial_df.groupby(level=['experiment', 'model_id', 'temporal_subset', 'partition']):
        fig_dir = os.path.join('figures', exp_id, model_id, temporal_subset, partition)
        # test_df = test_electrode_significance(channel_df, task)
        os.makedirs(fig_dir, exist_ok=True)
        if partition == 'bias':
            vmn, vmx = np.min(channel_df['accuracy'].values), np.max(channel_df['accuracy'].values)
            cmap = 'coolwarm'
        else:
            vmn, vmx = vmin, vmax
            cmap='turbo'
        if (partition != 'bias') or (plot_topomap_bias==True):
            plot_topomap(channel_df, vmin=vmn, vmax=vmx, cbar=True, cmap=cmap).savefig(os.path.join(fig_dir, 'topomap.pdf'), bbox_inches='tight', format='pdf')

def analyze_temporal_dynamics(results_df, task):
    kwargs = OPTIONS.get(task, {}).get('temporal', {})
    calculate_bias = kwargs.get('calculate_bias', False)
    cm_kwargs = kwargs.get('confusion_matrix', {})
    subtract_diagonal = cm_kwargs.pop('subtract_diagonal', False) if cm_kwargs is not None else False
    plot_timecourse_bias = kwargs['time_course'].pop('plot_bias', False) if kwargs['time_course'] is not None else False
    plot_cm_bias = kwargs['confusion_matrix'].pop('plot_bias', False) if kwargs['confusion_matrix'] is not None else False
    plot_mds_bias = kwargs['mds'].pop('plot_bias', False) if kwargs['mds'] is not None else False
    plot_dendrogram_bias = kwargs['dendrogram'].pop('plot_bias', False) if kwargs['dendrogram'] is not None else False

    results_df, window_info = get_temporal_windows(results_df)
    temporal_df = get_confusion_matrices(results_df, ['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'mid_timepoint', 'partition'])
    cm_limits_df = temporal_df.copy()
    if subtract_diagonal:
        cm_limits_df = subtract_cm_diagonal(temporal_df)
    cm_vmin, cm_vmax = np.min(cm_limits_df.values), np.max(cm_limits_df.values)

    class_accuracies = get_cm_diagonal(temporal_df)
    class_vmin, class_vmax = class_accuracies.min(), class_accuracies.max()

    if calculate_bias:
        temporal_df = get_bias(temporal_df)

    for (exp_id, model_id, spatial_subset, partition), window_df in temporal_df.groupby(['experiment', 'model_id', 'spatial_subset', 'partition']):
        fig_dir = os.path.join('figures', exp_id, model_id, spatial_subset, partition)
        os.makedirs(fig_dir, exist_ok=True)

        if partition == 'bias':
            cm_vmn, cm_vmx = np.min(window_df.values), np.max(window_df.values)
            class_bias = get_cm_diagonal(window_df)
            class_vmn, class_vmx = class_bias.min(), class_bias.max()
            cmap = 'coolwarm'
        else:
            cm_vmn, cm_vmx = cm_vmin, cm_vmax
            class_vmn, class_vmx = class_vmin, class_vmax
            cmap='turbo'
        if kwargs['time_course'] is not None:
            if (partition != 'bias') or (plot_timecourse_bias == True):
                plot_time_course(window_df.reset_index(), vmin=class_vmn, vmax=class_vmx, **kwargs['time_course']).savefig(os.path.join(fig_dir, 'time_course.pdf'), bbox_inches='tight', format='pdf')
        if kwargs['mds'] is not None:
            if (partition != 'bias') or (plot_mds_bias == True):
                plot_temporally_resolved_embedding(window_df, window_info=window_info, **kwargs['mds']).savefig(os.path.join(fig_dir, 'mds.pdf'), bbox_inches='tight', format='pdf')
        if kwargs['dendrogram'] is not None:
            if (partition != 'bias') or (plot_dendrogram_bias):
                plot_temporally_resolved_dendrograms(window_df, window_info=window_info, **kwargs['dendrogram']).savefig(os.path.join(fig_dir, 'dendrograms.pdf'), bbox_inches='tight', format='pdf')

        cm = window_df.copy()
        if subtract_diagonal:
            cm = subtract_cm_diagonal(cm)
        if kwargs['confusion_matrix'] is not None:
            if (partition != 'bias') or (plot_cm_bias == True):
                plot_temporally_resolved_confusion_matrices(cm, vmin=cm_vmn, vmax=cm_vmx, cmap=cmap, **kwargs['confusion_matrix'], window_info=window_info).savefig(os.path.join(fig_dir, 'confusion_matrices.pdf'), bbox_inches='tight', format='pdf')

def analyze_spatio_temporal_dynamics(results_df, task):
    kwargs = OPTIONS.get(task, {}).get('spatio_temporal', {})
    calculate_bias = kwargs.get('calculate_bias', False)
    plot_topomap_bias = kwargs['topomap'].pop('plot_bias', False) if kwargs['topomap'] is not None else False

    results_df['channel'] = get_channel(results_df)
    results_df, window_info = get_temporal_windows(results_df)
    
    results_df = results_df.groupby(['experiment', 'model_id', 'channel', 'temporal_subset', 'partition']).agg(accuracy=('accuracy', 'mean'))
    
    vmin, vmax = np.min(results_df['accuracy'].values), np.max(results_df['accuracy'].values)
    # vmin = 100/len(LABELS[task])
    if calculate_bias:
        results_df = get_bias(results_df)
    for (exp_id, model_id, partition), window_df in results_df.groupby(level=['experiment', 'model_id', 'partition']):
        fig_dir = os.path.join('figures', exp_id, model_id, partition)
        os.makedirs(fig_dir, exist_ok=True)
        test_df = test_electrode_significance(window_df, task)
        if partition == 'bias':
            vmn, vmx = np.min(window_df['accuracy'].values), np.max(window_df['accuracy'].values)
            cmap = 'coolwarm'
        else:
            vmn, vmx = vmin, vmax
            cmap='turbo'
        if (partition != 'bias') or (plot_topomap_bias == True):
            plot_temporally_resolved_topomaps(test_df, window_info=window_info, vmin=vmn, vmax=vmx, cmap=cmap, **kwargs['topomap']).savefig(os.path.join(fig_dir, 'topomaps.pdf'), bbox_inches='tight', format='pdf')

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

def select_models(results_df, metric='accuracy'):
    summary_df = summarize_results(results_df)
    opt_partition = f"val_{metric}"
    opt_df = summary_df.groupby(['experiment', 'model_id', 'subject', 'fold', 'hyperparameters']).mean(numeric_only=True)
    opt_df = opt_df.loc[opt_df.groupby(['experiment', 'subject', 'model_id', 'fold'])[opt_partition].idxmax()].reset_index()
    opt_df = results_df.merge(opt_df[['experiment', 'subject', 'model_id', 'fold', 'hyperparameters']], how='inner')
    # opt_df = opt_df.loc['test' in opt_df['partition']]
    return opt_df

def get_confusion_matrices(results_df, groups, method='size'):
    match method:
        case 'size':
            results_df = results_df.explode(["labels", "predicted"])
            cms = results_df.groupby(groups + ["labels", "predicted"]).size().unstack(fill_value=0)
            cms = cms.div(cms.sum(axis=1), axis=0) 
        case 'mean':
            cms = results_df.groupby(groups + ["labels", "predicted"])['accuracy'].mean().unstack(fill_value=0)
        case _:
            raise ValueError(f"Unsupported method {method}. Method must be one of: [size, mean].")
        
    return cms * 100

def get_cm_diagonal(confusion_matrix):
    lab = confusion_matrix.index.get_level_values('labels')
    j = confusion_matrix.columns.get_indexer(lab)    
    cm = confusion_matrix.to_numpy(copy=True)
    return cm[np.arange(cm.shape[0]), j]

def subtract_cm_diagonal(confusion_matrix):
    lab = confusion_matrix.index.get_level_values('labels')
    j = confusion_matrix.columns.get_indexer(lab)    
    cm = confusion_matrix.to_numpy(copy=True)         
    cm[np.arange(cm.shape[0]), j] = 0
    confusion_matrix = pd.DataFrame(cm, index=confusion_matrix.index, columns=confusion_matrix.columns)
    return confusion_matrix

def get_bias(df):
    bias = df.xs('confounded_test', level='partition', drop_level=True) - df.xs('unconfounded_test', level='partition', drop_level=True)
    bias = pd.concat({'bias': bias}, names=['partition']).reorder_levels(df.index.names)
    df = pd.concat([df, bias]).sort_index()
    return df

def get_rdm(confusion_matrix, clip=True):
    cm_norm = confusion_matrix.copy()/np.diag(confusion_matrix.values)
    rdm = 1 - np.sqrt(cm_norm.copy() * cm_norm.to_numpy().T)
    if clip:
        rdm = np.clip(rdm, a_min=0.0, a_max=None)
    return rdm

def get_significance_symbol(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def analyze_results(index_path, exp_cfg=None, alpha=0.05):
    result_dir = os.path.dirname(index_path)
    exp_id = result_dir.split('/')[1]

    files = glob.glob(f"{result_dir}/**/*.parquet", recursive=True)
    results_df = dataset(files, format="parquet").to_table().to_pandas()
    results_df = select_models(results_df)
    results_df.to_csv(os.path.join(result_dir, 'summary.csv'))

    results_df = results_df[results_df['temporal_subset'].isin(TEMPORAL_SUBSETS)]
    results_df = results_df[results_df['spatial_subset'].isin(SPATIAL_SUBSETS)]

    if exp_cfg is None:
        exp_cfg = load_experiment_cfg(exp_id)
    dataset_id, task, exp_type = get_exp_info(exp_cfg)
    print(dataset_id, task, exp_type)
    match exp_type:
        case "full_response":
            analyze_full_response_dynamics(results_df, task)
        case "spatial":
            analyze_spatial_dynamics(results_df, task)
        case "temporal":
            analyze_temporal_dynamics(results_df, task)
        case "spatio_temporal":
            analyze_spatio_temporal_dynamics(results_df, task)
    plt.close()
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze experiment results.')
    parser.add_argument('index_path', type=str, help='Path to the index file containing experiment results.')
    args = parser.parse_args()
    analyze_results(args.index_path)
