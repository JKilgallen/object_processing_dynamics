from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, PercentFormatter, IndexLocator
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import seaborn as sns
from mne import channels, create_info, viz
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS, TSNE
import numpy as np

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{tgheros}%\
                                \usepackage{sansmath}%\
                                \sansmath')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "tgheros"

MONTAGE = channels.make_standard_montage('GSN-HydroCel-128')
INFO = create_info(ch_names=MONTAGE.ch_names[:124], sfreq=62.5, ch_types="eeg")
INFO.set_montage(montage=MONTAGE)

def add_pretty_ticks(ax, labels, colors, tick_axes='both'):
    if tick_axes in ["both", "x"]:
        ax.set_xticks(ax.get_xticks(), [f"\\Large \\textbf{{{l}}}" for l in labels.values()])
        for ldx, label in enumerate(ax.get_xticklabels()):
            label.set_color(colors[ldx])
    else:
        ax.set_xlabel("")
        ax.set_xticks([], [])

    if tick_axes in ["both", "y"]:
        ax.set_yticks(ax.get_yticks(), [f"\\large \\textbf{{{l}}}" for l in labels.values()])
        for ldx, label in enumerate(ax.get_yticklabels()):
            label.set_color(colors[ldx])
    else:
        ax.set_ylabel("")
        ax.set_yticks([], [])

def add_pretty_markers(ax, coords, labels, colors):
    for i, label in enumerate(labels.values()):
        ax.annotate(f"\\huge \\textbf{{{label}}}", (coords[i, 0], coords[i, 1]), ha='center', va='center', color=colors[i])

def replace_labels_with_images(ax, labels, colors, tick_axes='both', zoom=1.0, offset = 0.03, stagger=False):
    if tick_axes in ["both", "x"]:
        for idx, loc in enumerate(ax.xaxis.get_majorticklocs()):
            y = -offset
            if stagger:
                y *= (1 + (idx%3))
            img_box = AnnotationBbox(OffsetImage(imread(labels[idx]), zoom=zoom), 
                           (loc, y),
                       xycoords=btf(ax.transData, ax.transAxes),    
                       box_alignment=(0.5,0.5), frameon=True, clip_on=False,
                       bboxprops=dict(fc='none', ec=colors[idx], lw=2, boxstyle='round,pad=0.05'))
            ax.add_artist(img_box)
        ax.set_xticks([], [])

    if tick_axes in ["both", "y"]:        
        for idx, loc in enumerate(ax.yaxis.get_majorticklocs()):
            x = -offset
            if stagger:
                x *= (1 + (idx%3))
            img_box = AnnotationBbox(OffsetImage(imread(labels[idx]), zoom=zoom), 
                           (x, loc),
                       xycoords=btf(ax.transAxes, ax.transData),    
                       box_alignment=(0.5,0.5), frameon=True, clip_on=False,
                       bboxprops=dict(fc='none', ec=colors[idx], lw=2, boxstyle='round,pad=0.05'))
            ax.add_artist(img_box)
        # 
        ax.set_yticks([], [])

def add_image_markers(ax, coords, labels, colors):
    images = list(map(imread, labels.values()))
    for idx, (x, y) in enumerate(coords):
        img_box = AnnotationBbox(OffsetImage(images[idx], zoom=0.07), 
                (x, y), 
                box_alignment=(0.5,0.5), frameon=True, clip_on=False,
                bboxprops=dict(fc='none', ec=colors[idx], lw=2, boxstyle='round,pad=0.05'))
        ax.add_artist(img_box)

def get_cnorm(vmin, vmax, center=None):
    center = (vmin + vmax) / 2 if center is None else center
    vrange = max(vmax - center, center - vmin)
    return Normalize(center - vrange, center + vrange)

def plot_confusion_matrix(cm, vmin=0, vmax=100, title=None, figsize=(4,4), label_axes="both", label_style='pretty', tick_axes = "both", cbar=True, ax = None, labels=None, colors=None, zoom=1.0, offset=0.03, stagger=False, cmap='turbo', cnorm=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    if (label_style == "pretty" and tick_axes in ["both", "x", "y"]) and (labels is None or colors is None):
        raise ValueError("Both labels and colors must be set to use tick_style='pretty'")

    if cnorm is None:
        cnorm = get_cnorm(vmin=vmin, vmax=vmax, center = 0 if cmap == 'coolwarm' else None)

    cmap=sns.color_palette(cmap, as_cmap=True)

    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax, norm=cnorm, linewidths=0.5, linecolor='black', square=True, cbar=False)

    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(labels)) + 0.5))
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(len(labels)) + 0.5))
    if label_style == "pretty":
        add_pretty_ticks(ax, labels, colors, tick_axes)
        ax.set_xlabel("\\Large \\textbf{{Predicted label}}", labelpad=5)
        ax.set_ylabel("\\Large \\textbf{{True label}}", labelpad=5)
    elif label_style == "image":
        ax.set_xlabel(""); ax.set_ylabel("")
        # ax.set_xlabel("\\Large Predicted stimulus", labelpad=38)
        # ax.set_ylabel("\\Large True stimulus", labelpad=38)
        replace_labels_with_images(ax, labels, colors, tick_axes, zoom=zoom, offset=offset, stagger=stagger)
    else:
        ax.set_xticks([], []); ax.set_yticks([], [])

    if not label_axes in ['x', 'both']:
        ax.set_xlabel("")
    if not label_axes in ['y', 'both']:    
        ax.set_ylabel("")
        
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')

    if cbar:
        sm = ScalarMappable(cmap=cmap, norm=cnorm)
        fig.colorbar(sm, ax=ax, location='top', label='\\Large \\textbf{{Classification rate}}', pad=0.01, shrink=0.85,
                 ticks=AutoLocator(), format=PercentFormatter(xmax=100, symbol='\%', is_latex=True))

    return fig

def plot_embedding(rdm, kind='MDS', init=None, n_components=2, jointplot = False, title=None, ax = None, axis_labels=True, marker_style = "pretty", labels=None, colors=None, boxplot_colors=None, boxplot_hue="labels"):
    n_plots = n_components//2
    if jointplot:
        fig = plt.figure(figsize=(n_plots * 6, 6))
        G = GridSpec(1, n_plots, figure=fig)
        ax = [None for _ in range(n_plots)]
    elif ax is None:
        fig, ax = plt.subplots(figsize=(n_plots * 4, 4))
    else:
        fig = None

    if n_plots == 1:
        ax = [ax]
    
    match kind:
        case 'MDS':
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, n_init=1)
            coords = mds.fit_transform(rdm, init=init)
        case 'tSNE':
            tsne = TSNE(n_components=n_components, metric='precomputed', init="random", method="exact", perplexity = 2)
            coords = tsne.fit_transform(rdm)

    rdm[[f'Dimension {idx+1}' for idx in range(n_components)]] = coords
    if type(boxplot_hue) != str:
        boxplot_hue, vals = boxplot_hue
        rdm[boxplot_hue] = vals
    for pdx in range(n_plots):
        if jointplot:
            sgs = G[pdx].subgridspec(2, 2, height_ratios=[1,4], width_ratios=[4,1], hspace=0.05, wspace=0.05)
            ax[pdx] = fig.add_subplot(sgs[1, 0])
            ax_mx, ax_my = fig.add_subplot(sgs[0,0], sharex=ax[pdx]), fig.add_subplot(sgs[1, 1], sharey=ax[pdx])
            sns.boxplot(data=rdm, x=f'Dimension {1 + (2*pdx)}', hue=boxplot_hue, ax=ax_mx, legend=False, palette=boxplot_colors)
            sns.boxplot(data=rdm, y=f'Dimension {2 + (2*pdx)}', hue=boxplot_hue, ax=ax_my, legend=False, palette=boxplot_colors)
            ax_mx.tick_params(left=False); ax_my.tick_params(bottom=False)
            ax_mx.xaxis.label.set_visible(False); ax_my.yaxis.label.set_visible(False)
        ax[pdx].scatter(x=rdm[f'Dimension {1 + (2*pdx)}'], y=rdm[f'Dimension {2 + (2*pdx)}'], s=0)
        
        if marker_style == 'pretty':
            add_pretty_markers(ax[pdx], coords[:, pdx*2:(pdx+1)*2], labels, colors)
        elif marker_style == 'image':
            add_image_markers(ax[pdx], coords[:, pdx*2:(pdx+1)*2], labels, colors)
        ax[pdx].set_xlim(coords[:, pdx*2].min() -0.1, coords[:, pdx*2].max() + 0.1)
        ax[pdx].set_ylim(coords[:, (pdx*2)+1].min() -0.1, coords[:, (pdx*2)+1].max() + 0.1)

        if axis_labels:
            ax[pdx].set_ylabel(f"\\Large \\textbf{{Dimension {2 + (2*pdx)}}}", labelpad=10)
            ax[pdx].set_xlabel(f"\\Large \\textbf{{Dimension {1 + (2*pdx)}}}", labelpad=10)
        else:
            ax[pdx].set_ylabel(""); ax[pdx].set_xlabel("");
        ax[pdx].set_xticks([])
        ax[pdx].set_yticks([])
        if title:
            ax[pdx].set_title(title)

    return fig

def plot_dendrogram(rdm, label_style="pretty", ylabel='left', figsize=(4, 4), title=None, ax=None, labels=None, colors=None, orientation='top', offset=0.03, zoom=1.0, stagger=False):
    if ax is None:
        if labels is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    hierarchy = linkage(rdm[np.triu_indices_from(rdm, k=1)], method='average')
    ddata = dendrogram(hierarchy, labels=list(labels.keys()), get_leaves=True, no_plot=True)
    for icoord, dcoord in list(zip(ddata['icoord'], ddata['dcoord'])):
        x0, x1, x2, x3 = icoord
        y0, y1, y2, y3 = dcoord
        if orientation=='top':
            ax.plot([x0, (x2+x0)/2], [y0, y2], color='black', linewidth=1)
            ax.plot([x3, (x1+x3)/2], [y3, y1], color='black', linewidth=1)
        elif orientation=='right':
            ax.plot([y0, y2], [x0, (x2+x0)/2], color='black', linewidth=1)
            ax.plot([y3, y1], [x3, (x1+x3)/2], color='black', linewidth=1) 
    
    leaves = ddata['ivl']
    ddata['leaves'] = np.array(ddata['leaves'])
    order = np.argsort(ddata['leaves'])
    locs = ddata['leaves'][order]*10 + 5
    
    labels = {idx: labels[leaf] for idx, leaf in enumerate(leaves)}
    colors = [colors[leaf] for leaf in leaves]
    if orientation=='top':
        ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
        tick_axis='x'
    else:
        ax.yaxis.set_major_locator(ticker.FixedLocator(locs))
        tick_axis='y'

    if label_style == "pretty":
        add_pretty_ticks(ax, labels, colors, tick_axis)
    elif label_style == 'image':
        replace_labels_with_images(ax, labels, colors, tick_axis, offset=offset, zoom=zoom, stagger=stagger)
    else:
        ax.set_xticks([], [])

    if ylabel is not None:
        if ylabel == 'right':
            ax.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True); 
            ax.yaxis.set_label_position('right')
            ax.spines[['left', 'top']].set_visible(False)
        else:
            ax.tick_params(axis='y', which = 'both', left=True, right=False, labelleft=True, labelright=False); 
            ax.spines[['right', 'top']].set_visible(False)
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
        ax.set_ylabel("\\Large \\textbf{{Dissimilarity}}")
    else:
        ax.spines[['right', 'left', 'top']].set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False, labelright=False); 
        ax.set_yticks([], [])
    
    ax.set_title(title)
    return fig

def plot_topomap(channel_df, title=None, vmin=0, vmax=100, cbar=True, ax=None, cmap="turbo", cnorm=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = None
    
    if cnorm is None:
        cnorm = get_cnorm(vmin=vmin, vmax=vmax, center = 0 if cmap == 'coolwarm' else None)
    cmap=sns.color_palette(cmap, as_cmap=True).copy()
    cmap.set_under("lightgray")
    channel_df = channel_df.reset_index()
    data = channel_df.loc[channel_df.sort_values('channel').index, 'accuracy'].to_numpy()
    viz.plot_topomap(data, INFO, cmap=cmap, sensors=False, axes=ax, show=False, sphere=0.09, image_interp='nearest', cnorm=cnorm, contours=0)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if cbar:        
        sm = ScalarMappable(cmap=cmap, norm=cnorm)
        fig.colorbar(sm, ax=ax, shrink=0.85, location='top', label='\\Large \\textbf{{Decoding accuracy}}', use_gridspec=True, ticks=AutoLocator(), format=PercentFormatter(xmax=100, symbol='\%', is_latex=True))
    
    return fig

def plot_time_course(results_df, vmin=0.1, vmax=1, title=None, legend=None, ax=None, figsize=(12, 4), labels=None, colors=None, hue='labels', linewidth=2):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    
    results_df['time'] = results_df['mid_timepoint'].astype(int) * 16
    results_df['accuracy'] = 0.0
    for l in results_df.columns.get_level_values('predicted'):
        idxs = results_df['labels'] == l
        if sum(idxs) > 0:
            results_df.loc[idxs, 'accuracy'] = results_df.loc[idxs, l]

    plot_legend = legend is not None
    sns.pointplot(data=results_df, x='time', y='accuracy', hue=hue, ax=ax, palette=colors, marker='o', legend=plot_legend, linewidth=linewidth)
    chance_line = ax.axhline(y=100/results_df['labels'].nunique(), linestyle=':', color='grey')
    results_df = results_df.groupby(['time']).agg(accuracy=('accuracy', 'mean')).reset_index()
    sns.pointplot(data=results_df, x='time', y='accuracy', ax=ax, marker='_', color='black', linestyle='--', legend=plot_legend, linewidth=linewidth)

    ax.set_xlabel('\\Large \\textbf{{Time (ms)}}')
    ax.set_ylabel('\\Large \\textbf{{Decoding accuracy}}')
    ax.set_ylim(vmin-1, vmax + 1)
    
    if title:
        ax.set_title(title, fontsize=16)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter())

    if plot_legend:
        handles, _ = ax.get_legend_handles_labels()
        handles = [handles[idx] for idx in legend]
        handles = handles + [plt.Line2D([0], [0], marker='o', color='black'), chance_line]
        labels = list(f"\\large {l}" for l in labels.values())+ ['Mean'] + ['Chance']
        ax.legend(title='\\Large Category', handles=handles, labels=labels, loc='upper right')

    ax.grid(True, which='both', axis='both')

    return fig

def plot_temporally_resolved_confusion_matrices(cm_df, window_info, vmin=0,  vmax=100, label_style="pretty", labels=None, colors=None, cmap="turbo", offset=0.03, stagger=False):
     
    fig, ax = plt.subplots(1, len(window_info), figsize=(24, 3))
    window_info = window_info.to_dict()

    vmin = np.min(cm_df.values) if vmin is None else vmin
    vmax = np.max(cm_df.values) if vmax is None else vmax
    cnorm = get_cnorm(vmin=vmin, vmax=vmax, center = 0 if cmap == 'coolwarm' else None)

    for (exp_id, model_id, spatial_subset, temporal_subset, partition), cm in cm_df.groupby(level=['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'partition']):
        idx = window_info['order'][temporal_subset] - 1
        temporal_window = window_info['temporal_window'][temporal_subset]
        tick_axes = 'both' if idx == 0 else 'x'
        label_axes = 'both' if idx == 0 else 'x'
        plot_confusion_matrix(cm, vmin=vmin, vmax=vmax, ax=ax[idx], cbar=False, label_axes=label_axes, tick_axes=tick_axes, label_style=label_style, title=f"\\Large \\textbf{{{temporal_window}}}", labels=labels, colors=colors, cmap=cmap, cnorm=cnorm)
    
    sm = ScalarMappable(cmap=cmap, norm=cnorm)
    fig.colorbar(sm, ax=ax, shrink=0.9, location='right', label='\\Large Decoding accuracy', pad=0.01, ticks=AutoLocator(), format=PercentFormatter(xmax=100, symbol='\%', is_latex=True))

    return fig

def plot_temporally_resolved_embedding(cm_df, window_info, kind='MDS', marker_style = "pretty", labels=None, colors=None):
     
    fig, ax = plt.subplots(1, len(window_info), figsize=(18, 2), sharex=True, sharey=True)
    window_info = window_info.to_dict()
    init = None
    for (exp_id, model_id, spatial_subset, temporal_subset, partition), cm in cm_df.groupby(level=['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'partition']):
        idx = window_info['order'][temporal_subset] - 1
        temporal_window = window_info['temporal_window'][temporal_subset]

        cm_norm = cm.copy()/(np.diag(cm.values) + 1e-8)
        rdm = 1 - np.sqrt(cm_norm.copy() * cm_norm.to_numpy().T)
        rdm = np.clip(rdm, a_min = 0, a_max = None)
        init = plot_embedding(rdm, kind=kind, init=init, ax=ax[idx], marker_style=marker_style, axis_labels = False, title=f"\\Large \\textbf{{{temporal_window}}}", labels=labels, colors=colors)    

    return fig

def plot_temporally_resolved_dendrograms(cm_df, window_info, label_style="pretty", labels=None, colors=None, orientation='top'):
    fig, ax = plt.subplots(1, len(window_info), figsize=(24, 3), sharey=True)
    window_info = window_info.to_dict()
    init = None
    for (exp_id, model_id, spatial_subset, temporal_subset, partition), cm in cm_df.groupby(level=['experiment', 'model_id', 'spatial_subset', 'temporal_subset', 'partition']):
        idx = window_info['order'][temporal_subset] - 1
        temporal_window = window_info['temporal_window'][temporal_subset]
        ylabel = 'right' if idx == len(window_info['order']) - 1 else None
        cm_norm = cm.copy()/np.diag(cm.values)
        rdm = 1 - np.sqrt(cm_norm.copy() * cm_norm.to_numpy().T)
        rdm = np.clip(rdm.to_numpy(), a_min = 0, a_max = None)
        plot_dendrogram(rdm, ax=ax[idx], ylabel = ylabel, label_style=label_style, title=f"\\Large \\textbf{{{temporal_window}}}", labels=labels, colors=colors, orientation=orientation)
    
    ax[0].yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax[0].yaxis.set_major_formatter("{x:.2f}")
    return fig
    
def plot_temporally_resolved_topomaps(window_df, window_info, vmin=0, vmax=100, cmap="turbo"):
     
    fig, ax = plt.subplots(1, len(window_info), figsize=(16, 2))
    window_info = window_info.to_dict()
    
    cnorm = get_cnorm(vmin=vmin, vmax=vmax, center = 0 if cmap == 'coolwarm' else None)
    for (exp_id, model_id, temporal_subset, partition), df in window_df.groupby(['experiment', 'model_id', 'temporal_subset', 'partition']):
        idx = window_info['order'][temporal_subset] - 1
        temporal_window = window_info['temporal_window'][temporal_subset]
        plot_topomap(df, vmin=vmin, vmax=vmax, ax=ax[idx], cbar=False, title=f"\\Large \\textbf{{{temporal_window}}}", cmap=cmap, cnorm=cnorm)
    
    sm = ScalarMappable(cmap=cmap, norm=cnorm)
    fig.colorbar(sm, ax=ax, shrink=0.9, location='bottom', label='\\Large \\textbf{{Decoding accuracy}}', pad=0.01,
                 ticks=AutoLocator(), format=PercentFormatter(xmax=100, symbol='\%', is_latex=True))

    return fig

        