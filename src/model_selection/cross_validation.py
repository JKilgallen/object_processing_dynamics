import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

def generate_stratified_folds(feature, n_folds=12, n_nested_folds=11):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = [[] for _ in range(n_folds)]
    for idx, (train_val_idxs, test_idxs) in enumerate(skf.split(feature, feature)):
        nskf = StratifiedKFold(n_splits = n_folds-1, shuffle=True, random_state=42)
        for train_idxs, val_idxs in nskf.split(feature[train_val_idxs], feature[train_val_idxs]):
                folds[idx].append({
                        'train': torch.tensor(train_val_idxs[train_idxs], dtype=torch.long),
                        'val': torch.tensor(train_val_idxs[val_idxs], dtype=torch.long)})
        
                assert np.allclose(np.unique(feature[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(feature[train_val_idxs[val_idxs]], return_counts=True)[1]/len(val_idxs)) and \
                        np.allclose(np.unique(feature[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(feature[test_idxs], return_counts=True)[1]/len(test_idxs)),\
                        "Feature stratification failed."
        folds[idx].append({
                'train': torch.tensor(train_val_idxs, dtype=torch.long),
                'test': torch.tensor(test_idxs, dtype=torch.long)})
    return folds

def generate_paired_folds(categories, exemplars, n_folds=12, n_nested_folds=11):
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    skgf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    folds = [[] for _ in range(n_folds)]

    trial_idxs = np.arange(len(categories))
    confounded_folds = list(skf.split(categories, exemplars))
    unconfounded_folds = list(skgf.split(categories, exemplars, groups= exemplars))
    for idx, ((_, confounded_test), (_, unconfounded_test)) in enumerate(zip(confounded_folds, unconfounded_folds)):

        train_val_idxs = trial_idxs[~np.isin(trial_idxs, confounded_test) & ~np.isin(trial_idxs, unconfounded_test)]

        confounded_test_idxs = confounded_test[~np.isin(confounded_test, unconfounded_test)]
        unconfounded_test_idxs = unconfounded_test[~np.isin(unconfounded_test, confounded_test)]
        nskf = StratifiedKFold(n_splits=n_folds - 1, shuffle=True, random_state=42)
        for train_idxs, val_idxs in nskf.split(exemplars[train_val_idxs], exemplars[train_val_idxs]):
                folds[idx].append({
                        'train': torch.tensor(train_val_idxs[train_idxs], dtype=torch.long),
                        'val': torch.tensor(train_val_idxs[val_idxs], dtype=torch.long)})
            
                assert np.allclose(np.unique(categories[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(categories[train_val_idxs[val_idxs]], return_counts=True)[1]/len(val_idxs)) and \
                        np.allclose(np.unique(categories[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(categories[confounded_test_idxs], return_counts=True)[1]/len(confounded_test_idxs)) and \
                        np.allclose(np.unique(categories[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(categories[unconfounded_test_idxs], return_counts=True)[1]/len(unconfounded_test_idxs)),\
                        "Partitions do not have the same category distribution. "

                assert np.all(np.isin(exemplars[train_val_idxs[train_idxs]], exemplars[train_val_idxs[val_idxs]])) and \
                np.all(np.isin(exemplars[train_val_idxs[train_idxs]], exemplars[confounded_test_idxs])), \
                "Stimulus-confounded partitions contain exemplars not in the training set."

                assert np.all(~np.isin(exemplars[unconfounded_test_idxs], exemplars[train_val_idxs[train_idxs]])) and \
                np.all(~np.isin(exemplars[unconfounded_test_idxs], exemplars[train_val_idxs[val_idxs]])), \
                "Stimulus-unconfounded partitions contain non-unique exemplars."

                assert np.allclose(np.unique(exemplars[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(exemplars[confounded_test_idxs], return_counts=True)[1]/len(confounded_test_idxs)) and \
                        np.allclose(np.unique(exemplars[train_val_idxs[train_idxs]], return_counts=True)[1]/len(train_idxs),
                                np.unique(exemplars[train_val_idxs[val_idxs]], return_counts=True)[1]/len(val_idxs)), \
                                "Stimulus-confounded partitions do not have the same exemplar distribution."
        folds[idx].append({'train': torch.tensor(train_val_idxs, dtype=torch.long),
                'confounded_test': torch.tensor(confounded_test_idxs, dtype=torch.long),
                'unconfounded_test': torch.tensor(unconfounded_test_idxs, dtype=torch.long)})
        
    return folds


