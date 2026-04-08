import numpy as np
from sklearn import metrics
import pickle
from math import comb

from tools import load


rng = np.random.default_rng(42)  # for reproducibility

n_hd = 10_000
n_out = 6

file_train = 'mix_100_20_1'
file_test = 'mix_50_20_1'
file_save = 'gs_binary_mixed'

grid_p = [0.01, 0.1]
grid_normalized = ['raw']
grid_kernel = ['top']
grid_n_fold = 10
grid_p_hd = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
grid_d = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


# --- Load data ---
train_data_raw, train_sequence, train_times_sec, train_sequence_sec = load(file_train, reduced=True)
test_data_raw, test_sequence, test_times_sec, test_sequence_sec = load(file_test, reduced=True)

h_train = np.median(np.diff(train_times_sec))
h_test  = np.median(np.diff(test_times_sec))
n_sensors = train_data_raw.shape[1]


# --- Expansion helpers ---
def backward_diff_array(y, h, n):
    """n-th backward finite difference (zero-padded to preserve length)."""
    coeffs = np.array([(-1)**k * comb(n, k) for k in range(n + 1)])
    raw = np.convolve(y, coeffs, mode='valid') / h**n
    return np.concatenate([np.zeros(n), raw])


def expand_with_derivatives(data, h, max_order):
    if max_order == 0:
        return data.copy()
    derivs = []
    for order in range(1, max_order + 1):
        d = np.apply_along_axis(
            lambda col: backward_diff_array(col, h, order), axis=0, arr=data)
        derivs.append(d)
    return np.hstack([data] + derivs)


# --- Per-fold pair resampling ---
all_ordered_pairs = [(i, j) for i in range(n_sensors)
                            for j in range(n_sensors) if i != j]   # 56 pairs
all_pairs_no_self = [(i, j) for i in range(n_sensors)
                            for j in range(n_sensors) if i != j]    # 56 pairs

def sample_pairs(seed):
    """Return fresh ratio_pairs and diff_pairs for a given seed."""
    rng_p = np.random.default_rng(seed)
    rp = [all_ordered_pairs[i] for i in rng_p.permutation(len(all_ordered_pairs))]
    dp = [all_ordered_pairs[i] for i in rng_p.permutation(len(all_ordered_pairs))]
    return rp, dp

def needs_resampling(n_ratios, n_diffs):
    """56R/56D uses all pairs (excl. self) — no resampling needed."""
    n_all = n_sensors * (n_sensors - 1)  # 56
    return (n_ratios > 0 and n_ratios < n_all) or (n_diffs > 0 and n_diffs < n_all)

def make_ratios(data, pairs, n):
    return np.column_stack([data[:, i] / (data[:, j] + 1e-8) for i, j in pairs[:n]])

def make_diffs(data, pairs, n):
    return np.column_stack([data[:, i] - data[:, j] for i, j in pairs[:n]])

def build_expansion(data, deriv_cache, max_order, n_ratios, n_diffs, rp, dp):
    parts = [deriv_cache[max_order]]
    if n_ratios > 0:
        parts.append(make_ratios(data, rp, n_ratios))
    if n_diffs > 0:
        parts.append(make_diffs(data, dp, n_diffs))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


# --- Config specs (same as run_expand_and_sparsify_mixed.py) ---
CONFIG_SPECS = {
    '∂¹+∂² + 56R + 56D':  (2, 56, 56),
}

# Precompute derivative expansions for train and test (deterministic, no pairs)
deriv_orders = set(spec[0] for spec in CONFIG_SPECS.values())
deriv_cache_train = {order: expand_with_derivatives(train_data_raw, h_train, max_order=order)
                     for order in deriv_orders}
deriv_cache_test  = {order: expand_with_derivatives(test_data_raw, h_test, max_order=order)
                     for order in deriv_orders}


# Build train labels
train_labels = np.zeros_like(train_times_sec)
for i, t in enumerate(train_sequence_sec):
    try:
        flag = (train_times_sec > train_sequence_sec[i]) & (train_times_sec < train_sequence_sec[i+1])
    except IndexError:
        flag = (train_times_sec > train_sequence_sec[i])
    train_labels[flag] = int(train_sequence[i][1])

# Build test labels
test_labels = np.zeros_like(test_times_sec)
for i, t in enumerate(test_sequence_sec):
    try:
        flag = (test_times_sec > test_sequence_sec[i]) & (test_times_sec < test_sequence_sec[i+1])
    except IndexError:
        flag = (test_times_sec > test_sequence_sec[i])
    test_labels[flag] = int(test_sequence[i][1])


# --- Grid search ---
params = {'expansion': [], 'p_hd': [], 'd': [], 'p': [], 'normalized': [], 'kernel': [], 'n_fold': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}
pair_log = {}  # exp_name -> [(rp, dp), ...] length grid_n_fold

for exp_name, (max_order, n_ratios, n_diffs) in CONFIG_SPECS.items():
    resample = needs_resampling(n_ratios, n_diffs)
    has_pairs = n_ratios > 0 or n_diffs > 0

    # Pre-sample pairs for all folds
    pairs_list = []
    fold_pairs = {}
    for fold in range(grid_n_fold):
        if resample:
            rp, dp = sample_pairs(fold)
            fold_pairs[fold] = (rp, dp)
            pairs_list.append((rp[:n_ratios], dp[:n_diffs]))
        elif has_pairs:
            fold_pairs[fold] = (all_pairs_no_self, all_pairs_no_self)
            pairs_list.append((all_pairs_no_self[:n_ratios], all_pairs_no_self[:n_diffs]))
        else:
            fold_pairs[fold] = (None, None)
            pairs_list.append(([], []))
    pair_log[exp_name] = pairs_list

    for kernel in grid_kernel:
        for p_hd in grid_p_hd:
            for d in grid_d:
                for p in grid_p:
                    for normalized in grid_normalized:
                        for n_fold in range(grid_n_fold):
                            k = int(d * n_hd)

                            # Build expansion with per-fold pairs
                            if has_pairs:
                                rp, dp = fold_pairs[n_fold]
                                train_expanded = build_expansion(
                                    train_data_raw, deriv_cache_train, max_order,
                                    n_ratios, n_diffs, rp, dp)
                                test_expanded = build_expansion(
                                    test_data_raw, deriv_cache_test, max_order,
                                    n_ratios, n_diffs, rp, dp)
                            else:
                                train_expanded = deriv_cache_train[max_order]
                                test_expanded  = deriv_cache_test[max_order]

                            # No normalization (raw)
                            x_train = train_expanded
                            x_test  = test_expanded

                            n_dense = x_train.shape[1]

                            # HD projection (shared W_hd)
                            W_hd = np.random.binomial(n=1, p=p_hd, size=(n_hd, n_dense))

                            # Train: project and sparsify
                            x_hd_train = x_train @ W_hd.T
                            if kernel == 'top':
                                ranks = np.argsort(np.argsort(-x_hd_train, axis=1), axis=1)
                                z_hd_train = np.where(ranks < k, 1., 0.)
                            elif kernel == 'rank':
                                z_hd_train = np.where(np.argsort(x_hd_train) < k, 1., 0.)

                            # Test: project and sparsify
                            x_hd_test = x_test @ W_hd.T
                            if kernel == 'top':
                                ranks = np.argsort(np.argsort(-x_hd_test, axis=1), axis=1)
                                z_hd_test = np.where(ranks < k, 1., 0.)
                            elif kernel == 'rank':
                                z_hd_test = np.where(np.argsort(x_hd_test) < k, 1., 0.)

                            # Train readout weights
                            W_out = np.zeros((n_out, n_hd))
                            for i, row in enumerate(z_hd_train):
                                if train_labels[i] != 0:
                                    active_idx = np.flatnonzero(row)
                                    to_flip = active_idx[rng.random(active_idx.size) < p]
                                    W_out[int(train_labels[i])-1, to_flip] = 1./k

                            # Frame-based inference on train
                            z_out_train = z_hd_train @ W_out.T
                            train_preds = np.argmax(z_out_train, axis=1) + 1  # 1-indexed
                            train_labeled = train_labels > 0
                            train_acc = metrics.accuracy_score(train_labels[train_labeled], train_preds[train_labeled])

                            # Frame-based inference on test
                            z_out_test = z_hd_test @ W_out.T
                            test_preds = np.argmax(z_out_test, axis=1) + 1  # 1-indexed
                            test_labeled = test_labels > 0
                            test_acc = metrics.accuracy_score(test_labels[test_labeled], test_preds[test_labeled])

                            results['train_acc'].append(train_acc)
                            results['test_acc'].append(test_acc)
                            results['y_pred'].append(test_preds[test_labeled].copy())
                            results['y_true'].append(test_labels[test_labeled].copy())

                            params['expansion'].append(exp_name)
                            params['p_hd'].append(p_hd)
                            params['d'].append(d)
                            params['p'].append(p)
                            params['normalized'].append(normalized)
                            params['kernel'].append(kernel)
                            params['n_fold'].append(n_fold)

                            print(f'[{exp_name}] p_hd: {p_hd}, d: {d}, k: {k}, p: {p}, '
                                  f'normalized: {normalized}, kernel: {kernel}, n_fold: {n_fold}')
                            print(f'  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

# Convert to arrays
for key in params.keys():
    params[key] = np.array(params[key])
for k in ['train_acc', 'test_acc']:
    results[k] = np.array(results[k])
results['y_pred'] = np.array(results['y_pred'], dtype=object)
results['y_true'] = np.array(results['y_true'], dtype=object)

data = {'params': params, 'results': results, 'pair_log': pair_log,
        'config_specs': dict(CONFIG_SPECS)}

with open(f'data/{file_save}.pkl', 'wb') as f:
    pickle.dump(data, f)
