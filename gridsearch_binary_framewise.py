import numpy as np
from sklearn import metrics
import pickle
from math import comb

from tools import load, normalize, whiten


rng = np.random.default_rng(42)  # for reproducibility

n_hd = 10000
n_out = 6

file_train = 'mix_100_20_1'
file_test = 'mix_50_20_1'
file_save = 'gs_binary_framewise'

grid_p = [0.01]
grid_normalized = ['raw']
grid_kernel = ['top']
grid_n_fold = 5
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


# Pre-generate pair pools (ordered pairs, fixed seed)
rng_pairs = np.random.default_rng(0)
all_ordered_pairs = [(i, j) for i in range(n_sensors) for j in range(n_sensors) if i != j]
ratio_pairs = [all_ordered_pairs[i] for i in rng_pairs.permutation(len(all_ordered_pairs))]
diff_pairs  = [all_ordered_pairs[i] for i in rng_pairs.permutation(len(all_ordered_pairs))]


def build_expansion_configs(sensor_data, h):
    """Build the 4 32-trace configs (excluding 12R + 12D) for a given dataset."""
    x_d1   = expand_with_derivatives(sensor_data, h, max_order=1)   # 16 cols
    x_d12  = expand_with_derivatives(sensor_data, h, max_order=2)   # 24 cols
    x_d123 = expand_with_derivatives(sensor_data, h, max_order=3)   # 32 cols

    r8 = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8) for i, j in ratio_pairs[:8]])
    d8 = np.column_stack([sensor_data[:, i] - sensor_data[:, j] for i, j in diff_pairs[:8]])

    return {
        '∂¹+∂²+∂³':     x_d123,
        '∂¹+∂² + 8R':   np.hstack([x_d12, r8]),
        '∂¹+∂² + 8D':   np.hstack([x_d12, d8]),
        '∂¹ + 8R + 8D':  np.hstack([x_d1, r8, d8]),
    }


train_configs = build_expansion_configs(train_data_raw, h_train)
test_configs  = build_expansion_configs(test_data_raw, h_test)


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

for exp_name in train_configs.keys():
    train_expanded = train_configs[exp_name]
    test_expanded  = test_configs[exp_name]

    for kernel in grid_kernel:
        for p_hd in grid_p_hd:
            for d in grid_d:
                for p in grid_p:
                    for normalized in grid_normalized:
                        for n_fold in range(grid_n_fold):
                            k = int(d * n_hd)

                            # Normalization (fit on train, apply to both)
                            if normalized == 'min-max':
                                col_min = np.min(train_expanded, axis=0, keepdims=True)
                                col_max = np.max(train_expanded, axis=0, keepdims=True)
                                denom = col_max - col_min
                                denom[denom == 0] = 1.
                                x_train = (train_expanded - col_min) / denom
                                x_test  = (test_expanded - col_min) / denom
                            elif normalized == 'normalized':
                                x_train = normalize(train_expanded)
                                x_test  = normalize(test_expanded)
                            elif normalized == 'whitened':
                                x_train = whiten(train_expanded)
                                x_test  = whiten(test_expanded)
                            else:
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

data = {'params': params, 'results': results}

with open(f'data/{file_save}.pkl', 'wb') as f:
    pickle.dump(data, f)
