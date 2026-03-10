import numpy as np
from sklearn import metrics
import pickle
from math import comb

from tools import load, normalize, whiten


rng = np.random.default_rng(42)  # for reproducibility

n_hd = 10000
n_out = 3

n_train = 450
filename = '1_600_20'
file_save = 'gs_single_framewise'

grid_p = [0.01]
grid_normalized = ['raw', 'whitened']
grid_kernel = ['rank']
grid_n_fold = 5
grid_s = np.concatenate([np.arange(0.01, 0.05, 0.02), np.arange(0.1, 0.5, 0.1)])
grid_p_hd = np.arange(0.05, 0.50, 0.1)


# --- Load data ---
sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
h = np.median(np.diff(times_sec))
n_sensors = sensor_data.shape[1]


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


# --- Build 32-trace expansion configs (excluding 12R + 12D) ---
x_d1   = expand_with_derivatives(sensor_data, h, max_order=1)   # 16 cols
x_d12  = expand_with_derivatives(sensor_data, h, max_order=2)   # 24 cols
x_d123 = expand_with_derivatives(sensor_data, h, max_order=3)   # 32 cols

r8 = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8) for i, j in ratio_pairs[:8]])
d8 = np.column_stack([sensor_data[:, i] - sensor_data[:, j] for i, j in diff_pairs[:8]])

expansion_configs = {
    '∂¹+∂²+∂³':     x_d123,
    '∂¹+∂² + 8R':   np.hstack([x_d12, r8]),
    '∂¹+∂² + 8D':   np.hstack([x_d12, d8]),
    '∂¹ + 8R + 8D':  np.hstack([x_d1, r8, d8]),
}


# --- Grid search ---
params = {'expansion': [], 'p_hd': [], 's': [], 'p': [], 'normalized': [], 'kernel': [], 'n_fold': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}

for exp_name, expanded_data in expansion_configs.items():
    for kernel in grid_kernel:
        for p_hd in grid_p_hd:
            for s in grid_s:
                for p in grid_p:
                    for normalized in grid_normalized:
                        for n_fold in range(grid_n_fold):
                            k = int(s * n_hd)

                            # Build train labels (only first n_train sequences)
                            labels = np.zeros_like(times_sec)
                            for i, t in enumerate(sequence_sec[:n_train]):
                                try:
                                    flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                                except IndexError:
                                    flag = (times_sec > sequence_sec[i])
                                labels[flag] = int(sequence[i][1])

                            idx_last_flag = np.where(labels != 0)[0][-1]

                            # Normalization
                            if normalized == 'min-max':
                                col_min = np.min(expanded_data, axis=0, keepdims=True)
                                col_max = np.max(expanded_data, axis=0, keepdims=True)
                                denom = col_max - col_min
                                denom[denom == 0] = 1.
                                x_dense = (expanded_data - col_min) / denom
                            elif normalized == 'normalized':
                                x_dense = normalize(expanded_data)
                            elif normalized == 'whitened':
                                x_dense = whiten(expanded_data)
                            else:
                                x_dense = expanded_data

                            n_dense = x_dense.shape[1]

                            # HD projection
                            W_hd = np.random.binomial(n=1, p=p_hd, size=(n_hd, n_dense))
                            x_hd = x_dense @ W_hd.T

                            # Sparsification
                            if kernel == 'top':
                                ranks = np.argsort(np.argsort(-x_hd, axis=1), axis=1)
                                z_hd = np.where(ranks < k, 1., 0.)
                            elif kernel == 'rank':
                                z_hd = np.where(np.argsort(x_hd) < k, 1., 0)

                            # Train readout weights
                            W_out = np.zeros((n_out, n_hd))
                            for i, row in enumerate(z_hd[:idx_last_flag]):
                                if labels[i] != 0:
                                    active_idx = np.flatnonzero(row)
                                    to_flip = active_idx[rng.random(active_idx.size) < p]
                                    W_out[int(labels[i])-1, to_flip] = 1./k

                            # Compute output activities for all frames
                            z_out = z_hd @ W_out.T  # (n_samples, n_out)

                            # Build full frame-level ground truth
                            frame_labels = np.zeros_like(times_sec)
                            for i, t in enumerate(sequence_sec):
                                try:
                                    flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                                except IndexError:
                                    flag = (times_sec > sequence_sec[i])
                                frame_labels[flag] = int(sequence[i][1])

                            # Frame-based inference: prediction = most active output neuron per frame
                            frame_preds = np.argmax(z_out, axis=1) + 1  # 1-indexed

                            # Only evaluate on labeled frames
                            labeled_mask = frame_labels > 0

                            # Split train/test by sequence boundary
                            train_time_end = sequence_sec[n_train] if n_train < len(sequence_sec) else times_sec[-1] + 1
                            train_mask = labeled_mask & (times_sec < train_time_end)
                            test_mask  = labeled_mask & (times_sec >= train_time_end)

                            train_acc = metrics.accuracy_score(frame_labels[train_mask], frame_preds[train_mask])
                            test_acc  = metrics.accuracy_score(frame_labels[test_mask], frame_preds[test_mask])

                            results['train_acc'].append(train_acc)
                            results['test_acc'].append(test_acc)
                            results['y_pred'].append(frame_preds[test_mask].copy())
                            results['y_true'].append(frame_labels[test_mask].copy())

                            params['expansion'].append(exp_name)
                            params['p_hd'].append(p_hd)
                            params['s'].append(s)
                            params['p'].append(p)
                            params['normalized'].append(normalized)
                            params['kernel'].append(kernel)
                            params['n_fold'].append(n_fold)

                            print(f'[{exp_name}] p_hd: {p_hd:.2f}, s: {s}, k: {k}, p: {p}, '
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
