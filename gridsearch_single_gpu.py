import numpy as np
from sklearn import metrics
import pickle
import torch

from tools import load, estimate_derivative, pseudoderivative, normalize, whiten


device = torch.device('mps' if torch.backends.mps.is_available() else
                       'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

rng = np.random.default_rng(42)

n_hd = 10000
n_out = 3

n_train = 450
filename = '1_600_20'
file_save = 'gs_single_gpu'

grid_s = np.arange(0.01, 0.11, 0.01)
grid_p = 1 / np.concatenate([np.ones(1), np.arange(5, 105, 5)])
grid_p_hd = [0.05]
# grid_normalized = ['raw', 'min-max', 'normalized', 'whitened']
# grid_kernel = ['top', 'rank']
grid_normalized = ['raw']
grid_kernel = ['top']
grid_n_fold = 10


sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=sensor_data)
sensor_data = np.hstack((sensor_data, d_sensor_data))

params = {'p_hd': [], 's': [], 'p': [], 'normalized': [], 'kernel': [], 'n_fold': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}

for kernel in grid_kernel:
    for p_hd in grid_p_hd:
        for s in grid_s:
            for p in grid_p:
                for normalized in grid_normalized:
                    for n_fold in range(grid_n_fold):
                        k = int(s * n_hd)
                        for key in params.keys():
                            params[key].append(locals()[key])

                        labels = np.zeros_like(times_sec)
                        for i, t in enumerate(sequence_sec[:n_train]):
                            try:
                                flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                            except IndexError:
                                flag = (times_sec > sequence_sec[i])
                            labels[flag] = int(sequence[i][1])

                        idx_last_flag = np.where(labels != 0)[0][-1]

                        if normalized == 'min-max':
                            col_min = np.min(sensor_data, axis=0, keepdims=True)
                            col_max = np.max(sensor_data, axis=0, keepdims=True)
                            denom = col_max - col_min
                            denom[denom == 0] = 1.
                            x_dense = (sensor_data - col_min) / denom
                        elif normalized == 'normalized':
                            x_dense = normalize(sensor_data)
                        elif normalized == 'whitened':
                            x_dense = whiten(sensor_data)
                        else:
                            x_dense = sensor_data

                        n_dense = x_dense.shape[1]

                        # --- GPU path ---
                        x_dense_t = torch.tensor(x_dense, dtype=torch.float32, device=device)

                        W_hd_t = torch.bernoulli(torch.full((n_hd, n_dense), p_hd, device=device))
                        x_hd = x_dense_t @ W_hd_t.T

                        if kernel == 'top':
                            _, topk_idx = torch.topk(x_hd, k, dim=1, largest=True)
                            z_hd = torch.zeros_like(x_hd)
                            z_hd.scatter_(1, topk_idx, 1.0)
                        elif kernel == 'rank':
                            z_hd = torch.where(torch.argsort(x_hd) < k, 1., 0.)

                        # Training: build W_out on GPU
                        W_out = torch.zeros((n_out, n_hd), dtype=torch.float32, device=device)
                        labels_t = torch.tensor(labels[:idx_last_flag], dtype=torch.long, device=device)

                        for i in range(idx_last_flag):
                            if labels_t[i] != 0:
                                row = z_hd[i]
                                active_idx = torch.nonzero(row, as_tuple=True)[0]
                                mask = torch.rand(active_idx.shape[0], device=device) < p
                                to_flip = active_idx[mask]
                                W_out[labels_t[i] - 1, to_flip] = 1. / k

                        # Inference on GPU (all samples at once)
                        z_out_acc = z_hd @ W_out.T  # (n_samples, n_out)

                        # Move to CPU for per-sequence accuracy computation
                        z_out_np = z_out_acc.cpu().numpy()

                        z_pred = np.zeros_like(sequence_sec)
                        z_true = np.zeros_like(sequence_sec)
                        for i, t in enumerate(sequence_sec):
                            try:
                                flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                            except IndexError:
                                flag = (times_sec > sequence_sec[i])
                            z_pred[i] = np.argsort(np.sum(z_out_np[flag], axis=0))[-1] + 1
                            z_true[i] = sequence[i][1]

                        train_acc = metrics.accuracy_score(z_true[:n_train], z_pred[:n_train])
                        test_acc = metrics.accuracy_score(z_true[n_train:], z_pred[n_train:])
                        results['train_acc'].append(train_acc)
                        results['test_acc'].append(test_acc)
                        results['y_pred'].append(z_pred.copy())
                        results['y_true'].append(z_true.copy())

                        print(f'p_hd: {p_hd}, s: {s}, k: {k}, p: {p}, normalized: {normalized}, kernel: {kernel}, n_fold: {n_fold}')
                        print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

for key in params.keys():
    params[key] = np.array(params[key])
for k in ['train_acc', 'test_acc']:
    results[k] = np.array(results[k])
results['y_pred'] = np.array(results['y_pred'], dtype=object)
results['y_true'] = np.array(results['y_true'], dtype=object)

data = {'params': params, 'results': results}

with open(f'data/{file_save}.pkl', 'wb') as f:
    pickle.dump(data, f)
