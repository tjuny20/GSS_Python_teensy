import numpy as np
from sklearn import metrics
import pickle

from tools import load, estimate_derivative, normalize, whiten


rng = np.random.default_rng(42)  # for reproducibility

n_hd = 10000
n_out = 6

file_train = 'mix_100_20_1'
file_test = 'mix_50_20_1'
file_save = 'gs_binary'

grid_s = np.arange(0.01, 0.11, 0.01)
grid_p = 1/ np.concatenate([np.ones(1), np.arange(5,105,5)])
grid_p_hd = [0.05]
grid_normalized = ['raw', 'min-max', 'normalized', 'whitened']
grid_kernel = ['top', 'rank']
grid_n_fold = 10


train_data, train_sequence, train_times_sec, train_sequence_sec = load(file_train, reduced=True)
train_d = np.apply_along_axis(estimate_derivative, axis=0, arr=train_data)
train_data = np.hstack((train_data, train_d))

test_data, test_sequence, test_times_sec, test_sequence_sec = load(file_test, reduced=True)
test_d = np.apply_along_axis(estimate_derivative, axis=0, arr=test_data)
test_data = np.hstack((test_data, test_d))

# Build train labels
train_labels = np.zeros_like(train_times_sec)
for i, t in enumerate(train_sequence_sec):
    try:
        flag = (train_times_sec > train_sequence_sec[i]) & (train_times_sec < train_sequence_sec[i+1])
    except IndexError:
        flag = (train_times_sec > train_sequence_sec[i])
    train_labels[flag] = int(train_sequence[i][1])

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

                        # Normalization
                        if normalized=='min-max':
                            col_min = np.min(train_data, axis=0, keepdims=True)
                            col_max = np.max(train_data, axis=0, keepdims=True)
                            denom = col_max - col_min
                            denom[denom == 0] = 1.
                            x_train = (train_data - col_min) / denom
                            x_test = (test_data - col_min) / denom
                        elif normalized=='normalized':
                            x_train = normalize(train_data)
                            x_test = normalize(test_data)
                        elif normalized=='whitened':
                            x_train = whiten(train_data)
                            x_test = whiten(test_data)
                        else:
                            x_train = train_data
                            x_test = test_data

                        n_dense = x_train.shape[1]

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

                        # Evaluate on train
                        z_out_train = z_hd_train @ W_out.T

                        z_pred_train = np.zeros_like(train_sequence_sec)
                        z_true_train = np.zeros_like(train_sequence_sec)
                        for i, t in enumerate(train_sequence_sec):
                            try:
                                flag = (train_times_sec > train_sequence_sec[i]) & (train_times_sec < train_sequence_sec[i+1])
                            except IndexError:
                                flag = (train_times_sec > train_sequence_sec[i])
                            z_pred_train[i] = np.argsort(np.sum(z_out_train[flag], axis=0))[-1] + 1
                            z_true_train[i] = train_sequence[i][1]

                        train_acc = metrics.accuracy_score(z_true_train, z_pred_train)

                        # Evaluate on test
                        z_out_test = z_hd_test @ W_out.T

                        z_pred_test = np.zeros_like(test_sequence_sec)
                        z_true_test = np.zeros_like(test_sequence_sec)
                        for i, t in enumerate(test_sequence_sec):
                            try:
                                flag = (test_times_sec > test_sequence_sec[i]) & (test_times_sec < test_sequence_sec[i+1])
                            except IndexError:
                                flag = (test_times_sec > test_sequence_sec[i])
                            z_pred_test[i] = np.argsort(np.sum(z_out_test[flag], axis=0))[-1] + 1
                            z_true_test[i] = test_sequence[i][1]

                        test_acc = metrics.accuracy_score(z_true_test, z_pred_test)

                        results['train_acc'].append(train_acc)
                        results['test_acc'].append(test_acc)
                        results['y_pred'].append(z_pred_test.copy())
                        results['y_true'].append(z_true_test.copy())

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
