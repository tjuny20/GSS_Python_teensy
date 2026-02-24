import numpy as np
from sklearn import metrics
import pickle

from tools import load, estimate_derivative, pseudoderivative
from matplotlib import font_manager as fm, rcParams


rng = np.random.default_rng(42)  # for reproducibility

n_hd = 10000
n_out = 3

n_train = 450
filename = '1_600_20'
file_save = 'gs_single'

grid_s = np.arange(0.005, 0.105, 0.005)
grid_p = np.arange(0.005, 0.055, 0.005)
grid_p_hd = [0.05]
grid_normalized = [False, True]
grid_n_fold = 10


sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=sensor_data)
sensor_data = np.hstack((sensor_data, d_sensor_data))

# baseline = np.mean(sensor_data[:300], axis=0)  # Add baseline substraction
# sensor_data = (sensor_data - baseline)

params = {'p_hd': [], 's': [], 'p': [], 'normalized': [], 'n_fold': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}


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

                    if normalized:
                        col_min = np.min(sensor_data, axis=0, keepdims=True)
                        col_max = np.max(sensor_data, axis=0, keepdims=True)
                        denom = col_max - col_min
                        denom[denom == 0] = 1.
                        x_dense = (sensor_data - col_min) / denom
                    else:
                        x_dense = sensor_data

                    n_dense = x_dense.shape[1]

                    W_hd = np.random.binomial(n=1, p=p_hd, size=(n_hd, n_dense))  #Test random sparse weights
                    x_hd = x_dense @ W_hd.T
                    ranks = np.argsort(np.argsort(-x_hd, axis=1), axis=1)
                    z_hd = np.where(ranks < k, 1., 0.)
                    W_out = np.zeros((n_out, n_hd))
                    W = np.zeros((n_out, n_hd))

                    z_out_train = np.zeros((z_hd.shape[0],  n_out))
                    for i, row in enumerate(z_hd[:idx_last_flag]):
                        if labels[i] != 0:
                            active_idx = np.flatnonzero(row)
                            to_flip = active_idx[rng.random(active_idx.size) < p]     # Bernoulli(p) per active index# indices where z_hd==1
                            W_out[int(labels[i])-1, to_flip] = 1./k


                        out = row @ W_out.T
                        z_out_train[i] = out

                    z_out_acc = np.zeros((z_hd.shape[0],  n_out))
                    for i, row in enumerate(z_hd):
                        out = row @ W_out.T
                        z_out_acc[i] = out

                    ranks_out = np.argsort(np.argsort(-z_out_acc, axis=1), axis=1)
                    z_wta = np.where(ranks_out < 1, 1., 0.)

                    z_pred = np.zeros_like(sequence_sec)
                    z_true = np.zeros_like(sequence_sec)
                    for i, t in enumerate(sequence_sec):
                        try:
                            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                        except IndexError:
                            flag = (times_sec > sequence_sec[i])
                        z_pred[i] = np.argsort(np.sum(z_out_acc[flag], axis=0))[-1] + 1
                        z_true[i] = sequence[i][1]

                    train_acc = metrics.accuracy_score(z_true[:n_train], z_pred[:n_train])
                    test_acc = metrics.accuracy_score(z_true[n_train:], z_pred[n_train:])
                    results['train_acc'].append(train_acc)
                    results['test_acc'].append(test_acc)
                    results['y_pred'].append(z_pred.copy())
                    results['y_true'].append(z_true.copy())

                    print(f'p_hd: {p_hd}, s: {s}, k: {k}, p: {p}, normalized: {normalized}, n_fold: {n_fold}')
                    print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

for key in params.keys():
    params[key] = np.array(params[key])
for k in ['train_acc', 'test_acc']:
    results[k] = np.array(results[k])
# keep y_pred / y_true as lists-of-arrays (ragged OK) or cast to object arrays:
results['y_pred'] = np.array(results['y_pred'], dtype=object)
results['y_true'] = np.array(results['y_true'], dtype=object)

data = {'params': params, 'results': results}

with open(f'data/{file_save}.pkl', 'wb') as f:
    pickle.dump(data, f)
