import numpy as np
import pickle
import csv
import torch
from torch.linalg import inv, eig, pinv
from matplotlib import pyplot as plt
from tools import whiten, adap_whitening, adap_whitening_2, estimate_derivative
from sklearn import svm, metrics
from sklearn.decomposition import PCA
from datetime import datetime
import sklearn
import pickle
from scipy.ndimage import gaussian_filter1d
from tools import load, split, pseudoderivative, estimate_derivative, plot_two_intervals, get_samples

filename = '1_600_20'
rng = np.random.default_rng(42)  # for reproducibility

sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=sensor_data)
sensor_data = np.hstack((sensor_data, d_sensor_data))

n_hd = 10000
n_out = 3
k = 15
p = 0.7
n_train = 450

grid_k = np.arange(10,25,5)
grid_p = np.arange(0.1,1.1,0.1)
grid_n_fold = 5
n_samples = np.arange(1, 11)

params = {'k': [], 'p': [], 'n_fold': [], 'n_sample': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}

x_dense = sensor_data
n_dense = x_dense.shape[1]

labels = np.zeros_like(times_sec)
for i, t in enumerate(sequence_sec[:n_train]):
    try:
        flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
    except IndexError:
        flag = (times_sec > sequence_sec[i])
    labels[flag] = int(sequence[i][1])

idx_last_flag = np.where(labels != 0)[0][-1]
for k in grid_k:
    for p in grid_p:
        for n_sample in n_samples:
            for n_fold in range(grid_n_fold):
                W_hd = np.random.binomial(n=1, p=0.05, size=(n_hd, n_dense))  #Test random sparse weights
                x_hd = x_dense @ W_hd.T
                z_hd = np.where(np.argsort(x_hd)<k, 1., 0)
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

                z_wta = np.where(np.argsort(z_out_acc, axis=1)<1, 1., 0)

                z_pred = np.zeros_like(sequence_sec)
                z_true = np.zeros_like(sequence_sec)
                for i, t in enumerate(sequence_sec):
                    try:
                        flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i+1])
                    except IndexError:
                        flag = (times_sec > sequence_sec[i])
                    try:
                        z_pred[i] = np.argsort(np.sum(z_out_acc[flag][:n_sample], axis=0))[-1] + 1
                    except IndexError:
                        z_pred[i] = 0
                        print(f'IndexError at i={i}, t={n_sample}')
                    z_true[i] = sequence[i][1]

                train_acc = sklearn.metrics.accuracy_score(z_true[:n_train], z_pred[:n_train])
                test_acc = sklearn.metrics.accuracy_score(z_true[n_train:], z_pred[n_train:])
                test_acc = metrics.accuracy_score(z_true[n_train:], z_pred[n_train:])

                for key in params.keys():
                    params[key].append(locals()[key])
                results['train_acc'].append(train_acc)
                results['test_acc'].append(test_acc)
                results['y_pred'].append(z_pred.copy())
                results['y_true'].append(z_true.copy())

                print(f'k: {k}, p: {p}, n_sample: {n_sample}, n_fold: {n_fold}')
                print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

for key in params.keys():
    params[key] = np.array(params[key])
for k in ['train_acc', 'test_acc']:
    results[k] = np.array(results[k])
# keep y_pred / y_true as lists-of-arrays (ragged OK) or cast to object arrays:
results['y_pred'] = np.array(results['y_pred'], dtype=object)
results['y_true'] = np.array(results['y_true'], dtype=object)

data = {'params': params, 'results': results}

with open('data/gridsearch_rapid_inference_short.pkl', 'wb') as f:
    pickle.dump(data, f)
