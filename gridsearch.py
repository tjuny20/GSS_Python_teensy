import numpy as np
from tools import load, split, estimate_derivative, train, test
import pickle
from sklearn import metrics


n_hd = 10000
n_out = 3
k = 50
n_pot = 10
n_train = 225
w_teacher = 1.
t_training_delay = 5.
filename = '1_600_20'

grid_uniformW = [False]
grid_normalized = [False]
grid_whitened = [False]
grid_k = [10, 15, 20, 25]
grid_n_pot = [n for n in range(2, 20, 2)]
grid_t_training_delay = [n for n in range(0, 20, 5)]
grid_n_fold = 5


sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=sensor_data)
sensor_data = np.hstack((sensor_data, d_sensor_data))


params = {'uniformW': [], 'normalized': [], 'whitened': [], 'k': [], 'n_pot': [], 't_training_delay': [], 'n_fold': []}
results = {'train_acc': [], 'test_acc': [], 'y_pred': [], 'y_true': []}
for uniformW in grid_uniformW:
    for normalized in grid_normalized:
        for whitened in grid_whitened:
            for k in grid_k:
                for n_pot in grid_n_pot:
                    for t_training_delay in grid_t_training_delay:
                        for n_fold in range(grid_n_fold):
                            for key in params.keys():
                                params[key].append(locals()[key])

                            if normalized:
                                sensor_data_norm = (sensor_data - np.mean(sensor_data, axis=0))/ np.std(sensor_data, axis=0)
                            else:
                                sensor_data_norm = sensor_data

                            # x_dense, _, _, _ = adap_whitening_2(sensor_data_norm)
                            if whitened:
                                x_dense = adap_whitening_2(sensor_data_norm)
                            else:
                                x_dense = sensor_data_norm

                            n_dense = x_dense.shape[1]

                            labels = np.zeros_like(times_sec)
                            for i, t in enumerate(sequence_sec[:n_train]):
                                try:
                                    flag = (times_sec > sequence_sec[i] + t_training_delay) & (times_sec < sequence_sec[i+1])
                                except IndexError:
                                    flag = (times_sec > sequence_sec[i] + t_training_delay)
                                labels[flag] = int(sequence[i][1])

                            idx_last_flag = np.where(labels != 0)[0][-1]

                            if uniformW:
                                W_hd = np.random.uniform(high=1/np.sqrt(n_dense), size=(n_hd, n_dense))  #Test random sparse weights
                            else:
                                W_hd = np.random.binomial(n=1, p=0.05, size=(n_hd, n_dense))  #Test random sparse weights
                            x_hd = x_dense @ W_hd.T
                            ranks = np.argsort(np.argsort(-x_hd, axis=1), axis=1)
                            z_hd = np.where(ranks < k, 1., 0.)
                            W_out = np.zeros((n_out, n_hd))
                            W = np.zeros((n_out, n_hd))

                            z_out_train = np.zeros((z_hd.shape[0],  n_out))
                            for i, row in enumerate(z_hd[:idx_last_flag]):
                                teacher = np.zeros((n_out,))
                                if labels[i] != 0:
                                    teacher[int(labels[i]-1)] = w_teacher
                                out = row @ W_out.T + teacher
                                z_out_train[i] = out
                                dW = (1./n_pot)*(np.atleast_2d(out).T @ np.atleast_2d(row))
                                W += dW
                                W_out = np.where(W>=1., 1./k, 0.)
                                # if i%100 == 0:
                                #     print(np.sum(W_out, axis=1))

                            z_out = np.zeros((z_hd.shape[0],  n_out))
                            for i, row in enumerate(z_hd):
                                out = row @ W_out.T
                                z_out[i] = out

                            ranks_out = np.argsort(np.argsort(-z_out, axis=1), axis=1)
                            z_wta = np.where(ranks_out < 1, 1., 0.)

                            z_pred = np.zeros_like(sequence_sec)
                            z_true = np.zeros_like(sequence_sec)
                            for i, t in enumerate(sequence_sec):
                                try:
                                    flag = (times_sec > sequence_sec[i] + t_training_delay) & (times_sec < sequence_sec[i+1])
                                except IndexError:
                                    flag = (times_sec > sequence_sec[i] + t_training_delay)
                                z_pred[i] = np.argsort(np.sum(z_out[flag], axis=0))[-1] + 1
                                z_true[i] = sequence[i][1]

                            train_acc = metrics.accuracy_score(z_true[:n_train], z_pred[:n_train])
                            test_acc = metrics.accuracy_score(z_true[n_train:], z_pred[n_train:])
                            results['train_acc'].append(train_acc)
                            results['test_acc'].append(test_acc)
                            results['y_pred'].append(z_pred.copy())
                            results['y_true'].append(z_true.copy())

                            print(f'UniformW: {uniformW}, Normalized: {normalized}, Whitened: {whitened}, k: {k}, n_pot: {n_pot}, t_training_delay: {t_training_delay}, n_fold: {n_fold}')
                            print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

for key in params.keys():
    params[key] = np.array(params[key])
for k in ['train_acc', 'test_acc']:
    results[k] = np.array(results[k])
# keep y_pred_all / y_true_all as lists-of-arrays (ragged OK) or cast to object arrays:
results['y_pred'] = np.array(results['y_pred'], dtype=object)
results['y_true'] = np.array(results['y_true'], dtype=object)

data = {'params': params, 'results': results}

with open('data/gridsearch_full.pkl', 'wb') as f:
    pickle.dump(data, f)