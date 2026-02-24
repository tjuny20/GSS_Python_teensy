import numpy as np
from tools import load, split, estimate_derivative, train, test
import pickle


n_hd = 10000
n_out = 6
# n_training = [3, 9, 15, 30]
files = ['mix_100_20_1']
file_save = 'gs_binary'

# for n in n_training:
for file in files:
    file_train = file
    file_test = 'mix_50_20_1'

    grid_s = np.arange(0.005, 0.105, 0.005)
    grid_p = np.arange(0.005, 0.055, 0.005)
    grid_p_hd = [0.05]
    grid_normalized = [False, True]
    grid_n_fold = 10



    params = {'p_hd': [], 's': [], 'p': [], 'normalized': [], 'n_fold': []}
    results = {'train_acc': [], 'test_acc': []}

    for p_hd in grid_p_hd:
        for s in grid_s:
            for p in grid_p:
                for normalized in grid_normalized:
                    for n_fold in range(grid_n_fold):
                        k = int(s * n_hd)

                        for key in params.keys():
                            params[key].append(locals()[key])

                        train_data, train_sequence, train_times_sec, train_sequence_sec = load(file_train, reduced=True)
                        train_d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=train_data)
                        train_data = np.hstack((train_data, train_d_sensor_data))

                        test_data, test_sequence, test_times_sec, test_sequence_sec = load(file_test, reduced=True)
                        test_d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=test_data)
                        test_data = np.hstack((test_data, test_d_sensor_data))

                        if normalized:
                            train_min = np.min(train_data, axis=0, keepdims=True)
                            train_max = np.max(train_data, axis=0, keepdims=True)
                            train_denom = train_max - train_min
                            train_denom[train_denom == 0] = 1.
                            train_data_proc = (train_data - train_min) / train_denom

                            test_min = np.min(test_data, axis=0, keepdims=True)
                            test_max = np.max(test_data, axis=0, keepdims=True)
                            test_denom = test_max - test_min
                            test_denom[test_denom == 0] = 1.
                            test_data_proc = (test_data - test_min) / test_denom
                        else:
                            train_data_proc = train_data
                            test_data_proc = test_data

                        W_hd, W_out = train(train_data_proc, train_sequence, train_times_sec, train_sequence_sec,
                                            n_hd=n_hd, n_out=n_out, k=k, p=p, p_hd=p_hd,
                                            normalized=False, whitened=False)

                        train_acc = test(train_data_proc, train_sequence, train_times_sec, train_sequence_sec,
                                         W_hd, W_out,
                                         n_hd=n_hd, n_out=n_out, k=k, integration_delay=0.,
                                         normalized=False, whitened=False)

                        test_acc = test(test_data_proc, test_sequence, test_times_sec, test_sequence_sec,
                                        W_hd, W_out,
                                        n_hd=n_hd, n_out=n_out, k=k, integration_delay=0.,
                                        normalized=False, whitened=False)

                        results['train_acc'].append(train_acc)
                        results['test_acc'].append(test_acc)

                        print(f'p_hd: {p_hd}, s: {s}, k: {k}, p: {p}, normalized: {normalized}, n_fold: {n_fold}')
                        print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n')

    for key in params.keys():
        params[key] = np.array(params[key])
    for key in results.keys():
        results[key] = np.array(results[key])

    data = {'params': params, 'results': results}

    with open(f'data/{file_save}.pkl', 'wb') as f:
        pickle.dump(data, f)
