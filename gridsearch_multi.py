import numpy as np
from tools import load, split, estimate_derivative, train, test
import pickle


n_hd = 10000
n_out = 3
n_training = [3, 9, 15, 30]


for n in n_training:
    file = f'1_{n}_20'

    grid_k = np.arange(5, 55, 5)
    grid_p = np.arange(0.1, 1.1, 0.1)
    grid_n_fold = 5


    params = {'k': [], 'p': [], 'n_fold': []}
    results = {'train_acc': [], 'test_acc': []}

    for k in grid_k:
        for p in grid_p:
            for n_fold in range(grid_n_fold):

                for key in params.keys():
                    params[key].append(locals()[key])

                train_data, train_sequence, train_times_sec, train_sequence_sec = load(file, reduced=True)
                train_d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=train_data)
                train_data = np.hstack((train_data, train_d_sensor_data))

                data, sequence, times_sec, sequence_sec = load('1_600_20', reduced=True)
                d_sensor_data = np.apply_along_axis(estimate_derivative, axis=0, arr=data)
                data_ = np.hstack((data, d_sensor_data))
                _, _, _, _, \
                    test_data, test_sequence, test_times_sec, test_sequence_sec = split(data_, sequence, times_sec,
                                                                                        sequence_sec, idx_split=450)

                W_hd, W_out = train(train_data, train_sequence, train_times_sec, train_sequence_sec,
                                    n_hd=n_hd, n_out=n_out, k=k, p=p,
                                    normalized=False, whitened=False)

                train_acc = test(train_data, train_sequence, train_times_sec, train_sequence_sec,
                                 W_hd, W_out,
                                 n_hd=n_hd, n_out=n_out, k=k, integration_delay=0.,
                                 normalized=False, whitened=False)

                test_acc = test(test_data, test_sequence, test_times_sec, test_sequence_sec,
                                W_hd, W_out,
                                n_hd=n_hd, n_out=n_out, k=k, integration_delay=0.,
                                normalized=False, whitened=False)

                results['train_acc'].append(train_acc)
                results['test_acc'].append(test_acc)

                print(f'k: {k}, p: {p}, n_fold: {n_fold}')
                print(f'Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}\n')

    for key in params.keys():
        params[key] = np.array(params[key])
    for key in results.keys():
        results[key] = np.array(results[key])

    data = {'params': params, 'results': results}

    with open(f'data/gridsearch_{n}.pkl', 'wb') as f:
        pickle.dump(data, f)