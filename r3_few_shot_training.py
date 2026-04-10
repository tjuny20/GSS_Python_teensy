"""Stand-alone training script for the R3 few-shot learning notebook.

Runs the expand-and-sparsify few-shot learning loop (plus an RBF SVM
baseline on the same training data) on the single-gases and binary-mixture
datasets, then pickles the results to ``data/r3_few_shot_results.pkl``.

Usage
-----
    python r3_few_shot_training.py
"""

import os
import pickle
from math import comb

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from tools import load


# ------------------------------------------------------------------ #
# Parameters                                                         #
# ------------------------------------------------------------------ #
N_HD = 10000
P_HD = 0.025
D = 0.5
P = 0.01
NORMALIZED = 'raw'
N_TIMES = 10

OUT_PATH = 'data/r3_few_shot_results.pkl'


# ------------------------------------------------------------------ #
# Expansion helpers                                                  #
# ------------------------------------------------------------------ #
def backward_diff_array(y, h, n):
    """n-th backward finite difference (zero-padded to preserve length)."""
    coeffs = np.array([(-1) ** k * comb(n, k) for k in range(n + 1)])
    raw = np.convolve(y, coeffs, mode='valid') / h ** n
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


def build_152_expansion(sensor_data, h, ratio_pairs, diff_pairs):
    """8 raw + 8 d1 + 8 d2 + 64 diff + 64 ratio = 152 traces."""
    x_d12 = expand_with_derivatives(sensor_data, h, max_order=2)
    r64 = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8)
                           for i, j in ratio_pairs[:64]])
    d64 = np.column_stack([sensor_data[:, i] - sensor_data[:, j]
                           for i, j in diff_pairs[:64]])
    return np.hstack([x_d12, d64, r64])


# ------------------------------------------------------------------ #
# Pipeline functions                                                 #
# ------------------------------------------------------------------ #
def expand_and_sparsify(x_dense, n_hd, p_hd, d, rng_proj):
    """Project to HD space and sparsify (top-k)."""
    k = int(d * n_hd)
    n_dense = x_dense.shape[1]
    W_hd = rng_proj.binomial(n=1, p=p_hd, size=(n_hd, n_dense))
    x_hd = x_dense @ W_hd.T
    ranks = np.argsort(np.argsort(-x_hd, axis=1), axis=1)
    z_hd = np.where(ranks < k, 1., 0.)
    return z_hd, k


def train_W_out(z_hd, labels, sample_mask, n_out, k, p, rng):
    """Hebbian training of W_out on the samples indicated by sample_mask."""
    W_out = np.zeros((n_out, z_hd.shape[1]))
    for i in np.where(sample_mask)[0]:
        if labels[i] != 0:
            active_idx = np.flatnonzero(z_hd[i])
            to_flip = active_idx[rng.random(active_idx.size) < p]
            W_out[int(labels[i]) - 1, to_flip] = 1. / k
    return W_out


def compute_accuracy(z_hd, W_out, true_labels, eval_mask):
    """Frame-level accuracy on the subset defined by eval_mask."""
    z_out = z_hd @ W_out.T
    preds = np.argmax(z_out, axis=1) + 1  # 1-indexed
    return metrics.accuracy_score(true_labels[eval_mask], preds[eval_mask])


# ------------------------------------------------------------------ #
# Single gases                                                       #
# ------------------------------------------------------------------ #
def run_single():
    print("=== Single gases ===")
    sensor_data, sequence, times_sec, sequence_sec = load('1_600_20', reduced=True)
    h = np.median(np.diff(times_sec))
    n_sensors = sensor_data.shape[1]
    n_out_single = 3
    n_train = 450

    rng_pairs = np.random.default_rng(0)
    all_ordered_pairs = [(i, j) for i in range(n_sensors)
                         for j in range(n_sensors) if i != j]
    ratio_pairs = [all_ordered_pairs[i]
                   for i in rng_pairs.permutation(len(all_ordered_pairs))]
    diff_pairs = [all_ordered_pairs[i]
                  for i in rng_pairs.permutation(len(all_ordered_pairs))]

    expanded_data = build_152_expansion(sensor_data, h, ratio_pairs, diff_pairs)

    frame_labels = np.zeros_like(times_sec)
    for i in range(len(sequence_sec)):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        frame_labels[flag] = int(sequence[i][1])

    train_time_end = sequence_sec[n_train] if n_train < len(sequence_sec) else times_sec[-1] + 1
    labeled_mask = frame_labels > 0
    train_mask = labeled_mask & (times_sec < train_time_end)
    test_mask = labeled_mask & (times_sec >= train_time_end)

    train_seq_info = []
    for i in range(n_train):
        cat = int(sequence[i][1])
        if cat == 0:
            continue
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        indices = np.where(flag)[0]
        if len(indices) > 0:
            train_seq_info.append((cat, indices))

    categories = sorted(set(cat for cat, _ in train_seq_info))
    seq_by_cat = {cat: [(c, idx) for c, idx in train_seq_info if c == cat]
                  for cat in categories}

    n_seqs_per_cat = min(len(v) for v in seq_by_cat.values())
    print(f"Categories: {categories}")
    print(f"Sequences per category: {[len(v) for v in seq_by_cat.values()]}")
    print(f"Max picks: {n_seqs_per_cat}")

    all_train_accs = []
    all_test_accs = []
    all_test_accs_svm = []

    for rep in range(N_TIMES):
        rng_rep = np.random.default_rng(rep)
        rng_proj = np.random.default_rng(rep * 1000)

        shuffled = {cat: [seqs[i] for i in rng_rep.permutation(len(seqs))]
                    for cat, seqs in seq_by_cat.items()}

        if NORMALIZED == 'whitened':
            idx_last = np.where(train_mask)[0][-1]
            train_mean = np.mean(expanded_data[:idx_last], axis=0)
            pca = PCA()
            pca.fit(expanded_data[:idx_last] - train_mean)
            var_mask = pca.explained_variance_ > 1e-12
            U = np.diag(1.0 / np.sqrt(pca.explained_variance_[var_mask])) @ pca.components_[var_mask]
            x_dense = (U @ (expanded_data - train_mean).T).T
        else:
            x_dense = expanded_data

        z_hd, k = expand_and_sparsify(x_dense, N_HD, P_HD, D, rng_proj)

        train_accs = []
        test_accs = []
        test_accs_svm = []
        cumulative_mask = np.zeros(len(times_sec), dtype=bool)

        X_test_svm = x_dense[test_mask]
        y_test_svm = frame_labels[test_mask].astype(int)

        for pick in range(n_seqs_per_cat):
            for cat in categories:
                _, indices = shuffled[cat][pick]
                cumulative_mask[indices] = True

            rng_train = np.random.default_rng(rep * 10000 + pick)
            W_out = train_W_out(z_hd, frame_labels, cumulative_mask,
                                n_out_single, k, P, rng_train)

            train_accs.append(
                compute_accuracy(z_hd, W_out, frame_labels, cumulative_mask))
            test_accs.append(
                compute_accuracy(z_hd, W_out, frame_labels, test_mask))

            X_tr_svm = x_dense[cumulative_mask]
            y_tr_svm = frame_labels[cumulative_mask].astype(int)
            if len(np.unique(y_tr_svm)) < 2:
                test_accs_svm.append(np.nan)
            else:
                clf = SVC(kernel='rbf', C=50, gamma='scale')
                clf.fit(X_tr_svm, y_tr_svm)
                test_accs_svm.append(clf.score(X_test_svm, y_test_svm))

        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_test_accs_svm.append(test_accs_svm)
        print(f"  Rep {rep}: E&S test = {test_accs[-1]:.4f}, "
              f"RBF SVM test = {test_accs_svm[-1]:.4f}")

    return {
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        'test_accs_svm': np.array(all_test_accs_svm),
        'n_picks': n_seqs_per_cat,
    }


# ------------------------------------------------------------------ #
# Binary mixtures                                                    #
# ------------------------------------------------------------------ #
def run_binary():
    print("\n=== Binary mixtures ===")
    train_data_raw, train_sequence, train_times_sec, train_sequence_sec = load(
        'mix_100_20_1', reduced=True)
    test_data_raw, test_sequence, test_times_sec, test_sequence_sec = load(
        'mix_50_20_1', reduced=True)

    h_train = np.median(np.diff(train_times_sec))
    h_test = np.median(np.diff(test_times_sec))
    n_sensors_b = train_data_raw.shape[1]
    n_out_binary = 6

    rng_pairs = np.random.default_rng(0)
    all_ordered_pairs = [(i, j) for i in range(n_sensors_b)
                         for j in range(n_sensors_b) if i != j]
    ratio_pairs = [all_ordered_pairs[i]
                   for i in rng_pairs.permutation(len(all_ordered_pairs))]
    diff_pairs = [all_ordered_pairs[i]
                  for i in rng_pairs.permutation(len(all_ordered_pairs))]

    train_expanded = build_152_expansion(train_data_raw, h_train, ratio_pairs, diff_pairs)
    test_expanded = build_152_expansion(test_data_raw, h_test, ratio_pairs, diff_pairs)

    train_labels = np.zeros_like(train_times_sec)
    for i in range(len(train_sequence_sec)):
        try:
            flag = (train_times_sec > train_sequence_sec[i]) & \
                   (train_times_sec < train_sequence_sec[i + 1])
        except IndexError:
            flag = (train_times_sec > train_sequence_sec[i])
        train_labels[flag] = int(train_sequence[i][1])

    test_labels = np.zeros_like(test_times_sec)
    for i in range(len(test_sequence_sec)):
        try:
            flag = (test_times_sec > test_sequence_sec[i]) & \
                   (test_times_sec < test_sequence_sec[i + 1])
        except IndexError:
            flag = (test_times_sec > test_sequence_sec[i])
        test_labels[flag] = int(test_sequence[i][1])

    test_labeled = test_labels > 0

    train_seq_info = []
    for i in range(len(train_sequence_sec)):
        cat = int(train_sequence[i][1])
        if cat == 0:
            continue
        try:
            flag = (train_times_sec > train_sequence_sec[i]) & \
                   (train_times_sec < train_sequence_sec[i + 1])
        except IndexError:
            flag = (train_times_sec > train_sequence_sec[i])
        indices = np.where(flag)[0]
        if len(indices) > 0:
            train_seq_info.append((cat, indices))

    categories = sorted(set(cat for cat, _ in train_seq_info))
    seq_by_cat = {cat: [(c, idx) for c, idx in train_seq_info if c == cat]
                  for cat in categories}

    n_seqs_per_cat = min(len(v) for v in seq_by_cat.values())
    print(f"Categories: {categories}")
    print(f"Sequences per category: {[len(v) for v in seq_by_cat.values()]}")
    print(f"Max picks: {n_seqs_per_cat}")

    all_train_accs = []
    all_test_accs = []
    all_test_accs_svm = []

    for rep in range(N_TIMES):
        rng_rep = np.random.default_rng(rep)
        rng_proj = np.random.default_rng(rep * 1000)

        shuffled = {cat: [seqs[i] for i in rng_rep.permutation(len(seqs))]
                    for cat, seqs in seq_by_cat.items()}

        if NORMALIZED == 'whitened':
            train_mean = np.mean(train_expanded, axis=0)
            pca = PCA()
            pca.fit(train_expanded - train_mean)
            var_mask = pca.explained_variance_ > 1e-12
            U = np.diag(1.0 / np.sqrt(pca.explained_variance_[var_mask])) @ pca.components_[var_mask]
            x_train = (U @ (train_expanded - train_mean).T).T
            x_test = (U @ (test_expanded - train_mean).T).T
        else:
            x_train = train_expanded
            x_test = test_expanded

        n_dense = x_train.shape[1]
        k = int(D * N_HD)
        W_hd = rng_proj.binomial(n=1, p=P_HD, size=(N_HD, n_dense))

        x_hd_train = x_train @ W_hd.T
        ranks = np.argsort(np.argsort(-x_hd_train, axis=1), axis=1)
        z_hd_train = np.where(ranks < k, 1., 0.)

        x_hd_test = x_test @ W_hd.T
        ranks = np.argsort(np.argsort(-x_hd_test, axis=1), axis=1)
        z_hd_test = np.where(ranks < k, 1., 0.)

        train_accs = []
        test_accs = []
        test_accs_svm = []
        cumulative_mask = np.zeros(len(train_times_sec), dtype=bool)

        X_test_svm = x_test[test_labeled]
        y_test_svm = test_labels[test_labeled].astype(int)

        for pick in range(n_seqs_per_cat):
            for cat in categories:
                _, indices = shuffled[cat][pick]
                cumulative_mask[indices] = True

            rng_train = np.random.default_rng(rep * 10000 + pick)
            W_out = train_W_out(z_hd_train, train_labels, cumulative_mask,
                                n_out_binary, k, P, rng_train)

            z_out_train = z_hd_train @ W_out.T
            train_preds = np.argmax(z_out_train, axis=1) + 1
            train_accs.append(metrics.accuracy_score(
                train_labels[cumulative_mask], train_preds[cumulative_mask]))

            z_out_test = z_hd_test @ W_out.T
            test_preds = np.argmax(z_out_test, axis=1) + 1
            test_accs.append(metrics.accuracy_score(
                test_labels[test_labeled], test_preds[test_labeled]))

            X_tr_svm = x_train[cumulative_mask]
            y_tr_svm = train_labels[cumulative_mask].astype(int)
            if len(np.unique(y_tr_svm)) < 2:
                test_accs_svm.append(np.nan)
            else:
                clf = SVC(kernel='rbf', C=50, gamma='scale')
                clf.fit(X_tr_svm, y_tr_svm)
                test_accs_svm.append(clf.score(X_test_svm, y_test_svm))

        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_test_accs_svm.append(test_accs_svm)
        print(f"  Rep {rep}: E&S test = {test_accs[-1]:.4f}, "
              f"RBF SVM test = {test_accs_svm[-1]:.4f}")

    return {
        'train_accs': np.array(all_train_accs),
        'test_accs': np.array(all_test_accs),
        'test_accs_svm': np.array(all_test_accs_svm),
        'n_picks': n_seqs_per_cat,
    }


# ------------------------------------------------------------------ #
# Entry point                                                        #
# ------------------------------------------------------------------ #
def main():
    params = {
        'n_hd': N_HD,
        'p_hd': P_HD,
        'd': D,
        'p': P,
        'normalized': NORMALIZED,
        'n_times': N_TIMES,
    }

    single = run_single()
    binary = run_binary()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump({'params': params, 'single': single, 'binary': binary}, f)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == '__main__':
    main()
