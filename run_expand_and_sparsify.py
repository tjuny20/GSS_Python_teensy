
#!/usr/bin/env python3
"""
Expand-and-sparsify gridsearch for GPU server.

Single gas:    1_600_20 with n_train=450 sequence-based split (as in gridsearch_single.py)
Binary mix:    mix_100_20_1 (train) / mix_50_20_1 (test)  (as in gridsearch_binary.py)

Usage:
    python run_expand_and_sparsify.py
"""

import numpy as np
import torch
import pickle
import os
import csv
import time
from math import comb
from datetime import datetime


# ── Device ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


# ── Data loading (self-contained, no tools.py dependency) ───────────────
def load(filename, reduced=True):
    responding_sens = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0]
    sensor_data = []
    times = []
    with open(f'data/{filename}.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'Timestamp':
                continue
            times.append(row[0])
            values = []
            for i in range(17):
                b1 = int(row[2 * i + 1])
                b2 = int(row[2 * i + 2])
                values.append(int.from_bytes([b1, b2], byteorder="little"))
            sensor_data.append(values)
    sensor_data = np.array(sensor_data)
    if reduced:
        sensor_data = np.delete(
            sensor_data,
            np.where(np.array(responding_sens) == 0)[0], axis=1)
    sequence = pickle.load(open(f'data/{filename}_sequence.pkl', 'rb'))
    times_sec = []
    for dt_str in times:
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
        times_sec.append(seconds)
    sequence_sec = []
    for dt_str in sequence:
        dt = datetime.strptime(dt_str[0], '%a %b %d %H:%M:%S %Y')
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        sequence_sec.append(seconds)
    return sensor_data, sequence, np.array(times_sec), np.array(sequence_sec)


# ── Expansion helpers ───────────────────────────────────────────────────
def backward_diff_array(y, h, n):
    """n-th backward finite difference (zero-padded to preserve length)."""
    coeffs = np.array([(-1)**k * comb(n, k) for k in range(n + 1)])
    raw = np.convolve(y, coeffs, mode='valid') / h**n
    return np.concatenate([np.zeros(n), raw])


def expand_with_derivatives(data, h, max_order):
    """Append backward differences of orders 1..max_order (8 cols each)."""
    if max_order == 0:
        return data.copy()
    derivs = []
    for order in range(1, max_order + 1):
        d = np.apply_along_axis(
            lambda col: backward_diff_array(col, h, order), axis=0, arr=data)
        derivs.append(d)
    return np.hstack([data] + derivs)


# ── Linear SVM (Crammer-Singer on GPU) ──────────────────────────────────
def train_linear_svm(X, y, n_classes, C=1.0, lr=1e-3, n_epochs=200):
    n, d = X.shape
    W = torch.zeros(n_classes, d, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {'params': W, 'weight_decay': 1.0 / (C * n)},
        {'params': b, 'weight_decay': 0.0}
    ], lr=lr)
    idx = torch.arange(n, device=device)
    for _ in range(n_epochs):
        scores = X @ W.T + b
        correct_scores = scores[idx, y].unsqueeze(1)
        margins = scores - correct_scores + 1.0
        mask = torch.ones_like(scores)
        mask[idx, y] = 0.0
        loss = (torch.clamp(margins, min=0) * mask).sum() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


@torch.no_grad()
def score_linear_svm(X, y, W, b):
    preds = (X @ W.T + b).argmax(dim=1)
    return (preds == y).float().mean().item()


# ── Build configs for a given dataset ───────────────────────────────────
def build_configs(sensor_data, h, n_sensors):
    """Return the 4 expansion configs (all 32 traces) + pair pools."""
    rng_pairs = np.random.default_rng(0)
    all_ordered_pairs = [(i, j) for i in range(n_sensors)
                                for j in range(n_sensors) if i != j]
    ratio_pairs = [all_ordered_pairs[i]
                   for i in rng_pairs.permutation(len(all_ordered_pairs))]
    diff_pairs  = [all_ordered_pairs[i]
                   for i in rng_pairs.permutation(len(all_ordered_pairs))]

    x_d1   = expand_with_derivatives(sensor_data, h, max_order=1)   # 16
    x_d12  = expand_with_derivatives(sensor_data, h, max_order=2)   # 24
    x_d123 = expand_with_derivatives(sensor_data, h, max_order=3)   # 32
    r8 = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8)
                          for i, j in ratio_pairs[:8]])
    d8 = np.column_stack([sensor_data[:, i] - sensor_data[:, j]
                          for i, j in diff_pairs[:8]])

    configs = {
        '\u2202\u00b9+\u2202\u00b2+\u2202\u00b3':     x_d123,
        '\u2202\u00b9+\u2202\u00b2 + 8R':              np.hstack([x_d12, r8]),
        '\u2202\u00b9+\u2202\u00b2 + 8D':              np.hstack([x_d12, d8]),
        '\u2202\u00b9 + 8R + 8D':                      np.hstack([x_d1, r8, d8]),
    }
    return configs


# ── Gridsearch: single file with sequence-based train/test split ────────
def run_gridsearch_single(filename, pkl_path, n_train=450,
                          n_hd=10_000,
                          p_hd_sweep=(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975),
                          d_sweep=(0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                          n_pairs=1_000,
                          n_repeats=10):
    """Expand-and-sparsify gridsearch for a single file, split at n_train sequences."""

    if os.path.exists(pkl_path):
        print(f"\n  {pkl_path} already exists -- skipping. Delete to recompute.")
        return

    print(f"\n{'='*64}")
    print(f"  {filename}  (n_train={n_train})  ->  {pkl_path}")
    print(f"{'='*64}")
    t0_total = time.time()

    # ---- load ----
    sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
    h = np.median(np.diff(times_sec))
    n_sensors = sensor_data.shape[1]

    # ---- build frame-level labels ----
    labels = np.zeros_like(times_sec)
    for i in range(len(sequence_sec)):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        labels[flag] = int(sequence[i][1])

    # ---- sequence-based train/test boundary (same as gridsearch_single) ----
    # Frames labeled by the first n_train sequences -> train
    # Frames labeled by the remaining sequences    -> test
    train_frame_mask = np.zeros(len(times_sec), dtype=bool)
    test_frame_mask  = np.zeros(len(times_sec), dtype=bool)
    for i in range(len(sequence_sec)):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        if labels[flag].any():
            if i < n_train:
                train_frame_mask |= flag
            else:
                test_frame_mask |= flag

    # Keep only labeled frames
    train_labeled = train_frame_mask & (labels > 0)
    test_labeled  = test_frame_mask  & (labels > 0)

    y_train_0 = labels[train_labeled].astype(int) - 1
    y_test_0  = labels[test_labeled].astype(int) - 1
    n_classes = max(int(y_train_0.max()), int(y_test_0.max())) + 1

    print(f"  Sensor data : {sensor_data.shape}")
    print(f"  Train frames: {np.sum(train_labeled)}  Test frames: {np.sum(test_labeled)}")
    print(f"  Classes     : {n_classes}")
    print(f"  n_hd={n_hd}  n_repeats={n_repeats}  n_pairs={n_pairs}")
    print(f"  p_hd_sweep  : {list(p_hd_sweep)}")
    print(f"  d_sweep     : {list(d_sweep)}")

    # ---- configs ----
    configs = build_configs(sensor_data, h, n_sensors)
    config_names = list(configs.keys())
    for name, x in configs.items():
        print(f"  {name}: {x.shape}")

    # ---- gridsearch ----
    cos_input  = {}
    cos_output = {}
    acc_table  = {}

    y_train_t = torch.tensor(y_train_0, dtype=torch.long, device=device)
    y_test_t  = torch.tensor(y_test_0,  dtype=torch.long, device=device)

    for ci, cfg_name in enumerate(config_names):
        print(f"\n  [{ci+1}/{len(config_names)}] {cfg_name}")
        t0_cfg = time.time()

        x_dense  = configs[cfg_name]
        n_dense  = x_dense.shape[1]
        x_dense_t = torch.tensor(x_dense, dtype=torch.float32, device=device)

        # -- dense cosine similarities (sampled pairs) --
        n_samples = x_dense.shape[0]
        pair_idx = np.array([np.random.choice(n_samples, size=2, replace=False)
                             for _ in range(n_pairs)])
        i1 = torch.tensor(pair_idx[:, 0], device=device)
        i2 = torch.tensor(pair_idx[:, 1], device=device)
        v1 = x_dense_t[i1]
        v2 = x_dense_t[i2]
        cos_dense = (v1 * v2).sum(dim=1) / (v1.norm(dim=1) * v2.norm(dim=1))
        cos_input[cfg_name] = cos_dense.cpu().numpy()
        del v1, v2

        x_train_labeled_t = torch.tensor(
            x_dense[train_labeled], dtype=torch.float32, device=device)
        x_test_labeled_t = torch.tensor(
            x_dense[test_labeled], dtype=torch.float32, device=device)

        n_combos = len(p_hd_sweep) * len(d_sweep)
        combo_i  = 0

        for p in p_hd_sweep:
            for d in d_sweep:
                combo_i += 1
                k = int(d * n_hd)

                # -- project + sparsify (inner product scatter) --
                W_hd = torch.bernoulli(
                    torch.full((n_hd, n_dense), p, device=device))
                x_hd = x_dense_t @ W_hd.T
                _, topk_idx = torch.topk(x_hd, k, dim=1, largest=True)
                z_hd = torch.zeros_like(x_hd)
                z_hd.scatter_(1, topk_idx, 1.0)

                cos_hd = (z_hd[i1] * z_hd[i2]).sum(dim=1) / k
                cos_output[(cfg_name, p, d)] = cos_hd.cpu().numpy()
                del x_hd, z_hd, W_hd

                # -- classification accuracy (multiple repeats) --
                accs = []
                for seed in range(n_repeats):
                    torch.manual_seed(seed)
                    W_hd = torch.bernoulli(
                        torch.full((n_hd, n_dense), p, device=device))

                    x_hd_train = x_train_labeled_t @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_train, k, dim=1, largest=True)
                    z_hd_train = torch.zeros_like(x_hd_train)
                    z_hd_train.scatter_(1, topk_idx, 1.0)

                    x_hd_test = x_test_labeled_t @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_test, k, dim=1, largest=True)
                    z_hd_test = torch.zeros_like(x_hd_test)
                    z_hd_test.scatter_(1, topk_idx, 1.0)

                    W, b = train_linear_svm(
                        z_hd_train, y_train_t, n_classes)
                    accs.append(score_linear_svm(
                        z_hd_test, y_test_t, W, b))
                    del W_hd, x_hd_train, z_hd_train, x_hd_test, z_hd_test

                acc_table[(cfg_name, p, d)] = (np.mean(accs), np.std(accs))
                print(f"    [{combo_i:2d}/{n_combos}]  p={p}  d={d}  "
                      f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        del x_dense_t, x_train_labeled_t, x_test_labeled_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    done in {time.time() - t0_cfg:.1f}s")

    # ---- save ----
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'cos_input':    cos_input,
            'cos_output':   cos_output,
            'acc_table':    acc_table,
            'config_names': config_names,
            'p_hd_sweep':   list(p_hd_sweep),
            'd_sweep':      list(d_sweep),
            'n_hd':         n_hd,
            'n_pairs':      n_pairs,
            'n_repeats':    n_repeats,
        }, f)

    elapsed = time.time() - t0_total
    print(f"\n  Saved -> {pkl_path}  ({elapsed:.1f}s total)")


# ── Gridsearch: separate train/test files ───────────────────────────────
def run_gridsearch_binary(train_filename, test_filename, pkl_path,
                          n_hd=10_000,
                          p_hd_sweep=(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975),
                          d_sweep=(0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                          n_pairs=1_000,
                          n_repeats=10):
    """Expand-and-sparsify gridsearch with separate train/test data files."""

    if os.path.exists(pkl_path):
        print(f"\n  {pkl_path} already exists -- skipping. Delete to recompute.")
        return

    print(f"\n{'='*64}")
    print(f"  train={train_filename}  test={test_filename}  ->  {pkl_path}")
    print(f"{'='*64}")
    t0_total = time.time()

    # ---- load training data ----
    train_sensor, train_seq, train_times, train_seq_sec = load(train_filename, reduced=True)
    h_train = np.median(np.diff(train_times))
    n_sensors = train_sensor.shape[1]

    train_labels = np.zeros_like(train_times)
    for i in range(len(train_seq_sec)):
        try:
            flag = (train_times > train_seq_sec[i]) & (train_times < train_seq_sec[i + 1])
        except IndexError:
            flag = (train_times > train_seq_sec[i])
        train_labels[flag] = int(train_seq[i][1])

    train_labeled_mask = train_labels > 0
    y_train_0 = train_labels[train_labeled_mask].astype(int) - 1

    # ---- load test data ----
    test_sensor, test_seq, test_times, test_seq_sec = load(test_filename, reduced=True)
    h_test = np.median(np.diff(test_times))

    test_labels = np.zeros_like(test_times)
    for i in range(len(test_seq_sec)):
        try:
            flag = (test_times > test_seq_sec[i]) & (test_times < test_seq_sec[i + 1])
        except IndexError:
            flag = (test_times > test_seq_sec[i])
        test_labels[flag] = int(test_seq[i][1])

    test_labeled_mask = test_labels > 0
    y_test_0 = test_labels[test_labeled_mask].astype(int) - 1

    n_classes = max(int(y_train_0.max()), int(y_test_0.max())) + 1

    print(f"  Train data  : {train_sensor.shape}  (labeled: {np.sum(train_labeled_mask)})")
    print(f"  Test data   : {test_sensor.shape}  (labeled: {np.sum(test_labeled_mask)})")
    print(f"  Classes     : {n_classes}")
    print(f"  n_hd={n_hd}  n_repeats={n_repeats}  n_pairs={n_pairs}")
    print(f"  p_hd_sweep  : {list(p_hd_sweep)}")
    print(f"  d_sweep     : {list(d_sweep)}")

    # ---- configs ----
    train_configs = build_configs(train_sensor, h_train, n_sensors)
    test_configs  = build_configs(test_sensor, h_test, n_sensors)
    config_names = list(train_configs.keys())
    for name, x in train_configs.items():
        print(f"  {name}: {x.shape}")

    # ---- gridsearch ----
    cos_input  = {}
    cos_output = {}
    acc_table  = {}

    y_train_t = torch.tensor(y_train_0, dtype=torch.long, device=device)
    y_test_t  = torch.tensor(y_test_0,  dtype=torch.long, device=device)

    for ci, cfg_name in enumerate(config_names):
        print(f"\n  [{ci+1}/{len(config_names)}] {cfg_name}")
        t0_cfg = time.time()

        x_train_dense = train_configs[cfg_name]
        n_dense = x_train_dense.shape[1]
        x_train_dense_t = torch.tensor(x_train_dense, dtype=torch.float32, device=device)

        # -- dense cosine similarities (sampled pairs from training data) --
        n_samples = x_train_dense.shape[0]
        pair_idx = np.array([np.random.choice(n_samples, size=2, replace=False)
                             for _ in range(n_pairs)])
        i1 = torch.tensor(pair_idx[:, 0], device=device)
        i2 = torch.tensor(pair_idx[:, 1], device=device)
        v1 = x_train_dense_t[i1]
        v2 = x_train_dense_t[i2]
        cos_dense = (v1 * v2).sum(dim=1) / (v1.norm(dim=1) * v2.norm(dim=1))
        cos_input[cfg_name] = cos_dense.cpu().numpy()
        del v1, v2

        x_train_labeled_t = torch.tensor(
            x_train_dense[train_labeled_mask], dtype=torch.float32, device=device)
        x_test_labeled_t = torch.tensor(
            test_configs[cfg_name][test_labeled_mask], dtype=torch.float32, device=device)

        n_combos = len(p_hd_sweep) * len(d_sweep)
        combo_i  = 0

        for p in p_hd_sweep:
            for d in d_sweep:
                combo_i += 1
                k = int(d * n_hd)

                # -- project + sparsify (inner product scatter on training data) --
                W_hd = torch.bernoulli(
                    torch.full((n_hd, n_dense), p, device=device))
                x_hd = x_train_dense_t @ W_hd.T
                _, topk_idx = torch.topk(x_hd, k, dim=1, largest=True)
                z_hd = torch.zeros_like(x_hd)
                z_hd.scatter_(1, topk_idx, 1.0)

                cos_hd = (z_hd[i1] * z_hd[i2]).sum(dim=1) / k
                cos_output[(cfg_name, p, d)] = cos_hd.cpu().numpy()
                del x_hd, z_hd, W_hd

                # -- classification accuracy (multiple repeats) --
                accs = []
                for seed in range(n_repeats):
                    torch.manual_seed(seed)
                    W_hd = torch.bernoulli(
                        torch.full((n_hd, n_dense), p, device=device))

                    x_hd_train = x_train_labeled_t @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_train, k, dim=1, largest=True)
                    z_hd_train = torch.zeros_like(x_hd_train)
                    z_hd_train.scatter_(1, topk_idx, 1.0)

                    x_hd_test = x_test_labeled_t @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_test, k, dim=1, largest=True)
                    z_hd_test = torch.zeros_like(x_hd_test)
                    z_hd_test.scatter_(1, topk_idx, 1.0)

                    W, b = train_linear_svm(
                        z_hd_train, y_train_t, n_classes)
                    accs.append(score_linear_svm(
                        z_hd_test, y_test_t, W, b))
                    del W_hd, x_hd_train, z_hd_train, x_hd_test, z_hd_test

                acc_table[(cfg_name, p, d)] = (np.mean(accs), np.std(accs))
                print(f"    [{combo_i:2d}/{n_combos}]  p={p}  d={d}  "
                      f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        del x_train_dense_t, x_train_labeled_t, x_test_labeled_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    done in {time.time() - t0_cfg:.1f}s")

    # ---- save ----
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'cos_input':    cos_input,
            'cos_output':   cos_output,
            'acc_table':    acc_table,
            'config_names': config_names,
            'p_hd_sweep':   list(p_hd_sweep),
            'd_sweep':      list(d_sweep),
            'n_hd':         n_hd,
            'n_pairs':      n_pairs,
            'n_repeats':    n_repeats,
        }, f)

    elapsed = time.time() - t0_total
    print(f"\n  Saved -> {pkl_path}  ({elapsed:.1f}s total)")


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_start = time.time()

    # Single gas: 1_600_20 with n_train=450 sequence split
    run_gridsearch_single(
        '1_600_20',
        'data/expand_sparsify_results.pkl',
        n_train=450)

    # Binary mixtures: separate train/test files
    run_gridsearch_binary(
        'mix_100_20_1', 'mix_50_20_1',
        'data/expand_sparsify_mix_results.pkl')

    print(f"\nAll done in {time.time() - t_start:.1f}s")
