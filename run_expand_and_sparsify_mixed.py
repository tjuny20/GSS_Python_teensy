
#!/usr/bin/env python3
"""
Expand-and-sparsify gridsearch for GPU server.

Configs (with per-repeat pair resampling for ratios & differences):
    ∂¹+∂²                  (24 traces)
    ∂¹+∂² + 8R + 8D        (40 traces)
    ∂¹+∂² + 64R + 64D      (152 traces)
    ∂¹⁻⁷ + 64R + 64D       (192 traces)

Single gas:    1_600_20 with n_train=450 sequence-based split
Binary mix:    mix_100_20_1 (train) / mix_50_20_1 (test)

Usage:
    python run_expand_and_sparsify_mixed.py
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


# ── Per-seed pair resampling ────────────────────────────────────────────
def sample_pairs(n_sensors, seed):
    """Return fresh ratio_pairs and diff_pairs for a given seed."""
    all_ordered = [(i, j) for i in range(n_sensors)
                          for j in range(n_sensors) if i != j]
    rng = np.random.default_rng(seed)
    rp = [all_ordered[i] for i in rng.permutation(len(all_ordered))]
    dp = [all_ordered[i] for i in rng.permutation(len(all_ordered))]
    return rp, dp


def all_pairs_with_self(n_sensors):
    """All n_sensors*n_sensors ordered pairs (including self)."""
    return [(i, j) for i in range(n_sensors) for j in range(n_sensors)]


def needs_resampling(n_ratios, n_diffs, n_sensors):
    """64R/64D uses all pairs — no resampling needed."""
    n_all = n_sensors * n_sensors  # 64
    return (n_ratios > 0 and n_ratios < n_all) or (n_diffs > 0 and n_diffs < n_all)


def make_ratios(data, pairs, n):
    return np.column_stack([data[:, i] / (data[:, j] + 1e-8)
                            for i, j in pairs[:n]])


def make_diffs(data, pairs, n):
    return np.column_stack([data[:, i] - data[:, j]
                            for i, j in pairs[:n]])


def build_expansion(data, deriv_cache, max_order, n_ratios, n_diffs, rp, dp):
    """Assemble expansion from precomputed derivatives + fresh pairs."""
    parts = [deriv_cache[max_order]]
    if n_ratios > 0:
        parts.append(make_ratios(data, rp, n_ratios))
    if n_diffs > 0:
        parts.append(make_diffs(data, dp, n_diffs))
    return np.hstack(parts) if len(parts) > 1 else parts[0]


# ── Config specs: name -> (max_order, n_ratios, n_diffs) ───────────────
CONFIG_SPECS = {
    '\u2202\u00b9+\u2202\u00b2':              (2, 0, 0),
    '\u2202\u00b9+\u2202\u00b2 + 8R + 8D':    (2, 8, 8),
    '\u2202\u00b9+\u2202\u00b2 + 64R + 64D':  (2, 64, 64),
    '\u2202\u00b9\u207b\u2077 + 64R + 64D':   (7, 64, 64),
}


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

    # ---- sequence-based train/test boundary ----
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

    # ---- precompute derivative expansions ----
    deriv_orders = set(spec[0] for spec in CONFIG_SPECS.values())
    deriv_cache = {order: expand_with_derivatives(sensor_data, h, max_order=order)
                   for order in deriv_orders}

    config_names = list(CONFIG_SPECS.keys())
    for name, (mo, nr, nd) in CONFIG_SPECS.items():
        n_dense = n_sensors * (mo + 1) + nr + nd
        print(f"  {name}: {n_dense} cols")

    # ---- gridsearch ----
    cos_input  = {}
    cos_output = {}
    acc_table  = {}
    pair_log   = {}   # cfg_name -> [(rp, dp), ...] length n_repeats

    y_train_t = torch.tensor(y_train_0, dtype=torch.long, device=device)
    y_test_t  = torch.tensor(y_test_0,  dtype=torch.long, device=device)

    full_pairs = all_pairs_with_self(n_sensors)

    for ci, cfg_name in enumerate(config_names):
        max_order, n_ratios, n_diffs = CONFIG_SPECS[cfg_name]
        resample = needs_resampling(n_ratios, n_diffs, n_sensors)
        has_pairs = n_ratios > 0 or n_diffs > 0
        print(f"\n  [{ci+1}/{len(config_names)}] {cfg_name}")
        t0_cfg = time.time()

        # -- pre-sample pairs for all repeats and log them --
        seed_pairs = {}
        pairs_list = []
        for seed in range(n_repeats):
            if resample:
                rp, dp = sample_pairs(n_sensors, seed)
                seed_pairs[seed] = (rp, dp)
                pairs_list.append((rp[:n_ratios], dp[:n_diffs]))
            elif has_pairs:
                # Full 64-pair set — deterministic
                seed_pairs[seed] = (full_pairs, full_pairs)
                pairs_list.append((full_pairs[:n_ratios], full_pairs[:n_diffs]))
            else:
                seed_pairs[seed] = (None, None)
                pairs_list.append(([], []))
        pair_log[cfg_name] = pairs_list

        # -- build reference expansion (seed=0) for cosine similarities --
        if has_pairs:
            rp0, dp0 = seed_pairs[0]
            x_dense = build_expansion(sensor_data, deriv_cache, max_order,
                                      n_ratios, n_diffs, rp0, dp0)
        else:
            x_dense = deriv_cache[max_order]

        n_dense = x_dense.shape[1]
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

        n_combos = len(p_hd_sweep) * len(d_sweep)
        combo_i  = 0

        for p in p_hd_sweep:
            for d in d_sweep:
                combo_i += 1
                k = int(d * n_hd)

                # -- project + sparsify (cosine scatter, seed=0 expansion) --
                W_hd = torch.bernoulli(
                    torch.full((n_hd, n_dense), p, device=device))
                x_hd = x_dense_t @ W_hd.T
                _, topk_idx = torch.topk(x_hd, k, dim=1, largest=True)
                z_hd = torch.zeros_like(x_hd)
                z_hd.scatter_(1, topk_idx, 1.0)

                cos_hd = (z_hd[i1] * z_hd[i2]).sum(dim=1) / k
                cos_output[(cfg_name, p, d)] = cos_hd.cpu().numpy()
                del x_hd, z_hd, W_hd

                # -- classification accuracy (fresh pairs each repeat) --
                accs = []
                for seed in range(n_repeats):
                    torch.manual_seed(seed)
                    rp, dp = seed_pairs[seed]

                    if has_pairs:
                        x_exp = build_expansion(sensor_data, deriv_cache,
                                                max_order, n_ratios, n_diffs,
                                                rp, dp)
                    else:
                        x_exp = deriv_cache[max_order]

                    x_train_labeled = torch.tensor(
                        x_exp[train_labeled], dtype=torch.float32, device=device)
                    x_test_labeled = torch.tensor(
                        x_exp[test_labeled], dtype=torch.float32, device=device)

                    W_hd = torch.bernoulli(
                        torch.full((n_hd, n_dense), p, device=device))

                    x_hd_train = x_train_labeled @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_train, k, dim=1, largest=True)
                    z_hd_train = torch.zeros_like(x_hd_train)
                    z_hd_train.scatter_(1, topk_idx, 1.0)

                    x_hd_test = x_test_labeled @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_test, k, dim=1, largest=True)
                    z_hd_test = torch.zeros_like(x_hd_test)
                    z_hd_test.scatter_(1, topk_idx, 1.0)

                    W, b = train_linear_svm(
                        z_hd_train, y_train_t, n_classes)
                    accs.append(score_linear_svm(
                        z_hd_test, y_test_t, W, b))
                    del W_hd, x_hd_train, z_hd_train, x_hd_test, z_hd_test
                    del x_train_labeled, x_test_labeled

                acc_table[(cfg_name, p, d)] = (np.mean(accs), np.std(accs))
                print(f"    [{combo_i:2d}/{n_combos}]  p={p}  d={d}  "
                      f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        del x_dense_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    done in {time.time() - t0_cfg:.1f}s")

    # ---- save ----
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'cos_input':    cos_input,
            'cos_output':   cos_output,
            'acc_table':    acc_table,
            'pair_log':     pair_log,
            'config_specs': dict(CONFIG_SPECS),
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

    # ---- precompute derivative expansions for train and test ----
    deriv_orders = set(spec[0] for spec in CONFIG_SPECS.values())
    deriv_cache_train = {order: expand_with_derivatives(train_sensor, h_train, max_order=order)
                         for order in deriv_orders}
    deriv_cache_test  = {order: expand_with_derivatives(test_sensor, h_test, max_order=order)
                         for order in deriv_orders}

    config_names = list(CONFIG_SPECS.keys())
    for name, (mo, nr, nd) in CONFIG_SPECS.items():
        n_dense = n_sensors * (mo + 1) + nr + nd
        print(f"  {name}: {n_dense} cols")

    # ---- gridsearch ----
    cos_input  = {}
    cos_output = {}
    acc_table  = {}
    pair_log   = {}   # cfg_name -> [(rp, dp), ...] length n_repeats

    y_train_t = torch.tensor(y_train_0, dtype=torch.long, device=device)
    y_test_t  = torch.tensor(y_test_0,  dtype=torch.long, device=device)

    full_pairs = all_pairs_with_self(n_sensors)

    for ci, cfg_name in enumerate(config_names):
        max_order, n_ratios, n_diffs = CONFIG_SPECS[cfg_name]
        resample = needs_resampling(n_ratios, n_diffs, n_sensors)
        has_pairs = n_ratios > 0 or n_diffs > 0
        print(f"\n  [{ci+1}/{len(config_names)}] {cfg_name}")
        t0_cfg = time.time()

        # -- pre-sample pairs for all repeats and log them --
        seed_pairs = {}
        pairs_list = []
        for seed in range(n_repeats):
            if resample:
                rp, dp = sample_pairs(n_sensors, seed)
                seed_pairs[seed] = (rp, dp)
                pairs_list.append((rp[:n_ratios], dp[:n_diffs]))
            elif has_pairs:
                seed_pairs[seed] = (full_pairs, full_pairs)
                pairs_list.append((full_pairs[:n_ratios], full_pairs[:n_diffs]))
            else:
                seed_pairs[seed] = (None, None)
                pairs_list.append(([], []))
        pair_log[cfg_name] = pairs_list

        # -- build reference expansion (seed=0) for cosine similarities --
        if has_pairs:
            rp0, dp0 = seed_pairs[0]
            x_train_dense = build_expansion(train_sensor, deriv_cache_train,
                                            max_order, n_ratios, n_diffs, rp0, dp0)
        else:
            x_train_dense = deriv_cache_train[max_order]

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

        n_combos = len(p_hd_sweep) * len(d_sweep)
        combo_i  = 0

        for p in p_hd_sweep:
            for d in d_sweep:
                combo_i += 1
                k = int(d * n_hd)

                # -- project + sparsify (cosine scatter, seed=0 expansion) --
                W_hd = torch.bernoulli(
                    torch.full((n_hd, n_dense), p, device=device))
                x_hd = x_train_dense_t @ W_hd.T
                _, topk_idx = torch.topk(x_hd, k, dim=1, largest=True)
                z_hd = torch.zeros_like(x_hd)
                z_hd.scatter_(1, topk_idx, 1.0)

                cos_hd = (z_hd[i1] * z_hd[i2]).sum(dim=1) / k
                cos_output[(cfg_name, p, d)] = cos_hd.cpu().numpy()
                del x_hd, z_hd, W_hd

                # -- classification accuracy (fresh pairs each repeat) --
                accs = []
                for seed in range(n_repeats):
                    torch.manual_seed(seed)
                    rp, dp = seed_pairs[seed]

                    if has_pairs:
                        x_exp_train = build_expansion(
                            train_sensor, deriv_cache_train, max_order,
                            n_ratios, n_diffs, rp, dp)
                        x_exp_test = build_expansion(
                            test_sensor, deriv_cache_test, max_order,
                            n_ratios, n_diffs, rp, dp)
                    else:
                        x_exp_train = deriv_cache_train[max_order]
                        x_exp_test  = deriv_cache_test[max_order]

                    x_train_labeled = torch.tensor(
                        x_exp_train[train_labeled_mask],
                        dtype=torch.float32, device=device)
                    x_test_labeled = torch.tensor(
                        x_exp_test[test_labeled_mask],
                        dtype=torch.float32, device=device)

                    W_hd = torch.bernoulli(
                        torch.full((n_hd, n_dense), p, device=device))

                    x_hd_train = x_train_labeled @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_train, k, dim=1, largest=True)
                    z_hd_train = torch.zeros_like(x_hd_train)
                    z_hd_train.scatter_(1, topk_idx, 1.0)

                    x_hd_test = x_test_labeled @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_test, k, dim=1, largest=True)
                    z_hd_test = torch.zeros_like(x_hd_test)
                    z_hd_test.scatter_(1, topk_idx, 1.0)

                    W, b = train_linear_svm(
                        z_hd_train, y_train_t, n_classes)
                    accs.append(score_linear_svm(
                        z_hd_test, y_test_t, W, b))
                    del W_hd, x_hd_train, z_hd_train, x_hd_test, z_hd_test
                    del x_train_labeled, x_test_labeled

                acc_table[(cfg_name, p, d)] = (np.mean(accs), np.std(accs))
                print(f"    [{combo_i:2d}/{n_combos}]  p={p}  d={d}  "
                      f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        del x_train_dense_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    done in {time.time() - t0_cfg:.1f}s")

    # ---- save ----
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'cos_input':    cos_input,
            'cos_output':   cos_output,
            'acc_table':    acc_table,
            'pair_log':     pair_log,
            'config_specs': dict(CONFIG_SPECS),
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
        'data/expand_sparsify_mixed_results.pkl',
        n_train=450)

    # Binary mixtures: separate train/test files
    run_gridsearch_binary(
        'mix_100_20_1', 'mix_50_20_1',
        'data/expand_sparsify_mixed_mix_results.pkl')

    print(f"\nAll done in {time.time() - t_start:.1f}s")
