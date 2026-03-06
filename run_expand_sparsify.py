#!/usr/bin/env python3
"""
Expand-and-sparsify gridsearch for GPU server.

Runs the full p_hd x d sweep for both 1_600_20 and mix_100_20_1 sequences,
computing cosine similarities (input vs output inner products) and linear SVM
accuracy on z_hd. Saves results as pickles that the Jupyter notebooks can
load directly.

Usage:
    python run_expand_sparsify.py

Expects:
    data/1_600_20.csv,            data/1_600_20_sequence.pkl
    data/mix_100_20_1.csv,        data/mix_100_20_1_sequence.pkl
"""

import numpy as np
import torch
import pickle
import os
import csv
import time
from math import comb
from datetime import datetime
from sklearn.model_selection import train_test_split


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


# ── Gridsearch ──────────────────────────────────────────────────────────
def run_gridsearch(filename, pkl_path,
                   n_hd=10_000,
                   p_hd_sweep=(0.05, 0.1, 0.2, 0.4),
                   d_sweep=(0.03, 0.05, 0.1, 0.2, 0.4, 0.6),
                   n_pairs=10_000,
                   n_repeats=20):
    """Run the expand-and-sparsify gridsearch for one dataset."""

    if os.path.exists(pkl_path):
        print(f"\n  {pkl_path} already exists -- skipping. Delete to recompute.")
        return

    print(f"\n{'='*64}")
    print(f"  {filename}  ->  {pkl_path}")
    print(f"{'='*64}")
    t0_total = time.time()

    # ---- load ----
    sensor_data, sequence, times_sec, sequence_sec = load(filename, reduced=True)
    h = np.median(np.diff(times_sec))
    n_sensors = sensor_data.shape[1]

    labels = np.zeros_like(times_sec)
    for i in range(len(sequence_sec)):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        labels[flag] = int(sequence[i][1])

    labeled_mask = labels > 0
    y_labels_0 = labels[labeled_mask].astype(int) - 1
    n_classes = int(y_labels_0.max()) + 1

    print(f"  Sensor data : {sensor_data.shape}")
    print(f"  Labeled     : {np.sum(labeled_mask)}")
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

    y_t = torch.tensor(y_labels_0, dtype=torch.long, device=device)

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

        x_labeled   = x_dense[labeled_mask]
        x_labeled_t = torch.tensor(x_labeled, dtype=torch.float32, device=device)

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
                    x_hd_lab = x_labeled_t @ W_hd.T
                    _, topk_idx = torch.topk(x_hd_lab, k, dim=1, largest=True)
                    z_hd_lab = torch.zeros_like(x_hd_lab)
                    z_hd_lab.scatter_(1, topk_idx, 1.0)

                    indices = np.arange(len(y_labels_0))
                    tr_idx, te_idx = train_test_split(
                        indices, test_size=0.2,
                        random_state=seed, stratify=y_labels_0)
                    tr_idx = torch.tensor(tr_idx, device=device)
                    te_idx = torch.tensor(te_idx, device=device)

                    W, b = train_linear_svm(
                        z_hd_lab[tr_idx], y_t[tr_idx], n_classes)
                    accs.append(score_linear_svm(
                        z_hd_lab[te_idx], y_t[te_idx], W, b))
                    del W_hd, x_hd_lab, z_hd_lab

                acc_table[(cfg_name, p, d)] = (np.mean(accs), np.std(accs))
                print(f"    [{combo_i:2d}/{n_combos}]  p={p}  d={d:.2f}  "
                      f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        del x_dense_t, x_labeled_t
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

    run_gridsearch('1_600_20',     'data/expand_sparsify_results.pkl')
    run_gridsearch('mix_100_20_1', 'data/expand_sparsify_mix_results.pkl')

    print(f"\nAll done in {time.time() - t_start:.1f}s")
