"""
Gridsearch for the 'Hazardous emission detection - texel' notebook.

Sweeps:
    N  — number of training presentations (first N from the corrected sequence)
    s  — sparsity of the expand-and-sparsify layer; k = round(s * n_hd)

Fixed (same as the notebook):
    dataset      = '1_600_20_corrected'  (gap-fixed sequence)
    expansion    = ∂¹+∂² + 56R + 56D  (136 features)
    n_hd         = 4640
    p_hd         = 0.025
    p            = 1.0    (Hebbian flip probability per active HD component)
    n_out        = 2      (neuron 0 = hazardous, neuron 1 = not hazardous)
    hazardous    = {1}    (CO)

Two evaluation methods per run:
  1. argmax  — pred = argmax(z_out, axis=1)
  2. thresh  — pred = z_out[:, 0] > θ
               θ swept over the unique training-set scores; θ* chosen by
               training accuracy, then evaluated on the test set.

Per (N, s, seed) the script stores: train / test accuracy for both methods,
the selected θ*, and the derived k.

Output: data/hazardous_emission_gridsearch.pkl
"""

import numpy as np
import pickle, os, time
from math import comb

from tools import load


# ── Sweep grid ─────────────────────────────────────────────────────────────
N_LIST = [1, 2, 3, 5, 10, 20, 50, 100, 200]
S_LIST = [0.003, 0.01, 0.017, 0.033, 0.1, 0.167, 0.333, 0.5]  # sparsity; k = round(s * N_HD)
SEEDS  = np.arange(50)                # repeat each (N, k) with several seeds

# ── Fixed config (same as the notebook) ────────────────────────────────────
FILENAME       = '1_600_20_corrected'
N_TEST_SEQ     = 150                    # last 150 presentations -> test
N_HD           = 3000
N_OUT          = 2
P_HD           = 0.025
P              = 1.0
HAZARDOUS_GAS  = {1}
OUT_PKL        = 'data/hazardous_emission_gridsearch.pkl'


# ── Expansion helpers (∂¹+∂² + 56R + 56D = 136) ────────────────────────────
def backward_diff_array(y, h, n):
    coeffs = np.array([(-1) ** kk * comb(n, kk) for kk in range(n + 1)])
    raw = np.convolve(y, coeffs, mode='valid') / h ** n
    return np.concatenate([np.zeros(n), raw])


def expand_with_derivatives(data, h, max_order):
    if max_order == 0:
        return data.copy()
    derivs = [np.apply_along_axis(lambda c: backward_diff_array(c, h, o),
                                  axis=0, arr=data)
              for o in range(1, max_order + 1)]
    return np.hstack([data] + derivs)


def build_expansion(sensor_data, h):
    n_sensors = sensor_data.shape[1]
    all_pairs = [(i, j) for i in range(n_sensors)
                        for j in range(n_sensors) if i != j]
    deriv  = expand_with_derivatives(sensor_data, h, max_order=2)
    ratios = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8)
                              for i, j in all_pairs])
    diffs  = np.column_stack([sensor_data[:, i] - sensor_data[:, j]
                              for i, j in all_pairs])
    return np.hstack([deriv, ratios, diffs])


# ── Load data + frame-level labels / masks ─────────────────────────────────
sensor_data, sequence, times_sec, sequence_sec = load(FILENAME, reduced=True)
h_step = float(np.median(np.diff(times_sec)))
expanded = build_expansion(sensor_data.astype(float), h_step)
n_dense  = expanded.shape[1]
n_frames = expanded.shape[0]
n_seq    = len(sequence)

print(f'sensor_data: {sensor_data.shape}  h={h_step:.3f}s  expanded={expanded.shape}')
print(f'sequence: {n_seq} presentations')

# Frame -> presentation index (-1 if unlabeled / between presentations).
frame_pres = -np.ones(n_frames, dtype=int)
y_frame    = -np.ones(n_frames, dtype=int)   # 0 = hazardous, 1 = not hazardous
for i, (_, gas) in enumerate(sequence):
    if i + 1 < n_seq:
        flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
    else:
        flag = (times_sec > sequence_sec[i])
    frame_pres[flag] = i
    y_frame[flag]    = 0 if int(gas) in HAZARDOUS_GAS else 1

# Test mask: last N_TEST_SEQ presentations.
test_pres_start = n_seq - N_TEST_SEQ
test_mask = (frame_pres >= test_pres_start) & (y_frame >= 0)
y_test    = y_frame[test_mask]
print(f'test  frames: {test_mask.sum()}  '
      f'({(y_test == 0).sum()} hazardous, {(y_test == 1).sum()} not hazardous)')


# ── Train/eval helpers ─────────────────────────────────────────────────────
def train_readout(z_hd, train_idx, y, rng, p, n_out, n_hd, k):
    W_out = np.zeros((n_out, n_hd), dtype=np.float32)
    for i in train_idx:
        active = np.flatnonzero(z_hd[i])
        to_flip = active[rng.random(active.size) < p]
        W_out[y[i], to_flip] = 1.0 / k
    return W_out


def threshold_sweep(score_tr, y_tr_pos, score_te, y_te_pos):
    """Pick θ maximising train accuracy of (score > θ) == y_pos.
    Returns (best_thr, acc_train, acc_test)."""
    cand = np.unique(np.concatenate(([0.0], score_tr, [score_tr.max() + 1e-6,
                                                       score_te.max() + 1e-6])))
    best_acc = -1.0
    best_thr = 0.0
    for thr in cand:
        acc = ((score_tr > thr) == y_tr_pos).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    acc_te = float(((score_te > best_thr) == y_te_pos).mean())
    return best_thr, float(best_acc), acc_te


# ── Sweep ──────────────────────────────────────────────────────────────────
records = []
t0 = time.time()
n_runs_total = len(SEEDS) * len(S_LIST) * len(N_LIST)
n_done = 0

for seed in SEEDS:
    rng_hd = np.random.default_rng(seed)
    W_hd   = rng_hd.binomial(n=1, p=P_HD, size=(N_HD, n_dense)).astype(np.float32)
    x_hd   = expanded @ W_hd.T

    # top-k cache: we recompute argsort once per k (cheap relative to training).
    # Pre-rank once and slice the top-k mask per k.
    neg_ranks = np.argsort(np.argsort(-x_hd, axis=1), axis=1)

    for s in S_LIST:
        k = int(round(s * N_HD))
        z_hd = (neg_ranks < k).astype(np.float32)

        for N in N_LIST:
            # Training mask: first N presentations.
            train_mask = (frame_pres >= 0) & (frame_pres < N) & (y_frame >= 0)
            train_idx  = np.flatnonzero(train_mask)
            y_tr_pos   = (y_frame[train_mask] == 0)

            # Hebbian readout (one rng per (seed, k, N) for reproducibility).
            rng = np.random.default_rng(seed * 100003 + k * 7 + N)
            W_out = train_readout(z_hd, train_idx, y_frame, rng,
                                  P, N_OUT, N_HD, k)

            # Frozen-W_out inference on all frames.
            z_out = z_hd @ W_out.T

            pred_arg = z_out.argmax(axis=1)
            acc_arg_tr = float((pred_arg[train_mask] == y_frame[train_mask]).mean())
            acc_arg_te = float((pred_arg[test_mask]  == y_frame[test_mask]).mean())

            # Threshold sweep on the hazardous neuron's activity.
            score    = z_out[:, 0]
            best_thr, acc_thr_tr, acc_thr_te = threshold_sweep(
                score[train_mask], y_tr_pos,
                score[test_mask],  (y_frame[test_mask] == 0),
            )

            records.append({
                'seed':         seed,
                'N':            N,
                's':            s,
                'k':            k,
                'n_train_frames': int(train_mask.sum()),
                'acc_argmax_train': acc_arg_tr,
                'acc_argmax_test':  acc_arg_te,
                'acc_thresh_train': acc_thr_tr,
                'acc_thresh_test':  acc_thr_te,
                'best_thr':         best_thr,
            })

            n_done += 1
            print(f'[{n_done:3d}/{n_runs_total}]  '
                  f'seed={seed}  N={N:>4d}  s={s:>6.3f}  k={k:>4d}  '
                  f'argmax(tr/te)={acc_arg_tr:.3f}/{acc_arg_te:.3f}  '
                  f'thresh(tr/te)={acc_thr_tr:.3f}/{acc_thr_te:.3f}  '
                  f'θ*={best_thr:.4f}')


# ── Save ───────────────────────────────────────────────────────────────────
results = {
    'records': records,
    'config': {
        'filename':       FILENAME,
        'n_hd':           N_HD,
        'n_out':          N_OUT,
        'p_hd':           P_HD,
        'p':              P,
        'hazardous_gas':  sorted(HAZARDOUS_GAS),
        'n_test_seq':     N_TEST_SEQ,
        'N_list':         N_LIST,
        's_list':         S_LIST,
        'k_list':         [int(round(s * N_HD)) for s in S_LIST],
        'seeds':          SEEDS,
        'expansion':      '∂¹+∂² + 56R + 56D',
        'n_dense':        n_dense,
    },
}

os.makedirs('data', exist_ok=True)
with open(OUT_PKL, 'wb') as f:
    pickle.dump(results, f)

print(f'\nSaved {len(records)} runs to {OUT_PKL}  '
      f'(elapsed {time.time() - t0:.1f}s)')
