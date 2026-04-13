"""
Gridsearch for RBF and polynomial SVM hyperparameters on the
∂¹+∂² + 56R + 56D (136 traces) expansion, using the binary mixture
dataset (mix_100_20_1).

Results are saved to data/gridsearch_svm_136.pkl

Usage:
    python run_gridsearch_svm_136.py
"""

import numpy as np
import pickle, os
from math import comb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from tools import load


pkl_gs = 'data/gridsearch_svm_136.pkl'

if os.path.exists(pkl_gs):
    print(f"{pkl_gs} already exists -- delete to recompute.")
    raise SystemExit(0)


# ── Expansion helpers ─────────────────────────────────────────────────────

def backward_diff_array(y, h, n):
    coeffs = np.array([(-1)**k * comb(n, k) for k in range(n + 1)])
    raw = np.convolve(y, coeffs, mode='valid') / h**n
    return np.concatenate([np.zeros(n), raw])


def expand_with_derivatives(data, h, max_order):
    if max_order == 0:
        return data.copy()
    derivs = [np.apply_along_axis(lambda c: backward_diff_array(c, h, o),
                                  axis=0, arr=data)
              for o in range(1, max_order + 1)]
    return np.hstack([data] + derivs)


n_sensors = 8
rng_pairs = np.random.default_rng(0)
all_ordered_pairs = [(i, j) for i in range(n_sensors)
                     for j in range(n_sensors) if i != j]
ratio_pairs = [all_ordered_pairs[i]
               for i in rng_pairs.permutation(len(all_ordered_pairs))]
diff_pairs  = [all_ordered_pairs[i]
               for i in rng_pairs.permutation(len(all_ordered_pairs))]


def build_136(sensor_data, h):
    x_d12 = expand_with_derivatives(sensor_data, h, 2)
    r56 = np.column_stack([sensor_data[:, i] / (sensor_data[:, j] + 1e-8)
                           for i, j in ratio_pairs[:56]])
    d56 = np.column_stack([sensor_data[:, i] - sensor_data[:, j]
                           for i, j in diff_pairs[:56]])
    return np.hstack([x_d12, r56, d56])


def make_labels(times_sec, sequence, sequence_sec):
    labels = np.zeros_like(times_sec)
    for i in range(len(sequence_sec)):
        try:
            flag = (times_sec > sequence_sec[i]) & (times_sec < sequence_sec[i + 1])
        except IndexError:
            flag = (times_sec > sequence_sec[i])
        labels[flag] = int(sequence[i][1])
    mask = labels > 0
    return labels[mask].astype(int) - 1, mask


# ══════════════════════════════════════════════════════════════════════════
#  Gridsearch on binary mixture dataset (mix_100_20_1)
# ══════════════════════════════════════════════════════════════════════════

print("--- Gridsearch on mix_100_20_1 ---")
sd, seq, ts, ss = load('mix_100_20_1', reduced=True)
h = np.median(np.diff(ts))
y, mask = make_labels(ts, seq, ss)
x = build_136(sd, h)[mask]

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
C_range     = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)

# --- RBF gridsearch ---
print("Running RBF gridsearch ...")
rbf_grid = GridSearchCV(SVC(kernel='rbf'),
                        param_grid=dict(C=C_range, gamma=gamma_range),
                        cv=cv, n_jobs=-1)
rbf_grid.fit(x, y)
print(f"RBF best: C={rbf_grid.best_params_['C']}, "
      f"gamma={rbf_grid.best_params_['gamma']}, "
      f"score={rbf_grid.best_score_:.4f}")

# --- Polynomial gridsearch ---
print("Running polynomial gridsearch ...")
degree_range = [2, 3, 4, 5]
poly_grid = GridSearchCV(SVC(kernel='poly'),
                         param_grid=dict(C=C_range, gamma=gamma_range,
                                         degree=degree_range),
                         cv=cv, n_jobs=-1)
poly_grid.fit(x, y)
print(f"Poly best: C={poly_grid.best_params_['C']}, "
      f"gamma={poly_grid.best_params_['gamma']}, "
      f"degree={poly_grid.best_params_['degree']}, "
      f"score={poly_grid.best_score_:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────

results = {
    'rbf_best_params':  rbf_grid.best_params_,
    'rbf_best_score':   rbf_grid.best_score_,
    'poly_best_params': poly_grid.best_params_,
    'poly_best_score':  poly_grid.best_score_,
    'C_range':          C_range,
    'gamma_range':      gamma_range,
    'degree_range':     degree_range,
    'cv_n_splits':      5,
    'cv_test_size':     0.2,
    'dataset':          'mix_100_20_1',
}

with open(pkl_gs, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to {pkl_gs}")
