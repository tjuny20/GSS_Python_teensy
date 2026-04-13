"""
Compute train + test accuracy for the ∂¹+∂² + 56R + 56D (136 traces) expansion
across classification methods:
  1. Linear SVM (Crammer-Singer, 80/20 stratified, 100 seeds)
  2. RBF SVM (sklearn, 80/20 stratified, 100 seeds, gridsearch params)
  3. Polynomial SVM (sklearn, 80/20 stratified, 100 seeds, gridsearch params)
  4. Expand & sparsify + linear classifier (sequence-based split, best params, 10 seeds)

RBF and polynomial SVM params are loaded from data/gridsearch_svm_136.pkl
(produced by run_gridsearch_svm_136.py).

Results are saved to data/classification_136_train_test.pkl
"""

import numpy as np
import pickle, os, torch
from math import comb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tools import load


pkl_136 = 'data/classification_136_train_test.pkl'

if os.path.exists(pkl_136):
    print(f"{pkl_136} already exists -- delete to recompute.")
    raise SystemExit(0)


# ── Load gridsearch results ──────────────────────────────────────────────

pkl_gs = 'data/gridsearch_svm_136.pkl'
with open(pkl_gs, 'rb') as f:
    gs = pickle.load(f)

rbf_best_C     = gs['rbf_best_params']['C']
rbf_best_gamma = gs['rbf_best_params']['gamma']
poly_best_C      = gs['poly_best_params']['C']
poly_best_gamma  = gs['poly_best_params']['gamma']
poly_best_degree = gs['poly_best_params']['degree']

print(f"RBF params:  C={rbf_best_C}, gamma={rbf_best_gamma}")
print(f"Poly params: C={poly_best_C}, gamma={poly_best_gamma}, "
      f"degree={poly_best_degree}")


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


# ── Crammer-Singer SVM (same as R1 / expand_sparsify_filtered) ────────────

def train_linear_svm(X, y, n_classes, C=1.0, lr=1e-3, n_epochs=200):
    n, d = X.shape
    W = torch.zeros(n_classes, d, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([
        {'params': W, 'weight_decay': 1.0 / (C * n)},
        {'params': b, 'weight_decay': 0.0}
    ], lr=lr)
    idx = torch.arange(n)
    for _ in range(n_epochs):
        scores = X @ W.T + b
        correct_scores = scores[idx, y].unsqueeze(1)
        margins = scores - correct_scores + 1.0
        m = torch.ones_like(scores)
        m[idx, y] = 0.0
        loss = (torch.clamp(margins, min=0) * m).sum() / n
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


@torch.no_grad()
def score_linear_svm(X, y, W, b):
    preds = (X @ W.T + b).argmax(dim=1)
    return (preds == y).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════
#  Linear SVM, RBF SVM & Polynomial SVM  (80/20 stratified split, 100 seeds)
# ══════════════════════════════════════════════════════════════════════════

res_136 = {}
res_136['rbf_best_params']  = dict(C=rbf_best_C, gamma=rbf_best_gamma)
res_136['poly_best_params'] = dict(C=poly_best_C, gamma=poly_best_gamma,
                                   degree=poly_best_degree)
n_repeats_svm = 100

for filename, key in [('1_600_20', 'single'), ('mix_100_20_1', 'mix')]:
    sd, seq, ts, ss = load(filename, reduced=True)
    h = np.median(np.diff(ts))
    y, mask = make_labels(ts, seq, ss)
    x = build_136(sd, h)[mask]
    n_classes = int(y.max()) + 1

    # --- Linear SVM ---
    tr_l, te_l = [], []
    for seed in range(n_repeats_svm):
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        X_tr = torch.tensor(x[tr_idx], dtype=torch.float32)
        X_te = torch.tensor(x[te_idx], dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        y_te = torch.tensor(y[te_idx], dtype=torch.long)
        W, b = train_linear_svm(X_tr, y_tr, n_classes)
        tr_l.append(score_linear_svm(X_tr, y_tr, W, b))
        te_l.append(score_linear_svm(X_te, y_te, W, b))
    res_136[f'lsvm_train_{key}'] = (np.mean(tr_l), np.std(tr_l))
    res_136[f'lsvm_test_{key}']  = (np.mean(te_l), np.std(te_l))
    print(f"Linear SVM {key}: train={np.mean(tr_l):.4f}±{np.std(tr_l):.4f}  "
          f"test={np.mean(te_l):.4f}±{np.std(te_l):.4f}")

    # --- RBF SVM (gridsearch params) ---
    tr_r, te_r = [], []
    for seed in range(n_repeats_svm):
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        clf = SVC(kernel='rbf', C=rbf_best_C, gamma=rbf_best_gamma)
        clf.fit(x[tr_idx], y[tr_idx])
        tr_r.append(clf.score(x[tr_idx], y[tr_idx]))
        te_r.append(clf.score(x[te_idx], y[te_idx]))
    res_136[f'rbf_train_{key}'] = (np.mean(tr_r), np.std(tr_r))
    res_136[f'rbf_test_{key}']  = (np.mean(te_r), np.std(te_r))
    print(f"RBF SVM {key}:    train={np.mean(tr_r):.4f}±{np.std(tr_r):.4f}  "
          f"test={np.mean(te_r):.4f}±{np.std(te_r):.4f}")

    # --- Polynomial SVM (gridsearch params) ---
    tr_p, te_p = [], []
    for seed in range(n_repeats_svm):
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        clf = SVC(kernel='poly', C=poly_best_C, gamma=poly_best_gamma,
                  degree=poly_best_degree)
        clf.fit(x[tr_idx], y[tr_idx])
        tr_p.append(clf.score(x[tr_idx], y[tr_idx]))
        te_p.append(clf.score(x[te_idx], y[te_idx]))
    res_136[f'poly_train_{key}'] = (np.mean(tr_p), np.std(tr_p))
    res_136[f'poly_test_{key}']  = (np.mean(te_p), np.std(te_p))
    print(f"Poly SVM {key}:   train={np.mean(tr_p):.4f}±{np.std(tr_p):.4f}  "
          f"test={np.mean(te_p):.4f}±{np.std(te_p):.4f}")


# ══════════════════════════════════════════════════════════════════════════
#  Expand & sparsify + linear classifier  (sequence-based split)
# ══════════════════════════════════════════════════════════════════════════

CFG = '∂¹+∂² + 56R + 56D'

with open('data/expand_sparsify_filtered_results.pkl', 'rb') as f:
    esf_s = pickle.load(f)
with open('data/expand_sparsify_filtered_mix_results.pkl', 'rb') as f:
    esf_m = pickle.load(f)
n_hd = esf_s['n_hd']
n_repeats_es = esf_s['n_repeats']

best_s = max((k for k in esf_s['acc_table'] if k[0] == CFG),
             key=lambda k: esf_s['acc_table'][k][0])
best_m = max((k for k in esf_m['acc_table'] if k[0] == CFG),
             key=lambda k: esf_m['acc_table'][k][0])

# --- Single gas: sequence-based split at n_train=450 ---
sd, seq, ts, ss = load('1_600_20', reduced=True)
h = np.median(np.diff(ts))
x_full = build_136(sd, h)
n_dense = x_full.shape[1]
n_train_seq = 450

labels_full = np.zeros_like(ts)
for i in range(len(ss)):
    try:
        flag = (ts > ss[i]) & (ts < ss[i + 1])
    except IndexError:
        flag = (ts > ss[i])
    labels_full[flag] = int(seq[i][1])

train_frame = np.zeros(len(ts), dtype=bool)
test_frame = np.zeros(len(ts), dtype=bool)
for i in range(len(ss)):
    try:
        flag = (ts > ss[i]) & (ts < ss[i + 1])
    except IndexError:
        flag = (ts > ss[i])
    if labels_full[flag].any():
        if i < n_train_seq:
            train_frame |= flag
        else:
            test_frame |= flag
train_labeled = train_frame & (labels_full > 0)
test_labeled = test_frame & (labels_full > 0)
y_tr_s = torch.tensor(labels_full[train_labeled].astype(int) - 1, dtype=torch.long)
y_te_s = torch.tensor(labels_full[test_labeled].astype(int) - 1, dtype=torch.long)
n_cls_s = max(int(y_tr_s.max()), int(y_te_s.max())) + 1

p_hd_s, d_s = best_s[1], best_s[2]
k_s = int(d_s * n_hd)
print(f"\nES linear single: p_hd={p_hd_s}, d={d_s}, k={k_s}")
tr_es, te_es = [], []
for seed in range(n_repeats_es):
    torch.manual_seed(seed)
    W_hd = torch.bernoulli(torch.full((n_hd, n_dense), p_hd_s))
    xh = torch.tensor(x_full[train_labeled], dtype=torch.float32) @ W_hd.T
    _, topk = torch.topk(xh, k_s, dim=1)
    z_tr = torch.zeros_like(xh).scatter_(1, topk, 1.0)
    xh = torch.tensor(x_full[test_labeled], dtype=torch.float32) @ W_hd.T
    _, topk = torch.topk(xh, k_s, dim=1)
    z_te = torch.zeros_like(xh).scatter_(1, topk, 1.0)
    W, b = train_linear_svm(z_tr, y_tr_s, n_cls_s)
    tr_es.append(score_linear_svm(z_tr, y_tr_s, W, b))
    te_es.append(score_linear_svm(z_te, y_te_s, W, b))
res_136['es_train_single'] = (np.mean(tr_es), np.std(tr_es))
res_136['es_test_single']  = (np.mean(te_es), np.std(te_es))
print(f"ES linear single: train={np.mean(tr_es):.4f}±{np.std(tr_es):.4f}  "
      f"test={np.mean(te_es):.4f}±{np.std(te_es):.4f}")

# --- Mix: separate train/test files ---
sd_tr, seq_tr, ts_tr, ss_tr = load('mix_100_20_1', reduced=True)
sd_te, seq_te, ts_te, ss_te = load('mix_50_20_1', reduced=True)
h_tr = np.median(np.diff(ts_tr))
h_te = np.median(np.diff(ts_te))
x_tr_full = build_136(sd_tr, h_tr)
x_te_full = build_136(sd_te, h_te)

lab_tr = np.zeros_like(ts_tr)
for i in range(len(ss_tr)):
    try:
        flag = (ts_tr > ss_tr[i]) & (ts_tr < ss_tr[i + 1])
    except IndexError:
        flag = (ts_tr > ss_tr[i])
    lab_tr[flag] = int(seq_tr[i][1])
lab_te = np.zeros_like(ts_te)
for i in range(len(ss_te)):
    try:
        flag = (ts_te > ss_te[i]) & (ts_te < ss_te[i + 1])
    except IndexError:
        flag = (ts_te > ss_te[i])
    lab_te[flag] = int(seq_te[i][1])

tr_mask_m = lab_tr > 0
te_mask_m = lab_te > 0
y_tr_m = torch.tensor(lab_tr[tr_mask_m].astype(int) - 1, dtype=torch.long)
y_te_m = torch.tensor(lab_te[te_mask_m].astype(int) - 1, dtype=torch.long)
n_cls_m = max(int(y_tr_m.max()), int(y_te_m.max())) + 1

p_hd_m, d_m = best_m[1], best_m[2]
k_m = int(d_m * n_hd)
print(f"ES linear mix: p_hd={p_hd_m}, d={d_m}, k={k_m}")
tr_es, te_es = [], []
for seed in range(n_repeats_es):
    torch.manual_seed(seed)
    W_hd = torch.bernoulli(torch.full((n_hd, x_tr_full.shape[1]), p_hd_m))
    xh = torch.tensor(x_tr_full[tr_mask_m], dtype=torch.float32) @ W_hd.T
    _, topk = torch.topk(xh, k_m, dim=1)
    z_tr = torch.zeros_like(xh).scatter_(1, topk, 1.0)
    xh = torch.tensor(x_te_full[te_mask_m], dtype=torch.float32) @ W_hd.T
    _, topk = torch.topk(xh, k_m, dim=1)
    z_te = torch.zeros_like(xh).scatter_(1, topk, 1.0)
    W, b = train_linear_svm(z_tr, y_tr_m, n_cls_m)
    tr_es.append(score_linear_svm(z_tr, y_tr_m, W, b))
    te_es.append(score_linear_svm(z_te, y_te_m, W, b))
res_136['es_train_mix'] = (np.mean(tr_es), np.std(tr_es))
res_136['es_test_mix']  = (np.mean(te_es), np.std(te_es))
print(f"ES linear mix:    train={np.mean(tr_es):.4f}±{np.std(tr_es):.4f}  "
      f"test={np.mean(te_es):.4f}±{np.std(te_es):.4f}")


# ── Save ──────────────────────────────────────────────────────────────────

with open(pkl_136, 'wb') as f:
    pickle.dump(res_136, f)
print(f"\nSaved to {pkl_136}")
