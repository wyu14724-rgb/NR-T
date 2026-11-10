"""External unsupervised baselines on UNSW-NB15 with the checkpoint's preprocessing.

The main table's IF / OC-SVM / Deep SVDD numbers came from the StandardScaler
arrays in DRL_UNSWNB15/prepared/data.npz, while the DRL checkpoints and NR-T use
MinMax-scaled inputs. This script re-runs the three unsupervised baselines on the
same MinMax inputs as `unsw_fix_2025.py`, selects each baseline's configuration
on a development half (the baseline-training half of the test file, which is not
used for any reported evaluation number), and reports the selected model on the
shared evaluation half.

Outputs: results/baselines_unsw_minmax.csv / .json
"""
import os
import sys
import time
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unsw_fix_2025 import RES, build, split  # noqa: E402

SEED = 42
OCSVM_FIT = 8000


def macro_metrics(y, cats, s):
    normal = y == 0
    out = {}
    per = {}
    for k in sorted(set(cats.tolist()) - {'normal', 'Normal', 'BENIGN'}):
        m = cats == k
        pair = normal | m
        yb = m[pair].astype(int)
        fpr, tpr, _ = roc_curve(yb, s[pair])
        per[k] = (float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0,
                  float(roc_auc_score(yb, s[pair])))
    out['tpr1'] = float(np.mean([v[0] for v in per.values()]))
    out['auroc'] = float(np.mean([v[1] for v in per.values()]))
    return out, per


class DSVDD(nn.Module):
    def __init__(self, d_in, d_lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, d_lat))
        self.c = None

    def forward(self, x):
        return self.net(x)


def fit_dsvdd(x, d_lat, epochs, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DSVDD(x.shape[1], d_lat)
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        m.c = m(xt[:1000]).mean(0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-6)
    n = len(xt)
    bs = 512
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            z = m(xt[perm[i:i + bs]])
            loss = ((z - m.c) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def score_dsvdd(m, x):
    with torch.no_grad():
        z = m(torch.tensor(x, dtype=torch.float32))
        return ((z - m.c) ** 2).sum(1).numpy()


def main():
    t0 = time.time()
    x, _, x_te, y, cats = build()
    itr, ite = split(y)          # itr: development half, ite: evaluation half
    y_d, c_d, X_d = y[itr], cats[itr], x_te[itr]
    y_e, c_e, X_e = y[ite], cats[ite], x_te[ite]
    rng = np.random.RandomState(SEED)
    sub = x[rng.choice(len(x), size=min(OCSVM_FIT, len(x)), replace=False)]
    print(f'x={x.shape} dev={X_d.shape} eval={X_e.shape}', flush=True)

    dev_scores, eval_scores, chosen = {}, {}, {}

    # Isolation Forest: configuration by dev macro TPR@1%
    best = None
    for n_est in (100, 200, 300):
        t = time.time()
        m = IsolationForest(n_estimators=n_est, random_state=SEED, n_jobs=-1).fit(x)
        sd, se = -m.score_samples(X_d), -m.score_samples(X_e)
        agg_d, _ = macro_metrics(y_d, c_d, sd)
        print(f'  IF n={n_est} dev_tpr1={agg_d["tpr1"]:.4f} ({time.time()-t:.1f}s)', flush=True)
        if best is None or agg_d['tpr1'] > best[0]:
            best = (agg_d['tpr1'], n_est, sd, se)
    chosen['IF'] = f'n_estimators={best[1]}'
    dev_scores['IF'], eval_scores['IF'] = best[2], best[3]

    # OC-SVM (RBF, gamma=scale), fit on a subsample of training normals
    best = None
    for nu in (0.01, 0.05, 0.1):
        t = time.time()
        m = OneClassSVM(kernel='rbf', gamma='scale', nu=nu).fit(sub)
        sd, se = -m.decision_function(X_d), -m.decision_function(X_e)
        agg_d, _ = macro_metrics(y_d, c_d, sd)
        print(f'  OCSVM nu={nu} dev_tpr1={agg_d["tpr1"]:.4f} ({time.time()-t:.1f}s)', flush=True)
        if best is None or agg_d['tpr1'] > best[0]:
            best = (agg_d['tpr1'], nu, sd, se)
    chosen['OCSVM'] = f'nu={best[1]}, gamma=scale, fit_n={len(sub)}'
    dev_scores['OCSVM'], eval_scores['OCSVM'] = best[2], best[3]

    # Deep SVDD: latent width selected on dev
    best = None
    for d_lat in (16, 32):
        t = time.time()
        m = fit_dsvdd(x, d_lat, 30)
        sd, se = score_dsvdd(m, X_d), score_dsvdd(m, X_e)
        agg_d, _ = macro_metrics(y_d, c_d, sd)
        print(f'  DeepSVDD d_lat={d_lat} dev_tpr1={agg_d["tpr1"]:.4f} ({time.time()-t:.1f}s)',
              flush=True)
        if best is None or agg_d['tpr1'] > best[0]:
            best = (agg_d['tpr1'], d_lat, sd, se)
    chosen['DeepSVDD'] = f'd_lat={best[1]}, epochs=30'
    dev_scores['DeepSVDD'], eval_scores['DeepSVDD'] = best[2], best[3]

    rows = []
    for name in ('IF', 'OCSVM', 'DeepSVDD'):
        agg_d, _ = macro_metrics(y_d, c_d, dev_scores[name])
        agg_e, _ = macro_metrics(y_e, c_e, eval_scores[name])
        rows.append({'model': name, 'config': chosen[name],
                     'dev_tpr1': round(agg_d['tpr1'], 6), 'dev_auroc': round(agg_d['auroc'], 6),
                     'eval_tpr1': round(agg_e['tpr1'], 6), 'eval_auroc': round(agg_e['auroc'], 6)})
        print(f'{name:9s} dev={agg_d["tpr1"]:.4f} eval={agg_e["tpr1"]:.4f} '
              f'auroc={agg_e["auroc"]:.4f}', flush=True)

    import json
    (RES / 'baselines_unsw_minmax.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with (RES / 'baselines_unsw_minmax.csv').open('w', encoding='utf-8') as f:
        f.write('model,config,dev_tpr1,dev_auroc,eval_tpr1,eval_auroc\n')
        for r in rows:
            f.write(f'{r["model"]},"{r["config"]}",{r["dev_tpr1"]:.6f},{r["dev_auroc"]:.6f},'
                    f'{r["eval_tpr1"]:.6f},{r["eval_auroc"]:.6f}\n')
    print(f'total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
