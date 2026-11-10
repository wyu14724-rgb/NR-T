"""Development-half selection for the external unsupervised baselines (ToN / CIC).

Mirrors `baselines_unsw_minmax.py`: each baseline's configuration is chosen on the
baseline-training half of the test set (never used for any reported evaluation
number), then scored on the shared evaluation half.

Inputs are the same processed arrays the DRL checkpoints and NR-T use, via
`baselines.get_data(name)`; unlike UNSW-NB15 these already match the checkpoint
preprocessing (one-hot + min-max).

Usage: python baselines_dev_sel.py --dataset ton
       python baselines_dev_sel.py --dataset cic
Outputs: results/baselines_dev_<name>.csv / .json
"""
import argparse
import json
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
from sklearn.svm import OneClassSVM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import get_data, macro_metrics  # noqa: E402

SEED = 42
OCSVM_FIT = 8000


class DSVDD(nn.Module):
    def __init__(self, d_in, d_lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, d_lat))
        self.c = None

    def forward(self, x):
        return self.net(x)


def fit_dsvdd(x, d_lat, epochs=30, seed=SEED):
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
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 65536):
            z = m(torch.tensor(x[i:i + 65536], dtype=torch.float32))
            out.append(((z - m.c) ** 2).sum(1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ton', 'cic'])
    name = ap.parse_args().dataset
    t0 = time.time()

    x, _, d, y, cats = get_data(name)
    rng = np.random.RandomState(SEED)
    idx_tr, idx_te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        idx_tr.append(ix[:h])
        idx_te.append(ix[h:])
    itr, ite = np.concatenate(idx_tr), np.concatenate(idx_te)
    X_d, y_d, c_d = d[itr], y[itr], cats[itr]
    X_e, y_e, c_e = d[ite], y[ite], cats[ite]
    print(f'[{name}] fit_normals={x.shape} dev={X_d.shape} eval={X_e.shape}', flush=True)

    sub = x[rng.choice(len(x), size=min(OCSVM_FIT, len(x)), replace=False)]
    rows, chosen = [], {}

    best = None
    for n_est in (100, 200, 300):
        t = time.time()
        m = IsolationForest(n_estimators=n_est, random_state=SEED, n_jobs=-1).fit(x)
        sd, se = -m.score_samples(X_d), -m.score_samples(X_e)
        agg, _ = macro_metrics(y_d, c_d, sd)
        print(f'  IF n={n_est} dev_tpr1={agg["macro_tpr_1"]:.4f} ({time.time()-t:.1f}s)', flush=True)
        if best is None or agg['macro_tpr_1'] > best[0]:
            best = (agg['macro_tpr_1'], n_est, sd, se)
    chosen['IF'] = f'n_estimators={best[1]}'
    dev_if, eval_if = best[2], best[3]

    best = None
    for nu in (0.01, 0.05, 0.1):
        t = time.time()
        m = OneClassSVM(kernel='rbf', gamma='scale', nu=nu).fit(sub)
        sd, se = -m.decision_function(X_d), -m.decision_function(X_e)
        agg, _ = macro_metrics(y_d, c_d, sd)
        print(f'  OCSVM nu={nu} dev_tpr1={agg["macro_tpr_1"]:.4f} ({time.time()-t:.1f}s)', flush=True)
        if best is None or agg['macro_tpr_1'] > best[0]:
            best = (agg['macro_tpr_1'], nu, sd, se)
    chosen['OCSVM'] = f'nu={best[1]}, gamma=scale, fit_n={len(sub)}'
    dev_oc, eval_oc = best[2], best[3]

    best = None
    for d_lat in (16, 32):
        t = time.time()
        m = fit_dsvdd(x, d_lat)
        sd, se = score_dsvdd(m, X_d), score_dsvdd(m, X_e)
        agg, _ = macro_metrics(y_d, c_d, sd)
        print(f'  DeepSVDD d_lat={d_lat} dev_tpr1={agg["macro_tpr_1"]:.4f} ({time.time()-t:.1f}s)',
              flush=True)
        if best is None or agg['macro_tpr_1'] > best[0]:
            best = (agg['macro_tpr_1'], d_lat, sd, se)
    chosen['DeepSVDD'] = f'd_lat={best[1]}, epochs=30'
    dev_ds, eval_ds = best[2], best[3]

    for mname, sd, se in (('IF', dev_if, eval_if), ('OCSVM', dev_oc, eval_oc),
                          ('DeepSVDD', dev_ds, eval_ds)):
        agg_d, _ = macro_metrics(y_d, c_d, sd)
        agg_e, _ = macro_metrics(y_e, c_e, se)
        rows.append({'dataset': name, 'model': mname, 'config': chosen[mname],
                     'dev_tpr1': round(agg_d['macro_tpr_1'], 6),
                     'dev_auroc': round(agg_d['macro_auroc'], 6),
                     'eval_tpr1': round(agg_e['macro_tpr_1'], 6),
                     'eval_auroc': round(agg_e['macro_auroc'], 6)})
        print(f'{name} {mname:9s} dev={agg_d["macro_tpr_1"]:.4f} eval={agg_e["macro_tpr_1"]:.4f} '
              f'auroc={agg_e["macro_auroc"]:.4f}', flush=True)

    (HERE.parent / 'results' / f'baselines_dev_{name}.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with (HERE.parent / 'results' / f'baselines_dev_{name}.csv').open('w', encoding='utf-8') as f:
        f.write('dataset,model,config,dev_tpr1,dev_auroc,eval_tpr1,eval_auroc\n')
        for r in rows:
            f.write(f'{r["dataset"]},{r["model"]},"{r["config"]}",{r["dev_tpr1"]:.6f},'
                    f'{r["dev_auroc"]:.6f},{r["eval_tpr1"]:.6f},{r["eval_auroc"]:.6f}\n')
    print(f'[{name}] total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
