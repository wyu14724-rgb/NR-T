"""External baselines under the same low-FPR protocol.

Unsupervised models are fit on training normals only.
Supervised models are fit on a stratified 50% half of the test set (normals + attacks)
and evaluated, together with all unsupervised models, on the remaining half.

Metrics per attack class vs normals: TPR at FPR<=0.1%, TPR at FPR<=1%, AUROC, AUPR.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM

TRAE = Path(__file__).resolve().parents[2]
OUT = TRAE
RES = OUT / 'results'
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402

SEED = 42
OCSVM_FIT = 8000
SUP_CAP = 80000
DSVDD_EPOCHS = 30


def get_data(name):
    if name == 'cic':
        x, c, d, y, types = cic.load()
        return x, c, d, y.astype(np.int64), np.asarray(types, dtype=str)[y]
    x, c, d, y, cats = base.load(name)
    return x, c, d, y, np.asarray(cats, dtype=str)


def macro_metrics(y, cats, s):
    normal = y == 0
    per = {}
    for k in sorted(set(cats.tolist()) - {'normal', 'Normal', 'BENIGN'}):
        m = cats == k
        pair = normal | m
        yb = m[pair].astype(int)
        sc = s[pair]
        fpr, tpr, _ = roc_curve(yb, sc)
        per[k] = {
            'tpr_0_1': float(tpr[fpr <= 0.001].max()) if (fpr <= 0.001).any() else 0.0,
            'tpr_1': float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0,
            'auroc': float(roc_auc_score(yb, sc)),
            'aupr': float(average_precision_score(yb, sc)),
            'n': int(m.sum()),
        }
    keys = ['tpr_0_1', 'tpr_1', 'auroc', 'aupr']
    return {f'macro_{k}': float(np.mean([v[k] for v in per.values()])) for k in keys}, per


class DSVDD(nn.Module):
    def __init__(self, d_in, d_lat=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, d_lat))
        self.c = None

    def forward(self, x):
        return self.net(x)


def fit_dsvdd(x, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DSVDD(x.shape[1])
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        m.c = m(xt[:1000]).mean(0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-6)
    n = len(xt)
    bs = 512
    for ep in range(DSVDD_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            z = m(xt[idx])
            loss = ((z - m.c) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def score_dsvdd(m, x):
    with torch.no_grad():
        z = m(torch.tensor(x, dtype=torch.float32))
        return ((z - m.c) ** 2).sum(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ton')
    a = ap.parse_args()
    name = a.dataset

    t0 = time.time()
    x, c, d, y, cats = get_data(name)
    rng = np.random.RandomState(SEED)

    # stratified half split of the test set
    idx_tr, idx_te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        idx_tr.append(ix[:h])
        idx_te.append(ix[h:])
    itr, ite = np.concatenate(idx_tr), np.concatenate(idx_te)
    Xtr, ytr, ctr = d[itr], y[itr], cats[itr]
    Xte, yte, cte = d[ite], y[ite], cats[ite]
    print(f"[{name}] train_normals={len(x)} test_half={len(Xte)} "
          f"sup_train={len(Xtr)} d={d.shape[1]}", flush=True)

    rows = []
    models = {}

    t = time.time()
    ifo = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(x)
    models['IF'] = -ifo.score_samples(Xte)
    print(f"  IF {time.time()-t:.1f}s", flush=True)

    t = time.time()
    sub = x[rng.choice(len(x), size=min(OCSVM_FIT, len(x)), replace=False)]
    oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05).fit(sub)
    models['OCSVM'] = -oc.decision_function(Xte)
    print(f"  OCSVM {time.time()-t:.1f}s (fit n={len(sub)})", flush=True)

    t = time.time()
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(sub)
    ocz = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05).fit(sc.transform(sub))
    models['OCSVM_z'] = -ocz.decision_function(sc.transform(Xte))
    print(f"  OCSVM_z {time.time()-t:.1f}s", flush=True)

    t = time.time()
    m = fit_dsvdd(x)
    models['DeepSVDD'] = score_dsvdd(m, Xte)
    print(f"  DeepSVDD {time.time()-t:.1f}s", flush=True)

    Xtr_s = Xtr[:SUP_CAP]
    ytr_b = (ytr[:SUP_CAP] != 0).astype(int)
    t = time.time()
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr_s, ytr_b)
    models['RF_sup'] = rf.predict_proba(Xte)[:, 1]
    print(f"  RF_sup {time.time()-t:.1f}s", flush=True)

    t = time.time()
    hgb = HistGradientBoostingClassifier(random_state=SEED, max_iter=200).fit(Xtr_s, ytr_b)
    models['HGB_sup'] = hgb.predict_proba(Xte)[:, 1]
    print(f"  HGB_sup {time.time()-t:.1f}s", flush=True)

    t = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=60, random_state=SEED,
                        early_stopping=True).fit(Xtr_s, ytr_b)
    models['MLP_sup'] = mlp.predict_proba(Xte)[:, 1]
    print(f"  MLP_sup {time.time()-t:.1f}s", flush=True)

    for mname, s in models.items():
        agg, per = macro_metrics(yte, cte, s)
        rows.append({'dataset': name, 'model': mname, **{k: round(v, 6) for k, v in agg.items()},
                     'n_test': len(Xte)})
        (RES / f'perclass_{name}_{mname}.json').write_text(
            json.dumps(per, indent=2, ensure_ascii=False))
        print(f"{name:>4s} {mname:10s} tpr0.1={agg['macro_tpr_0_1']:.4f} "
              f"tpr1={agg['macro_tpr_1']:.4f} auroc={agg['macro_auroc']:.4f} "
              f"aupr={agg['macro_aupr']:.4f}", flush=True)
    (RES / f'baselines_{name}.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"[{name}] total {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
