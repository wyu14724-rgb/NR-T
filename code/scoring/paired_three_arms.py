"""Paired three-arm comparison on one checkpoint per (dataset, seed).

Arms, all scored with the same reference set (training normals) and the same
calibration set (held-out normals), all evaluated on the same held-out half:

  1. native   original DRL score of the checkpoint (subspace residual)
  2. raw1nn   1-NN distance in the preprocessed input space
  3. lat1nn   1-NN distance in the checkpoint's 128-d latent

For every arm we report both metric families:
  * macro_tpr1    per-class ROC, TPR at FPR<=1% (threshold-free)
  * recall99/fpr  recall and normal FPR at the 99th percentile of the
                  calibration scores (a concrete operating point)

The checkpoint is trained once per (dataset, seed) with the frozen harness and
both the residual and the latent are read from that same checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402
from baselines import macro_metrics  # noqa: E402

RES = TRAE / 'results'
SPLIT_SEED = 42
SEEDS = (42, 43, 44)


def split(y):
    rng = np.random.RandomState(SPLIT_SEED)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def load(name):
    if name == 'cic':
        x, c, d, y, classes = cic.load()
        return x, c, d, y.astype(np.int64), np.asarray(classes, dtype=str)[y]
    x, c, d, y, cats = base.load(name)
    return x, c, d, y, np.asarray(cats, dtype=str)


def train(name, x_train, seed):
    if name == 'cic':
        return cic.train_drl(x_train, seed, verbose=False)
    model, _ = base.train_model(name, seed, 0, 0, verbose=False)
    return model


def encode(name, model, x):
    return cic.embed(model, x) if name == 'cic' else base.encode(model, x)


def native(name, model, x):
    return cic.native_scores(model, x) if name == 'cic' else base.native(model, x, name)


def nn_scores(ref, cal, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return (np.sqrt(np.maximum(nn.kneighbors(cal)[0][:, 0], 0)),
            np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0)))


NORMAL = {'normal', 'Normal', 'BENIGN'}


def evaluate(s, cal, y, cats):
    """ROC-based macro TPR@1% plus the operating point fixed by the cal 99% quantile."""
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    normal = y == 0
    rec = [float(pred[cats == c].mean())
           for c in sorted(set(cats.tolist()) - NORMAL)]
    return {'macro_tpr1': agg['macro_tpr_1'], 'macro_auroc': agg['macro_auroc'],
            'recall99': float(np.mean(rec)), 'fpr': float(pred[normal].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='ton,unsw,cic')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    names = [s for s in a.datasets.split(',') if s]
    seeds = [int(s) for s in a.seeds.split(',') if s]

    rows = []
    for name in names:
        x, c, d, y, cats = load(name)
        _, ite = split(y)
        yte, cte = y[ite], cats[ite]
        for seed in seeds:
            model = train(name, x, seed)
            arms = {
                'native': (native(name, model, c), native(name, model, d)),
                'raw1nn': nn_scores(x, c, d),
                'lat1nn': nn_scores(encode(name, model, x), encode(name, model, c),
                                    encode(name, model, d)),
            }
            for arm, (sc, sq) in arms.items():
                met = evaluate(sq[ite], sc, yte, cte)
                rows.append({'dataset': name, 'seed': seed, 'arm': arm,
                             'macro_tpr1': round(met['macro_tpr1'], 6),
                             'macro_auroc': round(met['macro_auroc'], 6),
                             'recall99': round(met['recall99'], 6),
                             'fpr': round(met['fpr'], 6)})
                print(f"{name:>4s} s{seed} {arm:7s} tpr1={met['macro_tpr1']:.4f} "
                      f"auroc={met['macro_auroc']:.4f} recall99={met['recall99']:.4f} "
                      f"fpr={met['fpr']:.4f}", flush=True)

    out = RES / ('paired_three_arms' + (f'_{a.tag}' if a.tag else '') + '.csv')
    keys = list(rows[0])
    with out.open('w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    print(f'wrote {out}', flush=True)

    for name in names:
        for arm in ('native', 'raw1nn', 'lat1nn'):
            v = [r['macro_tpr1'] for r in rows if r['dataset'] == name and r['arm'] == arm]
            if v:
                sd = float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
                print(f'[{name}] {arm:7s} mean={np.mean(v):.4f} std={sd:.4f} '
                      f'seeds={len(v)}', flush=True)


if __name__ == '__main__':
    main()
