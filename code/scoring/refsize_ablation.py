"""Reference-set size ablation for NR-T (latent 1NN scoring).

Fixed scoring rule (k=1, squared euclidean, no compression), only the size of
the reference set changes: 100 / 50 / 25 / 10 percent of the training-normal
latents. Each non-trivial ratio is repeated 5 times with different random
subsets. Scores are evaluated on the same evaluation half used in the paper.

Output: results/refsize_{dataset}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
import recal_fix as R  # noqa: E402

RATIOS = (1.0, 0.5, 0.25, 0.1)
REPS = 5


def knn_d2(ref, q, chunk=512):
    """Squared distance to the nearest reference point, chunked to bound memory.

    float64 is required: with float32 the ``|q|^2 + |r|^2 - 2qr`` form loses
    precision at the 1% tail (catastrophic cancellation produces NaN).
    """
    rn = (ref * ref).sum(1)
    out = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        b = q[i:i + chunk]
        d2 = rn[None, :] - 2.0 * (b @ ref.T)
        d2 += (b * b).sum(1)[:, None]
        out[i:i + chunk] = np.maximum(d2.min(1), 0.0)
    return out


def macro_tpr1(norm_s, atk_s, cat_e, classes):
    tprs, aucs = [], []
    for k in classes:
        m = cat_e == k
        if m.sum() == 0:
            continue
        y = np.concatenate([np.zeros(len(norm_s)), np.ones(int(m.sum()))])
        s = np.concatenate([norm_s, atk_s[m]])
        fpr, tpr, _ = roc_curve(y, s)
        tprs.append(float(tpr[fpr <= 0.01].max()))
        aucs.append(float(roc_auc_score(y, s)))
    return float(np.mean(tprs)), float(np.mean(aucs))


def main(name, reps):
    _, _, _, y, cats = R.load(name)
    _, ite = R.split(y)
    fix = R.CACHE / f'{name}_s42_2025.npz'
    src = fix if fix.exists() else R.CACHE / f'{name}_s42.npz'
    with np.load(src) as z:
        ht, hq = z['ht'].astype(np.float64), z['hq'].astype(np.float64)
    q = hq[ite]
    yte, cte = y[ite], cats[ite]
    n_idx = np.where(yte == 0)[0]
    atk = np.where(yte != 0)[0]
    norm_s = None
    classes = sorted(set(cte[atk].tolist()) - R.NORMAL)

    rows = []
    for ratio in RATIOS:
        n_ref = int(round(len(ht) * ratio))
        n_rep = 1 if ratio == 1.0 else reps
        vals = []
        for rep in range(n_rep):
            if ratio == 1.0:
                ref = ht
            else:
                rng = np.random.RandomState(1000 * int(ratio * 100) + rep)
                ref = ht[np.sort(rng.choice(len(ht), n_ref, replace=False))]
            s = knn_d2(ref, q)
            mtpr, auc = macro_tpr1(s[n_idx], s[atk], cte[atk], classes)
            vals.append((mtpr, auc))
            rows.append((name, ratio, n_ref, rep, mtpr, auc))
            print(f'{name} ref={n_ref:6d} ({ratio:.0%}) rep={rep} '
                  f'mTPR@1%={mtpr:.4f} AUROC={auc:.4f}', flush=True)
        m = np.mean([v[0] for v in vals])
        sd = np.std([v[0] for v in vals])
        print(f'{name} ref={n_ref:6d} ({ratio:.0%}) mean={m:.4f} sd={sd:.4f}', flush=True)

    out = R.TRAE / 'results' / f'refsize_{name}.csv'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('dataset,ratio,n_ref,rep,mtpr1,auroc\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.6f},{r[5]:.6f}\n')
    print('saved', out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ton', 'unsw', 'cic'])
    ap.add_argument('--reps', type=int, default=REPS)
    a = ap.parse_args()
    main(a.dataset, a.reps)
