"""Does target-domain recalibration actually restore a 1% FPR?

UNSW's validation-calibrated threshold gives 5.3% FPR on the test split because
the two normal distributions differ. Here the test normals are split into two
disjoint halves: the threshold is calibrated on one half and the FPR / recall
are measured on the other, so that no data used to set the threshold is reused
for the measurement.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402

CACHE = TRAE / 'cache'
NORMAL = {'normal', 'Normal', 'BENIGN'}


def load(name):
    if name == 'cic':
        x, c, d, y, classes = cic.load()
        y = y.astype(np.int64)
        return x, c, d, y, np.asarray(classes, dtype=str)[y]
    x, c, d, y, cats = base.load(name)
    return x, c, d, y, np.asarray(cats, dtype=str)


def split(y):
    rng = np.random.RandomState(42)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def knn(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0))


def main():
    for name in ('ton', 'unsw', 'cic'):
        x, c, d, y, cats = load(name)
        _, ite = split(y)
        with np.load(CACHE / f'{name}_s42.npz') as z:
            ht, hc, hq = z['ht'], z['hc'], z['hq']
        s = knn(ht, hq)[ite]
        sc = knn(ht, hc)
        yte, cte = y[ite], cats[ite]
        n_idx = np.where(yte == 0)[0]
        atk = np.where(yte != 0)[0]
        cut_val = float(np.quantile(sc, .99, method='higher'))
        fpr_val = float((s[n_idx] > cut_val).mean())
        rec_val = float(np.mean([(s[atk][cte[atk] == k] > cut_val).mean()
                                 for k in sorted(set(cte[atk].tolist()) - NORMAL)]))
        fprs, recs = [], []
        for rep in range(20):
            rng = np.random.RandomState(rep)
            perm = rng.permutation(n_idx)
            a, b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            cut = float(np.quantile(s[a], .99, method='higher'))
            fprs.append(float((s[b] > cut).mean()))
            recs.append(float(np.mean([(s[atk][cte[atk] == k] > cut).mean()
                                       for k in sorted(set(cte[atk].tolist()) - NORMAL)])))
        print(f'{name}: val-cal FPR={fpr_val:.3%} recall={rec_val:.4f} | '
              f'target-cal FPR={np.mean(fprs):.3%}+-{np.std(fprs):.3%} '
              f'recall={np.mean(recs):.4f}', flush=True)


if __name__ == '__main__':
    main()
