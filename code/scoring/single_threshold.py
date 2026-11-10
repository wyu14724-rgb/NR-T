"""Single global threshold vs per-class ROC thresholds (reviewer point 2).

mTPR@1% allows each attack class to pick its own threshold tau_c subject to
FPR(tau_c) <= 1%.  Because FPR depends only on the normal sample and TPR_c is
monotone in tau, the feasible set is the same for every class, so the optimum
should collapse to one shared threshold.  This script checks that claim
explicitly (prints the per-class thresholds) and then reports the same cells
under three threshold sources:

  roc_per_class   sklearn roc_curve per class, tau_c free (paper §5.3)
  single_oracle   one tau from the evaluation normals at FPR<=1%
  single_calib    one tau from the calibration normal batch (deployment rule)

Output: results/single_threshold.csv
"""
import csv

import numpy as np
from sklearn.metrics import roc_curve

from calib_stress import CAP, NORMAL, RES, op_point, split

ARMS = ('native', 'raw1nn', 'lat1nn')
ALPHA = 0.01


def per_class_roc(norm, sq, cats):
    """Return (macro TPR, list of thresholds) with one tau per attack class."""
    tprs, thrs = [], []
    for c in sorted(set(cats.tolist()) - NORMAL):
        m = cats == c
        if not m.any():
            continue
        ysc = np.concatenate([np.zeros(len(norm)), np.ones(int(m.sum()))])
        s = np.concatenate([norm, sq[m]])
        fpr, tpr, thr = roc_curve(ysc, s, drop_intermediate=False)
        ok = np.where(fpr <= ALPHA)[0]
        i = ok[np.argmax(tpr[ok])]
        tprs.append(float(tpr[i]))
        thrs.append(float(thr[i]))
    return float(np.mean(tprs)), thrs


def optimal_tau(norm, alpha=ALPHA):
    """Smallest threshold whose empirical FPR is <= alpha (strict > rule).

    The feasible set {tau: FPR(tau) <= alpha} depends only on the normal
    scores, and TPR_c is non-increasing in tau, so this single value attains
    the per-class maximum of every attack class simultaneously.
    """
    n = len(norm)
    allowed = int(np.floor(alpha * n))
    return float(np.partition(norm, n - allowed - 1)[n - allowed - 1])


if __name__ == '__main__':
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        z = np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False)
        y, cats = z['y_test'], z['cats_test'].astype(str)
        _, ite = split(y)
        ye, ce = y[ite], cats[ite]
        for arm in ARMS:
            sc, sq = z[f'cal_{arm}'][:CAP[name]], z[f'test_{arm}'][ite]
            norm = sq[ye == 0]
            rm, _ = per_class_roc(norm, sq, ce)
            oracle = optimal_tau(norm)
            calib = float(np.quantile(sc, 1 - ALPHA, method='higher'))
            ro, fo = op_point(ye, ce, sq > oracle)
            rc, fc = op_point(ye, ce, sq > calib)
            rows += [(name, arm, 'roc_per_class', rm, 1 - ALPHA),
                     (name, arm, 'single_optimal', ro, fo),
                     (name, arm, 'single_calib', rc, fc)]
            print(f'{name:>4s} {arm:7s} roc={rm:.4f} '
                  f'single={ro:.4f}/{fo:.4f} calib={rc:.4f}/{fc:.4f} '
                  f'diff={ro - rm:+.4f}', flush=True)

    out = RES / 'single_threshold.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'arm', 'threshold', 'macro_recall', 'fpr'])
        w.writerows(rows)
    print(f'wrote {out}')
