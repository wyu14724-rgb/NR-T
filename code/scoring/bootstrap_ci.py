"""Stratified bootstrap CI for the paired scoring comparison (seed 42).

Resamples the evaluation normals and each attack class with replacement,
recomputes the single-threshold macro TPR at empirical FPR<=1% for both arms
on the *same* resampled indices (paired), and reports percentile CIs.

Scope: this isolates evaluation-sample uncertainty for one checkpoint.  The
checkpoint-level spread is reported separately (three training seeds).

Output: results/bootstrap_ci.csv
"""
import csv

import numpy as np

from calib_stress import NORMAL, RES, split
from single_threshold import optimal_tau

ALPHA = 0.01
NREP = 2000
RNG = np.random.default_rng(20261108)


def macro(norm, cls, tau):
    return float(np.mean([np.mean(s > tau) for s in cls]))


if __name__ == '__main__':
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        z = np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False)
        y, cats = z['y_test'], z['cats_test'].astype(str)
        _, ite = split(y)
        ye, ce = y[ite], cats[ite]
        norm_i = np.where(ye == 0)[0]
        cls_i = [np.where(ce == c)[0] for c in sorted(set(ce.tolist()) - NORMAL)]

        point, boot = {}, {}
        for arm in ('native', 'lat1nn'):
            s = z[f'test_{arm}'][ite]
            tau0 = optimal_tau(s[norm_i])
            point[arm] = macro(None, [s[i] for i in cls_i], tau0)
            rec = np.empty(NREP)
            for b in range(NREP):
                nn = RNG.integers(0, len(norm_i), len(norm_i))
                tau = optimal_tau(s[norm_i][nn])
                rec[b] = np.mean([
                    np.mean(s[i][RNG.integers(0, len(i), len(i))] > tau)
                    for i in cls_i])
            boot[arm] = rec
        d = boot['lat1nn'] - boot['native']
        lo, hi = np.percentile(d, [2.5, 97.5])
        for arm in ('native', 'lat1nn'):
            b = boot[arm]
            rows.append((name, arm, f'{point[arm]:.4f}',
                         f'{np.percentile(b, 2.5):.4f}',
                         f'{np.percentile(b, 97.5):.4f}'))
            print(f'{name:>4s} {arm:7s} {point[arm]:.4f} '
                  f'[{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]',
                  flush=True)
        rows.append((name, 'diff', f'{np.mean(d):+.4f}',
                     f'{lo:+.4f}', f'{hi:+.4f}'))
        print(f'{name:>4s} diff    {np.mean(d):+.4f} '
              f'[{lo:+.4f}, {hi:+.4f}] P(d<=0)={float((d <= 0).mean()):.4f}',
              flush=True)

    out = RES / 'bootstrap_ci.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'stat', 'point', 'lo95', 'hi95'])
        w.writerows(rows)
    print(f'wrote {out}')
