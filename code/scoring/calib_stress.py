"""Stress tests for the target-domain recalibration step.

Two questions, both answered from results/scores_<name>_s42.npz:

1. contamination  the calibration batch is assumed all-normal. What happens to
                  the operating point if 1% / 5% / 10% of it are attacks?
2. leakage        if the threshold is estimated on the very normals used to
                  report FPR, how optimistic does the reported FPR become?

Output: results/calib_stress.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

TRAE = Path(__file__).resolve().parents[2]
RES = TRAE / 'results'
ARMS = ('native', 'raw1nn', 'lat1nn')
NORMAL = {'normal', 'Normal', 'BENIGN'}
RATES = (0.0, 0.01, 0.05, 0.10)
REPS = 20
# source-domain calibration size frozen in the paper (§4.4 / §5.4)
CAP = {'ton': 4999, 'unsw': 2800, 'cic': 10000}


def split(y, seed=42):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def op_point(y, cats, pred, valid=None):
    """Macro recall over attack classes and normal FPR, computed only on rows
    flagged in ``valid`` (excluded rows leave both numerator and denominator)."""
    if valid is None:
        valid = np.ones(len(y), bool)
    rec = []
    for c in sorted(set(cats.tolist()) - NORMAL):
        m = (cats == c) & valid
        if m.any():
            rec.append(float(pred[m].mean()))
    fn = (y == 0) & valid
    return float(np.mean(rec)), float(pred[fn].mean())


if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ton', 'unsw', 'cic']
    rows = []
    for name in names:
        z = np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False)
        y, cats = z['y_test'], z['cats_test'].astype(str)
        _, ite = split(y)
        ye, ce = y[ite], cats[ite]
        normal_eval = np.where(ye == 0)[0]
        attack_pool = np.where(ye != 0)[0]
        for arm in ARMS:
            sc, sq = z[f'cal_{arm}'][:CAP[name]], z[f'test_{arm}'][ite]
            honest_cut = float(np.quantile(sc, .99, method='higher'))
            r, f = op_point(ye, ce, sq > honest_cut)
            rows.append((name, arm, 'honest', 0.0, 'disjoint calibration batch', r, f, '', ''))
            # leakage: same normals used to set and to score the threshold
            for frac in (0.5, 1.0):
                idx = np.random.RandomState(0).permutation(normal_eval)
                half = idx[:int(len(idx) * frac)]
                cut = float(np.quantile(sq[half], .99, method='higher'))
                rl, fl = op_point(ye, ce, sq > cut)
                rows.append((name, arm, 'leak', frac, 'threshold-on-eval-normals', rl, fl, '', ''))
            # contamination of the calibration batch
            for rate in RATES[1:]:
                recs, fprs, crecs, cfprs = [], [], [], []
                for rep in range(REPS):
                    rng = np.random.RandomState(1000 + rep)
                    k = int(round(rate * len(sc)))
                    chosen = rng.choice(attack_pool, k, replace=False)
                    att = sq[chosen]
                    clean = rng.choice(len(sc), len(sc) - k, replace=False)
                    cut = float(np.quantile(np.concatenate([sc[clean], att]), .99, method='higher'))
                    # mixed-in attacks are removed from both the numerator and the
                    # denominator of recall, so calibration and evaluation never
                    # share an attack sample
                    keep = np.ones(len(ye), bool)
                    keep[chosen] = False
                    r2, f2 = op_point(ye, ce, (sq > cut) & keep, keep)
                    rc, fc = op_point(ye, ce, (sq > honest_cut) & keep, keep)
                    recs.append(r2)
                    fprs.append(f2)
                    crecs.append(rc)
                    cfprs.append(fc)
                note = 'disjoint attack scores mixed in, excluded from numerator and denominator'
                rows.append((name, arm, 'contam_control', rate,
                             'paired clean control, same remaining attacks',
                             float(np.mean(crecs)), float(np.mean(cfprs)),
                             float(np.std(crecs)), float(np.std(cfprs))))
                rows.append((name, arm, 'contamination', rate, note,
                             float(np.mean(recs)), float(np.mean(fprs)),
                             float(np.std(recs)), float(np.std(fprs))))
                print(f'{name:>4s} {arm:7s} contam={rate:.0%} '
                      f'clean={np.mean(crecs):.4f} contam={np.mean(recs):.4f} '
                      f'loss={100 * (np.mean(crecs) - np.mean(recs)):.1f}pp '
                      f'fpr={np.mean(fprs):.4f}', flush=True)

    out = RES / 'calib_stress.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'arm', 'scenario', 'rate', 'note',
                    'recall', 'fpr', 'recall_sd', 'fpr_sd'])
        w.writerows(rows)
    print(f'wrote {out}')
