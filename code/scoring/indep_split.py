"""Independent-split re-evaluation of the three arms from cached full-test scores.

Every score array in results/scores_<name>_s42.npz covers the whole test split,
so a new stratified half split only changes which samples form the evaluation
half. No checkpoint, no reference set and no calibration set is touched.

Seed 42 reproduces the paper's split; 7 / 2025 / 1234 are disjoint re-splits
used to check that the k and variant screening did not ride on the evaluation
half.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
from baselines import macro_metrics  # noqa: E402

RES = TRAE / 'results'
ARMS = ('native', 'raw1nn', 'lat1nn')
NORMAL = {'normal', 'Normal', 'BENIGN'}
SPLITS = (42, 7, 2025, 1234)


def split(y, seed):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def evaluate(s, cal, y, cats):
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    rec = [float(pred[cats == c].mean()) for c in sorted(set(cats.tolist()) - NORMAL)]
    return agg['macro_tpr_1'], agg['macro_auroc'], float(np.mean(rec)), float(pred[y == 0].mean())


if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ton', 'unsw', 'cic']
    rows = []
    for name in names:
        z = np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False)
        y, cats = z['y_test'], z['cats_test'].astype(str)
        for seed in SPLITS:
            _, ite = split(y, seed)
            for arm in ARMS:
                t1, au, r99, fpr = evaluate(z[f'test_{arm}'][ite], z[f'cal_{arm}'],
                                            y[ite], cats[ite])
                rows.append({'dataset': name, 'split_seed': seed, 'arm': arm,
                             'macro_tpr1': round(t1, 6), 'macro_auroc': round(au, 6),
                             'recall99': round(r99, 6), 'fpr': round(fpr, 6)})
                print(f'{name:>4s} split{seed} {arm:7s} tpr1={t1:.4f} auroc={au:.4f} '
                      f'recall99={r99:.4f} fpr={fpr:.4f}', flush=True)

    out = RES / 'indep_splits.csv'
    keys = list(rows[0])
    with out.open('w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in keys) + '\n')

    print('\n== macro TPR@1% by split ==')
    for name in names:
        d = {a: [r for r in rows if r['dataset'] == name and r['arm'] == a] for a in ARMS}
        print(f'[{name}]')
        for a in ARMS:
            v = np.array([r['macro_tpr1'] for r in d[a]])
            print(f'  {a:7s} mean={v.mean():.4f} sd={v.std(ddof=1):.4f} '
                  f'vals={", ".join(f"{x:.4f}" for x in v)}')
        delta = np.array([r['macro_tpr1'] for r in d['lat1nn']]) - \
                np.array([r['macro_tpr1'] for r in d['native']])
        print(f'  lat-native delta pp: {", ".join(f"{x * 100:+.1f}" for x in delta)}')
    print(f'wrote {out}')
