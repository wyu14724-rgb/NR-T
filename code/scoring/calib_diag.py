"""Why is the calibrated FPR 5% on UNSW but ~1% on ToN/CIC?

Compares the calibration-normal and test-normal score distributions for the
latent 1NN readout (and, as a control, the raw-feature 1NN readout). Reports
quantiles, KS statistics, the realized FPR at the calibration 99th percentile,
and the macro recall at the *oracle* test-normal 1% threshold, so that the ROC
claim and the calibrated operating point can be compared directly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402
from baselines import macro_metrics  # noqa: E402

CACHE = TRAE / 'cache'
RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}


def load(name):
    if name == 'cic':
        x, c, d, y, classes = cic.load()
        return x, c, d, y.astype(np.int64), np.asarray(classes, dtype=str)[y]
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


def scores(ref, cal, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return (np.sqrt(np.maximum(nn.kneighbors(cal)[0][:, 0], 0)),
            np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0)))


def qs(v, ps=(50, 90, 99, 99.9)):
    return {f'p{p}': float(np.percentile(v, p)) for p in ps}


def report(name, arm, sc, sq, yte, cte, out):
    cal_n = sc
    te_n = sq[yte == 0]
    cut = float(np.quantile(cal_n, .99, method='higher'))
    fpr = float((te_n > cut).mean())
    oracle = float(np.quantile(te_n, .99, method='higher'))
    pred = sq[yte != 0] > oracle
    attack_means = [float(pred[cte[yte != 0] == c].mean())
                    for c in sorted(set(cte[yte != 0].tolist()) - NORMAL)]
    ks = ks_2samp(cal_n, te_n)
    d = {'n_cal': int(len(cal_n)), 'n_test_normal': int(len(te_n)),
         'cal_q': qs(cal_n), 'test_normal_q': qs(te_n),
         'median_ratio': float(np.median(te_n) / max(np.median(cal_n), 1e-12)),
         'ks': float(ks.statistic), 'ks_p': float(ks.pvalue),
         'cal99_cut': cut, 'fpr_at_cal99': fpr,
         'oracle99_cut': oracle, 'oracle_fpr': float((te_n > oracle).mean()),
         'recall_at_oracle99': float(np.mean(attack_means))}
    out.setdefault(name, {})[arm] = d
    print(f"[{name}/{arm}] med_ratio={d['median_ratio']:.2f} ks={d['ks']:.3f} "
          f"fpr@cal99={fpr:.3%} oracle_recall1%={d['recall_at_oracle99']:.4f} "
          f"cal_p99={cut:.3f} test_p99={oracle:.3f}", flush=True)


def main():
    out = {}
    for name in ('ton', 'unsw', 'cic'):
        x, c, d, y, cats = load(name)
        _, ite = split(y)
        with np.load(CACHE / f'{name}_s42.npz') as z:
            ht, hc, hq = z['ht'], z['hc'], z['hq']
        sc_l, sq_l = scores(ht, hc, hq)
        sc_r, sq_r = scores(x, c, d)
        report(name, 'lat1nn', sc_l, sq_l[ite], y[ite], cats[ite], out)
        report(name, 'raw1nn', sc_r, sq_r[ite], y[ite], cats[ite], out)
        # tier-1 check: are test normals simply farther from the reference?
        print(f'[{name}] train normals={len(x)}, cal={len(c)}, test={len(d)}', flush=True)

    p = RES / 'calib_diag.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'wrote {p}', flush=True)


if __name__ == '__main__':
    main()
