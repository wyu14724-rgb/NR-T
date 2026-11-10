"""CIC-IDS2017 cross-day experiment.

Days are recovered from the timestamps (5 capture days; gaps > 8h separate
them). The checkpoint is retrained on normals from the first three days only,
the reference set and the calibration batch are also restricted to those days,
and everything is evaluated on the last two days, which the model never saw.

This is the protocol the paper lists as future work ("split by capture day").
Output: results/cic_day_split.csv
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
from paired_2025_cic import train  # noqa: E402
from baselines import macro_metrics  # noqa: E402

RES = TRAE / 'results'
DATA = TRAE / 'cic' / 'data' / 'cicids2017.npz'
NORMAL = {'normal', 'Normal', 'BENIGN'}
NS = 1_000_000_000
SEED = 42


def day_ids(*ts_arrays):
    u = np.unique(np.concatenate(ts_arrays))
    cuts = (u[:-1][np.diff(u) > 8 * 3600 * NS] + u[1:][np.diff(u) > 8 * 3600 * NS]) // 2
    return [np.searchsorted(cuts, t) for t in ts_arrays], len(cuts) + 1


def knn(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0))


def evaluate(s, cal, y, cats):
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    rec = [float(pred[cats == c].mean()) for c in sorted(set(cats.tolist()) - NORMAL)]
    return agg['macro_tpr_1'], agg['macro_auroc'], float(np.mean(rec)), float(pred[y == 0].mean())


if __name__ == '__main__':
    with np.load(DATA, allow_pickle=True) as z:
        x_tr, x_cal, x_te = z['x_train'], z['x_cal'], z['x_test']
        y, classes = z['y_test'].astype(np.int64), z['classes'].astype(str)
        ts_tr, ts_cal, ts_te = z['ts_train'], z['ts_cal'], z['ts_test']
    cats = classes[y]
    (d_tr, d_cal, d_te), nday = day_ids(ts_tr, ts_cal, ts_te)
    print('days:', nday, 'train', np.bincount(d_tr, minlength=nday),
          'cal', np.bincount(d_cal, minlength=nday),
          'test', np.bincount(d_te, minlength=nday), flush=True)

    early = nday - 2                      # first nday-2 days for fitting
    fit, fit_cal = d_tr < early, d_cal < early
    held = d_te >= early                  # last two days, unseen
    print(f'fit normals={fit.sum()}  cal normals={fit_cal.sum()}  '
          f'held-out rows={held.sum()} (normals={(y[held] == 0).sum()})', flush=True)

    x_ref, x_caln = x_tr[fit], x_cal[fit_cal]
    xe, ye, ce = x_te[held], y[held], cats[held]
    model = train(x_ref, SEED)
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x_ref).float()).numpy()
        hc = model.encoder(torch.from_numpy(x_caln).float()).numpy()
        he = model.encoder(torch.from_numpy(xe).float()).numpy()
        rc = model(torch.from_numpy(x_caln).float()).numpy().ravel()
        re = model(torch.from_numpy(xe).float()).numpy().ravel()

    arms = {'native': (rc, re), 'raw1nn': (knn(x_ref, x_caln), knn(x_ref, xe)),
            'lat1nn': (knn(ht, hc), knn(ht, he))}
    n_held = np.where(ye == 0)[0]
    atk_mask = ye != 0
    out = RES / 'cic_day_split.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('arm,macro_tpr1,macro_auroc,recall99,fpr,day_recal_recall,day_recal_fpr\n')
        for arm, (sc, sq) in arms.items():
            t1, au, r99, fpr = evaluate(sq, sc, ye, ce)
            # target-day recalibration: 20 disjoint half splits of held-out normals
            recs, fprs = [], []
            for rep in range(20):
                rng = np.random.RandomState(500 + rep)
                perm = rng.permutation(n_held)
                h = len(perm) // 2
                cut = float(np.quantile(sq[perm[:h]], .99, method='higher'))
                rec = [float((sq[atk_mask & (ce == c)] > cut).mean())
                       for c in sorted(set(ce.tolist()) - NORMAL)]
                recs.append(np.mean(rec))
                fprs.append(float((sq[perm[h:]] > cut).mean()))
            f.write(f'{arm},{t1:.6f},{au:.6f},{r99:.6f},{fpr:.6f},'
                    f'{np.mean(recs):.6f},{np.mean(fprs):.6f}\n')
            print(f'{arm:7s} tpr1={t1:.4f} auroc={au:.4f} recall99={r99:.4f} '
                  f'fpr={fpr:.4f} | day-recal recall={np.mean(recs):.4f} '
                  f'fpr={np.mean(fprs):.4f}', flush=True)
    print(f'wrote {out}')
