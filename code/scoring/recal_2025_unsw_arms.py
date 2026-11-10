"""Recalibration check applied identically to all three scoring arms on UNSW-NB15.

Same corrected MinMax inputs and the same seed-42 checkpoint as
`unsw_fix_2025.py`. For each arm (native residual / raw-feature 1NN / latent 1NN)
we apply the same threshold rule: 99th percentile of a half of the target-domain
normal samples, then measure FPR on the other half and the macro recall at that
threshold. Repeating the half split 20 times gives mean +- sd.

The point is to test whether the 4.78% -> 1.05% improvement is specific to NR-T
or would also be obtained by applying the same rule to the residual score.
"""
import os
import sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unsw_fix_2025 import RUNS, RES, NORMAL, build, split, knn_scores  # noqa: E402


def macro_recall(sc, mask_atk, cls_mask, cut):
    return float(np.mean([(sc[mask_atk & (cls_mask == k)] > cut).mean() for k in np.unique(cls_mask[mask_atk])]))


def main():
    x_tr, x_cal, x_te, y, cats = build()
    x_cal = x_cal[:2800]
    _, ite = split(y)

    raw_c, raw_q = knn_scores(x_tr, x_cal), knn_scores(x_tr, x_te)
    model = torch.load(RUNS / 'final_minmax_seed42' / 'model.pt',
                       map_location='cpu', weights_only=False)
    model.eval()
    with torch.no_grad():
        rc = model(torch.from_numpy(x_cal).float()).numpy().ravel()
        rq = model(torch.from_numpy(x_te).float()).numpy().ravel()
        ht = model.encoder(torch.from_numpy(x_tr).float()).numpy()
        hc = model.encoder(torch.from_numpy(x_cal).float()).numpy()
        hq = model.encoder(torch.from_numpy(x_te).float()).numpy()
    lat_c, lat_q = knn_scores(ht, hc), knn_scores(ht, hq)

    arms = {'residual': (rc, rq), 'raw1nn': (raw_c, raw_q), 'lat1nn': (lat_c, lat_q)}
    y_e, c_e = y[ite], cats[ite]
    n_idx = np.where(y_e == 0)[0]
    atk = y_e != 0
    is_atk = np.zeros(len(y_e), dtype=bool)
    is_atk[np.where(atk)[0]] = True
    cls = c_e

    rows = []
    for name, (sc, sq) in arms.items():
        q = sq[ite]
        cut0 = float(np.quantile(sc, .99, method='higher'))
        source_fpr = float((q[n_idx] > cut0).mean())
        source_rec = macro_recall(q, is_atk, cls, cut0)
        fprs, recs = [], []
        for rep in range(20):
            perm = np.random.RandomState(rep).permutation(n_idx)
            a1, a2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
            cut = float(np.quantile(q[a1], .99, method='higher'))
            fprs.append(float((q[a2] > cut).mean()))
            recs.append(macro_recall(q, is_atk, cls, cut))
        ks = float(ks_2samp(sc, q[n_idx]).statistic)
        rows.append((name, ks, source_fpr, source_rec,
                     float(np.mean(fprs)), float(np.std(fprs)), float(np.mean(recs))))
        print(f'{name:9s} KS={ks:.4f} srcFPR={source_fpr:.4f} srcR={source_rec:.4f} '
              f'tgtFPR={np.mean(fprs):.4f}+-{np.std(fprs):.4f} tgtR={np.mean(recs):.4f}',
              flush=True)

    out = RES / 'recal_2025_unsw_arms.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('arm,ks,source_fpr,source_recall,target_fpr,target_fpr_std,target_recall\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
