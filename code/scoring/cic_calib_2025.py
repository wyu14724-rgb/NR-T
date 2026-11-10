"""CIC-IDS2017 calibration check on the ICLR 2025 implementation (seed 42).

The other two datasets already had 2025 latents; CIC did not, so the model is
retrained here once and evaluated exactly like recal_2025.py: source-domain
99% quantile -> test FPR, plus a 20-rep half-split target-domain recalibration.
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
import paired_2025_cic as c25  # noqa: E402
import run_cicids as cic  # noqa: E402

NORMAL = {'BENIGN', 'benign', 'Normal', 'normal'}
SEED = 42


def knn(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0))


def main():
    x, c, d, y, classes = cic.load()
    cats = np.asarray(classes, dtype=str)[y] if y.dtype.kind in 'iu' else np.asarray(classes, dtype=str)
    y = y.astype(np.int64)
    _, ite = c25.split(y)
    yte, cte = y[ite], cats[ite]
    model = c25.train(x, SEED)
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x).float()).numpy()
        hc = model.encoder(torch.from_numpy(c).float()).numpy()
        hq = model.encoder(torch.from_numpy(d).float()).numpy()[ite]
    s, sc = knn(ht, hq), knn(ht, hc)
    n_idx = np.where(yte == 0)[0]
    atk = np.where(yte != 0)[0]
    cls = sorted(set(cte[atk].tolist()) - NORMAL)
    cut_val = float(np.quantile(sc, .99, method='higher'))
    fpr_val = float((s[n_idx] > cut_val).mean())
    rec_val = float(np.mean([(s[atk][cte[atk] == k] > cut_val).mean() for k in cls]))
    ks = float(ks_2samp(sc, s[n_idx]).statistic)
    ratio = float(np.median(s[n_idx]) / max(np.median(sc), 1e-12))
    p99c, p99t = float(np.quantile(sc, .99)), float(np.quantile(s[n_idx], .99))
    fprs, recs = [], []
    for rep in range(20):
        perm = np.random.RandomState(rep).permutation(n_idx)
        a, b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        cut = float(np.quantile(s[a], .99, method='higher'))
        fprs.append(float((s[b] > cut).mean()))
        recs.append(float(np.mean([(s[atk][cte[atk] == k] > cut).mean() for k in cls])))
    print(f'cic2025 s{SEED}: KS={ks:.4f} median_ratio={ratio:.3f} '
          f'cal_p99={p99c:.4f} test_p99={p99t:.4f}', flush=True)
    print(f'  source-cal  FPR={fpr_val:.2%} recall={rec_val:.4f}', flush=True)
    print(f'  target-cal  FPR={np.mean(fprs):.2%}+-{np.std(fprs):.2%} recall={np.mean(recs):.4f}', flush=True)
    out = TRAE / 'results' / 'calib_cic_2025.txt'
    out.write_text(f'cic2025 seed{SEED}\nKS {ks:.6f}\nmedian_ratio {ratio:.6f}\n'
                   f'cal_p99 {p99c:.6f}\ntest_p99 {p99t:.6f}\n'
                   f'source_fpr {fpr_val:.6f}\nsource_recall {rec_val:.6f}\n'
                   f'target_fpr {np.mean(fprs):.6f}\ntarget_fpr_std {np.std(fprs):.6f}\n'
                   f'target_recall {np.mean(recs):.6f}\n', encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
