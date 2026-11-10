"""UNSW-NB15 paired scoring on the 2025 checkpoints with the CORRECT preprocessing.

The frozen DRL2025 checkpoints were trained on MinMax-scaled inputs, while the
earlier paired script fed them the StandardScaler arrays from
DRL_UNSWNB15_20261106/prepared/data.npz. This script rebuilds the MinMax inputs
from the raw CSVs (MinMaxScaler fit on the same 50,400 normal fit rows), verifies
the reconstruction against each run's stored development calibration scores,
then recomputes residual / raw-1NN / latent-1NN and the recalibration check.
"""
import os
import sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat'))
from baselines import macro_metrics  # noqa: E402
from DRL_Model.Model import DRL  # noqa: F401,E402

UNSW = BASE / 'artifacts' / 'drl_unsw'
RUNS = BASE / 'artifacts' / 'drl_baseline' / 'runs' / 'DRL2025'
RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}
TRAIN_CSV = STUDY / 'source' / 'datasets' / 'UNSW-NB15' / 'UNSW_NB15_training-set.csv'
TEST_CSV = STUDY / 'source' / 'datasets' / 'UNSW-NB15' / 'UNSW_NB15_testing-set.csv'


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


def knn_scores(ref, points):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(points)[0][:, 0], 0))


def build():
    with np.load(UNSW / 'prepared' / 'data.npz', allow_pickle=False) as z:
        tr_idx, va_idx = z['train_row_indices'], z['validation_row_indices']
    pre = joblib.load(UNSW / 'prepared' / 'preprocessor.joblib')
    num, cat = pre['numeric'], pre['categorical']
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    mm = MinMaxScaler().fit(tr.iloc[tr_idx][num].to_numpy(dtype=np.float64))
    enc = pre['encoder']

    def tx(df):
        n = mm.transform(df[num].to_numpy(dtype=np.float64)).astype(np.float32)
        o = enc.transform(df[cat])
        o = o.toarray() if hasattr(o, 'toarray') else np.asarray(o, dtype=np.float32)
        return np.concatenate([n, o.astype(np.float32)], axis=1).astype(np.float32)

    x_tr = tx(tr.iloc[tr_idx])
    x_cal = tx(tr.iloc[va_idx])
    x_te = tx(te)
    y = te['label'].to_numpy(dtype=np.int64)
    cats = np.where(y == 0, 'Normal', te['attack_cat'].to_numpy(dtype=str))
    return x_tr, x_cal, x_te, y, cats


def main():
    x_tr, x_cal, x_te, y, cats = build()
    # Independent calibration set: normal rows from the training-file validation
    # split (never from the official test). The frozen run used its own 2,800-row
    # calibration subset; row identity is not recoverable, so we fix our own.
    x_cal = x_cal[:2800]
    _, ite = split(y)
    raw_c, raw_q = knn_scores(x_tr, x_cal), knn_scores(x_tr, x_te)
    rows = []
    for seed in (42, 43, 44):
        folder = RUNS / f'final_minmax_seed{seed}'
        model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        model.eval()
        with torch.no_grad():
            rc = model(torch.from_numpy(x_cal).float()).numpy().ravel()
            rq = model(torch.from_numpy(x_te).float()).numpy().ravel()
            ht = model.encoder(torch.from_numpy(x_tr).float()).numpy()
            hc = model.encoder(torch.from_numpy(x_cal).float()).numpy()
            hq = model.encoder(torch.from_numpy(x_te).float()).numpy()
        with np.load(folder / 'development_scores.npz', allow_pickle=False) as z:
            ref = np.sort(z['calibration'])
        probe = np.searchsorted(np.sort(rc), ref)
        match = float(np.mean(np.abs(np.sort(rc)[np.clip(probe, 0, len(rc) - 1)] - ref) < 1e-6))
        lat_c, lat_q = knn_scores(ht, hc), knn_scores(ht, hq)
        for arm, (sc, sq) in {'native': (rc, rq), 'raw1nn': (raw_c, raw_q),
                              'lat1nn': (lat_c, lat_q)}.items():
            agg, _ = macro_metrics(y[ite], cats[ite], sq[ite])
            cut = float(np.quantile(sc, .99, method='higher'))
            pred = sq[ite] > cut
            rec99 = float(np.mean([pred[cats[ite] == k].mean()
                                   for k in sorted(set(cats[ite].tolist()) - NORMAL)]))
            fpr = float(pred[y[ite] == 0].mean())
            rows.append((seed, arm, agg['macro_tpr_1'], agg['macro_auroc'], rec99, fpr))
            print(f'unsw2025fix s{seed} {arm:7s} tpr1={agg["macro_tpr_1"]:.4f} '
                  f'auroc={agg["macro_auroc"]:.4f} recall99={rec99:.4f} fpr={fpr:.4f} '
                  f'calib_match={match:.3f}', flush=True)
    out = RES / 'paired_2025_unsw_minmax.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('seed,arm,macro_tpr1,macro_auroc,recall99,fpr\n')
        for r in rows:
            f.write(','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in r) + '\n')
    a = np.array([r[2] for r in rows if r[1] == 'lat1nn'])
    b = np.array([r[2] for r in rows if r[1] == 'native'])
    c = np.array([r[2] for r in rows if r[1] == 'raw1nn'])
    print(f'lat1nn mean={a.mean():.4f} native mean={b.mean():.4f} raw1nn mean={c.mean():.4f}')
    print(f'delta native={(a - b) * 100}')

    # recalibration on the corrected latent arm (seed 42 model)
    with torch.no_grad():
        model = torch.load(RUNS / 'final_minmax_seed42' / 'model.pt', map_location='cpu', weights_only=False)
        model.eval()
        ht = model.encoder(torch.from_numpy(x_tr).float()).numpy()
        hc = model.encoder(torch.from_numpy(x_cal).float()).numpy()
        hq = model.encoder(torch.from_numpy(x_te).float()).numpy()
    sc, sq = knn_scores(ht, hc), knn_scores(ht, hq)
    n_idx = np.where(y[ite] == 0)[0]
    atk = np.where(y[ite] != 0)[0]
    cls = sorted(set(cats[ite][atk].tolist()) - NORMAL)
    cut0 = float(np.quantile(sc, .99, method='higher'))
    fprs, recs = [], []
    for rep in range(20):
        perm = np.random.RandomState(rep).permutation(n_idx)
        a1, a2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
        cut = float(np.quantile(sq[ite][a1], .99, method='higher'))
        fprs.append(float((sq[ite][a2] > cut).mean()))
        recs.append(float(np.mean([(sq[ite][atk][cats[ite][atk] == k] > cut).mean() for k in cls])))
    ks = float(ks_2samp(sc, sq[ite][n_idx]).statistic)
    med_ratio = float(np.median(sq[ite][n_idx]) / np.median(sc))
    p99_ratio = float(np.quantile(sq[ite][n_idx], .99) / np.quantile(sc, .99))
    print(f'recal: KS={ks:.4f} med_ratio={med_ratio:.3f} p99_ratio={p99_ratio:.3f} '
          f'source_fpr={float((sq[ite][n_idx] > cut0).mean()):.4f} '
          f'target_fpr={np.mean(fprs):.4f}+-{np.std(fprs):.4f} recall={np.mean(recs):.4f}')
    with (RES / 'recal_2025_unsw_minmax.txt').open('w', encoding='utf-8') as f:
        f.write(f'KS {ks:.6f}\nsource_fpr {float((sq[ite][n_idx] > cut0).mean()):.6f}\n'
                f'target_fpr {np.mean(fprs):.6f}\ntarget_fpr_std {np.std(fprs):.6f}\n'
                f'target_recall {np.mean(recs):.6f}\n'
                f'median_ratio {med_ratio:.6f}\np99_ratio {p99_ratio:.6f}\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
