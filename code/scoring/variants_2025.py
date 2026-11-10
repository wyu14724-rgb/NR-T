"""Re-run the readout-variant screen on the ICLR 2025 checkpoints only.

All three datasets use the 2025 implementation (softmax weights). ToN reuses the
cached 2025 latents; UNSW loads its frozen 2025 checkpoints; CIC is retrained
here (seed 42) because no 2025 model was saved before.

Evaluation is on the same half split as the paired table (§5.3), seed 42, so the
deltas are directly comparable to that table's seed-42 column.
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(BASE / '实验产物' / 'DRL_基线对照_20261106' / 'source_snapshot' / 'DRL2025_compat'))
import paired_2025_cic as c25  # noqa: E402
import run_cicids as cic  # noqa: E402
from baselines import get_data, macro_metrics  # noqa: E402
from DRL_Model.Model import DRL  # noqa: E402

RES = TRAE / 'results'
CACHE = TRAE / 'cache'
RUNS = BASE / '实验产物' / 'DRL_基线对照_20261106' / 'runs' / 'DRL2025'
NORMAL = {'normal', 'Normal', 'BENIGN'}
SEED = 42
K_MODES, VAR_KEEP, MIX = 20, .95, .1


def split(y):
    return c25.split(y)


def knn(ref, q, k=1, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=metric, n_jobs=-1).fit(ref)
    return nn


def dk(ref, q, k=1):
    nn = knn(ref, q, k)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0], 0))


def build_modes(ht, hc):
    km = KMeans(n_clusters=K_MODES, n_init=10, random_state=42).fit(hc)
    cent = km.cluster_centers_
    ncc = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(cent)
    a_t = ncc.kneighbors(ht)[1][:, 0]
    subs = []
    for c in range(K_MODES):
        m = ht[a_t == c]
        mu = m.mean(0) if len(m) else cent[c]
        if len(m) > 5:
            xc = m - mu
            _, sv, vt = np.linalg.svd(xc, full_matrices=False)
            ev = sv ** 2
            cum = np.cumsum(ev) / max(ev.sum(), 1e-12)
            r = int(np.searchsorted(cum, VAR_KEEP) + 1)
            r = max(1, min(r, len(vt) - 1))
            P = vt[:r].T @ vt[:r]
        else:
            P = np.zeros((ht.shape[1], ht.shape[1]))
        subs.append({'mu': mu.astype(np.float32), 'P': P.astype(np.float32), 'n': int(len(m))})
    return subs, cent


def mor_scores(hq, ht, hc):
    nn = NearestNeighbors(n_neighbors=KMAX_D, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(ht)
    dd, ii = nn.kneighbors(hq)
    subs, cent = build_modes(ht, hc)
    ncc = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(cent)
    a_t = ncc.kneighbors(ht)[1][:, 0]
    mode = a_t[ii[:, 0]]
    s1 = np.empty(len(hq)); s2 = np.empty(len(hq))
    for c in range(len(subs)):
        idx = np.where(mode == c)[0]
        if len(idx) == 0:
            continue
        mu, P = subs[c]['mu'], subs[c]['P']
        z = hq[idx] - mu
        r2 = ((z - z @ P) ** 2).sum(1)
        s1[idx] = r2
        s2[idx] = r2 + MIX * (z ** 2).sum(1)
    return np.sqrt(dd[:, 0]), s1, s2


KMAX_D = 10


def variants(hq, ht, hc):
    out = {}
    out['d1'] = dk(ht, hq, 1)[:, 0]
    d5 = dk(ht, hq, 5)
    out['d5'] = d5.mean(1)
    nn = knn(ht, hq, 1, metric='cosine')
    out['cos'] = nn.kneighbors(hq)[0][:, 0]
    pca = PCA(n_components=min(128, ht.shape[1]), whiten=True, random_state=0).fit(ht)
    out['whiten'] = dk(pca.transform(ht), pca.transform(hq), 1)[:, 0]
    dt = dk(ht, ht, 6)
    sigma = dt[:, 5]
    nn1 = knn(ht, hq, 1)
    d1v, i1 = nn1.kneighbors(hq)
    d1v = np.sqrt(np.maximum(d1v[:, 0], 0))
    sig_star = sigma[i1[:, 0]]
    med = float(np.median(sigma))
    out['local'] = d1v / (sig_star + 1e-12)
    out['gated'] = np.where(sig_star > med, out['local'], d1v)
    d1m, mor1, mor2 = mor_scores(hq, ht, hc)
    out['mor'] = mor1
    return out


def main():
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        x, c, d, y, cats = get_data(name)
        y = y.astype(np.int64)
        cats = np.asarray(cats, dtype=str)
        _, ite = split(y)
        yte, cte = y[ite], cats[ite]
        if name == 'ton':
            z = np.load(CACHE / 'ton_s42.npz')
            ht, hc, hq = z['ht'], z['hc'], z['hq']
            rc = float('nan')  # residual recomputed from the paired run, see below
            rq = np.full(len(yte), np.nan)
        else:
            if name == 'unsw':
                m = torch.load(RUNS / f'final_minmax_seed{SEED}' / 'model.pt',
                               map_location='cpu', weights_only=False)
            else:
                m = c25.train(x, SEED)
            m.eval()
            with torch.no_grad():
                ht = m.encoder(torch.from_numpy(x).float()).numpy()
                hc = m.encoder(torch.from_numpy(c).float()).numpy()
                hq = m.encoder(torch.from_numpy(d).float()).numpy()
                rq = m(torch.from_numpy(d).float()).numpy().ravel()
        sc = variants(hq[ite], ht, hc)
        base_t1 = None
        for v, s in sc.items():
            agg, _ = macro_metrics(yte, cte, s)
            rows.append((name, v, agg['macro_tpr_1'], agg['macro_auroc']))
            if v == 'd1':
                base_t1 = agg['macro_tpr_1']
        for r in rows:
            if r[0] == name:
                print(f'{name:4s} {r[1]:7s} tpr1={r[2]:.4f} auroc={r[3]:.4f} '
                      f'delta={100 * (r[2] - base_t1):+.2f}pp', flush=True)
    with (RES / 'variants_2025.csv').open('w', encoding='utf-8') as f:
        f.write('dataset,variant,macro_tpr1,macro_auroc\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f}\n')
    print('wrote', RES / 'variants_2025.csv')


if __name__ == '__main__':
    main()
