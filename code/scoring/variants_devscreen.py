"""Re-screen the §5.5 readout variants on the development half.

§5.5 screened the variants on the same half that reports the main result. This
script scores every variant on both halves with one implementation:

    dev  = the other half of the same seed-42 class-stratified split
    eval = the half used in §5.3 / §5.5

Variant definitions are copied from variants_2025.py; only the query subset
changes.  ``residual`` is read from scores_<name>_s42.npz (test_native), which
covers the whole test partition, so no retraining is needed for it.

Output: results/variants_2025_dev.csv
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / '实验产物' / 'DRL_基线对照_20261106' / 'source_snapshot' / 'DRL2025_compat'))
import paired_2025_cic as c25  # noqa: E402
from baselines import get_data  # noqa: E402

RES = TRAE / 'results'
CACHE = TRAE / 'cache'
RUNS = BASE / '实验产物' / 'DRL_基线对照_20261106' / 'runs' / 'DRL2025'
NORMAL = {'normal', 'Normal', 'BENIGN'}
SEED = 42
K_MODES, VAR_KEEP, MIX = 20, .95, .1


def split(y):
    rng = np.random.RandomState(SEED)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def macro_metrics(y, cats, s):
    normal = y == 0
    tpr1, auroc = [], []
    for k in sorted(set(cats.tolist()) - NORMAL):
        m = cats == k
        pair = normal | m
        yb = m[pair].astype(int)
        fpr, tpr, _ = roc_curve(yb, s[pair])
        tpr1.append(float(tpr[fpr <= .01].max()) if (fpr <= .01).any() else 0.0)
        auroc.append(float(roc_auc_score(yb, s[pair])))
    return float(np.mean(tpr1)), float(np.mean(auroc))


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
            z = m - mu
            _, sv, vt = np.linalg.svd(z, full_matrices=False)
            ev = sv ** 2
            cum = np.cumsum(ev) / max(ev.sum(), 1e-12)
            r = int(np.searchsorted(cum, VAR_KEEP) + 1)
            r = max(1, min(r, len(vt) - 1))
            P = vt[:r].T @ vt[:r]
        else:
            P = np.zeros((ht.shape[1], ht.shape[1]))
        subs.append({'mu': mu.astype(np.float32), 'P': P.astype(np.float32)})
    return subs, cent, a_t


def variant_scores(hq, ht, hc):
    out = {}
    nn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(ht)
    dd, ii = nn.kneighbors(hq)
    d = np.sqrt(np.maximum(dd, 0))
    out['d1'] = d[:, 0]
    # mean of the five nearest Euclidean distances (the label used in §5.5);
    # the earlier dk() squared-then-sqrts, storing mean(sqrt(d)) instead.
    out['d5'] = d[:, :5].mean(1)

    cos = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine', n_jobs=-1).fit(ht)
    out['cos'] = cos.kneighbors(hq)[0][:, 0]

    pca = PCA(n_components=min(128, ht.shape[1]), whiten=True, random_state=0).fit(ht)
    hw, qw = pca.transform(ht), pca.transform(hq)
    nw = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(hw)
    out['whiten'] = np.sqrt(np.maximum(nw.kneighbors(qw)[0][:, 0], 0))

    sig = np.sqrt(np.maximum(
        NearestNeighbors(n_neighbors=6, algorithm='brute', metric='sqeuclidean', n_jobs=-1)
        .fit(ht).kneighbors(ht)[0][:, 5], 0))
    sig_star = sig[ii[:, 0]]
    med = float(np.median(sig))
    out['local'] = out['d1'] / (sig_star + 1e-12)
    out['gated'] = np.where(sig_star > med, out['local'], out['d1'])

    subs, cent, a_t = build_modes(ht, hc)
    mode = a_t[ii[:, 0]]
    s1 = np.empty(len(hq))
    for c in range(len(subs)):
        idx = np.where(mode == c)[0]
        if len(idx) == 0:
            continue
        mu, P = subs[c]['mu'], subs[c]['P']
        z = hq[idx] - mu
        s1[idx] = ((z - z @ P) ** 2).sum(1)
    out['mor'] = s1
    return out


def assets(name):
    """Frozen 2025 latent references and calibration normals (verified against
    scores_<name>_s42.npz: max |d1 diff| < 4e-6)."""
    if name == 'ton':
        z = np.load(CACHE / 'ton_s42.npz')
        return z['ht'], z['hc'], z['hq']
    if name == 'unsw':
        import unsw_fix_2025 as F
        z = np.load(CACHE / 'unsw_s42_2025.npz')
        _, x_cal, _, _, _ = F.build()
        x_cal = x_cal[:2800]          # same cap as variants_2025_unsw_fix.py
        m = torch.load(RUNS / f'final_minmax_seed{SEED}' / 'model.pt',
                       map_location='cpu', weights_only=False)
        m.eval()
        with torch.no_grad():
            hc = m.encoder(torch.from_numpy(x_cal).float()).numpy()
        return z['ht'], hc, z['hq']
    if name == 'cic':
        z = np.load(CACHE / 'cic_s42_2025.npz')
        x, c, _, _, _ = get_data('cic')
        m = c25.train(x, SEED)
        with torch.no_grad():
            hc = m.encoder(torch.from_numpy(c).float()).numpy()
        return z['ht'], hc, z['hq']
    raise ValueError(name)


def main():
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ton', 'unsw', 'cic']
    rows = []
    for name in names:
        s = np.load(RES / f'scores_{name}_s42.npz')
        y, cats = s['y_test'], s['cats_test'].astype(str)
        itr, ite = split(y)
        ht, hc, hq = assets(name)
        if hq.shape[0] != y.shape[0]:
            raise RuntimeError(f'{name}: hq {hq.shape[0]} vs labels {y.shape[0]}')
        sc = variant_scores(hq, ht, hc)
        sc['residual'] = s['test_native'].astype(float)
        for half, idx in (('dev', itr), ('eval', ite)):
            base = None
            for v in ('residual', 'd1', 'd5', 'cos', 'whiten', 'local', 'gated', 'mor'):
                t1, au = macro_metrics(y[idx], cats[idx], sc[v][idx])
                if v == 'd1':
                    base = t1
                rows.append((name, half, v, t1, au))
            print(f'{name:4s} {half:4s} ' + ' '.join(
                f'{r[2]}={r[3]:.4f}({100*(r[3]-base):+.1f})'
                for r in rows if r[0] == name and r[1] == half), flush=True)
    out = RES / 'variants_2025_dev.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('dataset,half,variant,macro_tpr1,macro_auroc\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f}\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
