"""Can the CIC-IDS2017 full attack-class set be recovered without retraining?

`results/scores_cic_s42.npz` already carries `y_test`,
`cats_test` and per-sample `test_lat1nn` / `cal_lat1nn` for the *whole* test
split (117,771 rows, 10 attack classes + BENIGN), so the class-wise alpha sweep
and recalibration are computed directly from it — no cache rebuild is needed.

This script documents that claim. The latent caches contain no labels or attack
categories at all, so they cannot supply classes on their own; what they can do
is confirm that the archived `lat1nn` scores really are 1-NN distances in the
cached latents. The 2025 checkpoint cache (`cic_s42_2025.npz`) reproduces them
exactly; the older `cic_s42.npz` comes from a different (pre-2025) checkpoint
(Spearman 0.979, not identical) and is therefore not the source of these numbers.
"""
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

TRAE = str(Path(__file__).resolve().parents[2])
RES = TRAE + r'\results'
C_2025 = TRAE + r'\cache\cic_s42_2025.npz'
C_OLD = TRAE + r'\cache\cic_s42.npz'
N_PROBE = 2000
SEED = 0


def knn1(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0)).astype(np.float64)


def main():
    with np.load(RES + r'\scores_cic_s42.npz', allow_pickle=False) as z:
        y, cats = z['y_test'], z['cats_test']
        lat_q = z['test_lat1nn'].astype(np.float64)
    u, n = np.unique(cats, return_counts=True)
    attack = [(a, int(k)) for a, k in zip(u, n) if a != 'BENIGN']
    print(f'npz test rows={len(y)} normals={int((y == 0).sum())} '
          f'attack classes={len(attack)}')
    for a, k in attack:
        print(f'   {a}: {k}')
    assert len(attack) == 10 and len(lat_q) == len(y), 'class coverage incomplete'

    rng = np.random.RandomState(SEED)
    qi = np.sort(rng.choice(len(lat_q), size=N_PROBE, replace=False))
    with np.load(C_2025, allow_pickle=False) as m:
        ht, hq = m['ht'], m['hq']
    assert len(hq) == len(lat_q) and hq.shape[1] == ht.shape[1]
    print(f'2025 cache: keys=ht{ht.shape}/hq{hq.shape}; labels stored: no')
    rec = knn1(ht, hq[qi])
    d_2025 = float(np.max(np.abs(rec - lat_q[qi])))
    print(f'subsample check ({N_PROBE} rows): max|recomputed - archived| = {d_2025:.3e}')
    assert d_2025 < 1e-6, d_2025

    with np.load(C_OLD, allow_pickle=False) as m:
        ht_o, hq_o = m['ht'], m['hq']
    rec_o = knn1(ht_o, hq_o[qi])
    print(f'older cic_s42.npz cache (different checkpoint): Spearman='
          f'{spearmanr(rec_o, lat_q[qi]).statistic:.4f}, '
          f'max|diff|={np.max(np.abs(rec_o - lat_q[qi])):.3e}')
    print('OK: CIC full-class evaluation is possible directly from the npz labels; '
          'no 1-NN rebuild needed (and the caches hold no labels anyway).')


if __name__ == '__main__':
    main()
