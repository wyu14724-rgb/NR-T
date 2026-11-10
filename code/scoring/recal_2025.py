"""Target-domain recalibration on the ICLR 2025 UNSW latents (3 seeds)."""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat'))
from DRL_Model.Model import DRL  # noqa: E402

UNSW = BASE / 'artifacts' / 'drl_unsw'
RUNS = BASE / 'artifacts' / 'drl_baseline' / 'runs' / 'DRL2025'
NORMAL = {'normal', 'Normal'}


def split(y):
    rng = np.random.RandomState(42)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h]); te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def knn(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0))


def main():
    with np.load(UNSW / 'prepared' / 'data.npz', allow_pickle=False) as z:
        x, c, d = z['x_train'], z['x_validation'], z['x_test']
        y, cats = z['y_test'].astype(np.int64), z['attack_cat'].astype(str)
    cats = np.where(y == 0, 'Normal', cats)
    _, ite = split(y)
    yte, cte = y[ite], cats[ite]
    n_idx, atk = np.where(yte == 0)[0], np.where(yte != 0)[0]
    for seed in (42, 43, 44):
        model = torch.load(RUNS / f'final_minmax_seed{seed}' / 'model.pt', map_location='cpu', weights_only=False)
        model.eval()
        with torch.no_grad():
            ht = model.encoder(torch.from_numpy(x).float()).numpy()
            hc = model.encoder(torch.from_numpy(c).float()).numpy()
            hq = model.encoder(torch.from_numpy(d).float()).numpy()[ite]
        s, sc = knn(ht, hq), knn(ht, hc)
        cut_val = float(np.quantile(sc, .99, method='higher'))
        fpr_val = float((s[n_idx] > cut_val).mean())
        rec_val = float(np.mean([(s[atk][cte[atk] == k] > cut_val).mean()
                                 for k in sorted(set(cte[atk].tolist()) - NORMAL)]))
        fprs, recs = [], []
        for rep in range(20):
            perm = np.random.RandomState(rep).permutation(n_idx)
            a, b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            cut = float(np.quantile(s[a], .99, method='higher'))
            fprs.append(float((s[b] > cut).mean()))
            recs.append(float(np.mean([(s[atk][cte[atk] == k] > cut).mean()
                                       for k in sorted(set(cte[atk].tolist()) - NORMAL)])))
        print(f'unsw2025 s{seed}: val-cal FPR={fpr_val:.2%} recall={rec_val:.4f} | '
              f'target-cal FPR={np.mean(fprs):.2%}+-{np.std(fprs):.2%} recall={np.mean(recs):.4f}', flush=True)


if __name__ == '__main__':
    main()
