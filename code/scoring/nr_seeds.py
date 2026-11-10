"""Fill Table 1: NR readout on the same held-out half, three training seeds.

Seeds 43/44 require retraining (cache holds only s42); latents are written via
mode_readout.get_latents so the cache is reused by later runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(TRAE / 'code' / 'readout'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402
from baselines import macro_metrics  # noqa: E402
import mode_readout as mr  # noqa: E402

SPLIT_SEED = 42


def split(y, cats):
    rng = np.random.RandomState(SPLIT_SEED)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def main():
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        x, c, d, y, cats = mr.get_data(name)
        _, ite = split(y, cats)
        for seed in (42, 43, 44):
            ht, _, hq = mr.get_latents(name, seed)
            nn = NearestNeighbors(n_neighbors=1, algorithm='brute',
                                  metric='sqeuclidean', n_jobs=-1).fit(ht)
            dist, _ = nn.kneighbors(hq[ite])
            s = np.sqrt(dist[:, 0])
            agg, _ = macro_metrics(y[ite], cats[ite], s)
            rows.append({'dataset': name, 'seed': seed,
                         'tpr1': round(agg['macro_tpr_1'], 6),
                         'auroc': round(agg['macro_auroc'], 6),
                         'aupr': round(agg['macro_aupr'], 6),
                         'n_test': int(len(ite))})
            print(f"{name:>4s} seed={seed} tpr1={rows[-1]['tpr1']:.4f} "
                  f"auroc={rows[-1]['auroc']:.4f} n={len(ite)}", flush=True)

    out = TRAE / 'results' / 'nr_seeds.json'
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for name in ('ton', 'unsw', 'cic'):
        v = [r['tpr1'] for r in rows if r['dataset'] == name]
        print(f"[{name}] 3-seed mean={np.mean(v):.4f} std={np.std(v, ddof=1):.4f} "
              f"min={min(v):.4f} max={max(v):.4f}", flush=True)


if __name__ == '__main__':
    main()
