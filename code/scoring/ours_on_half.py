"""Re-evaluate the NR readout on exactly the same held-out test half used by baselines.py."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'training'))
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402
from baselines import macro_metrics  # noqa: E402

SEED = 42


def split(y, cats):
    rng = np.random.RandomState(SEED)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def main():
    out = {}
    for name in ('ton', 'unsw', 'cic'):
        if name == 'cic':
            x, c, d, y, types = cic.load()
            y = y.astype(np.int64)
            cats = np.asarray(types, dtype=str)[y]
        else:
            x, c, d, y, cats = base.load(name)
            cats = np.asarray(cats, dtype=str)
        z = np.load(TRAE / 'cache' / f'{name}_s42.npz')
        ht, hq = z['ht'], z['hq']
        _, ite = split(y, cats)
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                              n_jobs=-1).fit(ht)
        d1, _ = nn.kneighbors(hq[ite])
        s = np.sqrt(d1[:, 0])
        agg, per = macro_metrics(y[ite], cats[ite], s)
        out[name] = {k: round(v, 6) for k, v in agg.items()} | {'n_test': int(len(ite))}
        (TRAE / 'results' / f'perclass_{name}_NR.json').write_text(
            json.dumps(per, indent=2, ensure_ascii=False))
        print(f"{name:>4s} NR tpr1={agg['macro_tpr_1']:.4f} auroc={agg['macro_auroc']:.4f} "
              f"aupr={agg['macro_aupr']:.4f} n={len(ite)}", flush=True)
    (TRAE / 'results' / 'ours_on_half.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
