"""UNSW-NB15 paired scoring on the ICLR 2025 conference checkpoints.

The main paired table used the GitHub PrototypeNet for UNSW. The conference
version (DRL_Model.Model.DRL, softmax weights) already has checkpoints for
seeds 42/43/44 trained on exactly the same prepared arrays, so the comparison
can be redone on the 2025 implementation without retraining.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat'))
from baselines import macro_metrics  # noqa: E402
from DRL_Model.Model import DRL  # noqa: E402

UNSW = BASE / 'artifacts' / 'drl_unsw'
RUNS = BASE / 'artifacts' / 'drl_baseline' / 'runs' / 'DRL2025'
RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}


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


def evaluate(s, cal, y, cats):
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    rec = [float(pred[cats == c].mean()) for c in sorted(set(cats.tolist()) - NORMAL)]
    return agg['macro_tpr_1'], agg['macro_auroc'], float(np.mean(rec)), float(pred[y == 0].mean())


def main():
    with np.load(UNSW / 'prepared' / 'data.npz', allow_pickle=False) as z:
        x, c, d = z['x_train'], z['x_validation'], z['x_test']
        y, cats = z['y_test'].astype(np.int64), z['attack_cat'].astype(str)
    cats = np.where(y == 0, 'Normal', cats)
    _, ite = split(y)
    raw_c, raw_q = knn_scores(x, c), knn_scores(x, d)
    rows = []
    for seed in (42, 43, 44):
        model = torch.load(RUNS / f'final_minmax_seed{seed}' / 'model.pt',
                           map_location='cpu', weights_only=False)
        model.eval()
        with torch.no_grad():
            ht = model.encoder(torch.from_numpy(x).float()).numpy()
            hc = model.encoder(torch.from_numpy(c).float()).numpy()
            hq = model.encoder(torch.from_numpy(d).float()).numpy()
            rc = model(torch.from_numpy(c).float()).numpy().ravel()
            rq = model(torch.from_numpy(d).float()).numpy().ravel()
        lat_c, lat_q = knn_scores(ht, hc), knn_scores(ht, hq)
        for arm, (sc, sq) in {'native': (rc, rq), 'raw1nn': (raw_c, raw_q),
                              'lat1nn': (lat_c, lat_q)}.items():
            t1, au, r99, fpr = evaluate(sq[ite], sc, y[ite], cats[ite])
            rows.append((seed, arm, t1, au, r99, fpr))
            print(f'unsw2025 s{seed} {arm:7s} tpr1={t1:.4f} auroc={au:.4f} '
                  f'recall99={r99:.4f} fpr={fpr:.4f}', flush=True)
    out = RES / 'paired_2025_unsw.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('seed,arm,macro_tpr1,macro_auroc,recall99,fpr\n')
        for r in rows:
            f.write(','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in r) + '\n')
    a = np.array([r[2] for r in rows if r[1] == 'lat1nn'])
    b = np.array([r[2] for r in rows if r[1] == 'native'])
    print(f'lat1nn mean={a.mean():.4f} native mean={b.mean():.4f} '
          f'delta range={(a - b).min() * 100:.1f}..{(a - b).max() * 100:.1f}pp')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
