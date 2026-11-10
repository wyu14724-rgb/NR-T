"""Dump per-sample scores for every test row of each dataset (seed 42 checkpoint).

Saved arrays per dataset (results/scores_<name>_s42.npz):
  y_test, cats_test          labels of the full test split
  ts_test                    epoch seconds (cic only)
  test_native/raw1nn/lat1nn  scores for the full test split
  cal_native/raw1nn/lat1nn   scores for the calibration normals

The three arms and their reference sets match paired_three_arms.py /
paired_2025.py exactly, so any re-split of y_test can be evaluated afterwards
without touching a checkpoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
for p in [TRAE / 'code' / 'training',
          TRAE / 'code' / 'prepare',
          TRAE / 'code' / 'scoring',
          BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat']:
    sys.path.insert(0, str(p))

import tail_objective as base  # noqa: E402
import run_cicids as cic  # noqa: E402

RES = TRAE / 'results'
UNSW = BASE / 'artifacts' / 'drl_unsw'
RUNS = BASE / 'artifacts' / 'drl_baseline' / 'runs' / 'DRL2025'
SEED = 42


def knn(ref, q, chunk=8192):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        d, _ = nn.kneighbors(q[i:i + chunk])
        out[i:i + chunk] = np.sqrt(np.maximum(d[:, 0], 0))
    return out


def ton():
    x, c, d, y, cats = base.load('ton')
    model, _ = base.train_model('ton', SEED, 0, 0, verbose=False)
    out = dict(test_native=base.native(model, d, 'ton'),
               cal_native=base.native(model, c, 'ton'))
    out['test_raw1nn'], out['cal_raw1nn'] = knn(x, d), knn(x, c)
    ht = base.encode(model, x)
    out['test_lat1nn'], out['cal_lat1nn'] = knn(ht, base.encode(model, d)), knn(ht, base.encode(model, c))
    return out, y, cats, None


def unsw():
    from unsw_fix_2025 import build
    x, c, d, y, cats = build()
    model = torch.load(RUNS / f'final_minmax_seed{SEED}' / 'model.pt',
                       map_location='cpu', weights_only=False)
    model.eval()
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x).float()).numpy()
        hc = model.encoder(torch.from_numpy(c).float()).numpy()
        hq = model.encoder(torch.from_numpy(d).float()).numpy()
        rc = model(torch.from_numpy(c).float()).numpy().ravel()
        rq = model(torch.from_numpy(d).float()).numpy().ravel()
    out = dict(test_native=rq, cal_native=rc,
               test_raw1nn=knn(x, d), cal_raw1nn=knn(x, c),
               test_lat1nn=knn(ht, hq), cal_lat1nn=knn(ht, hc))
    return out, y, cats, None


def cic_():
    from paired_2025_cic import train as cic_train
    x, c, d, y, classes = cic.load()
    model = cic_train(x, SEED)
    cats = np.asarray(classes, dtype=str)[y]
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x).float()).numpy()
        hc = model.encoder(torch.from_numpy(c).float()).numpy()
        hq = model.encoder(torch.from_numpy(d).float()).numpy()
        rc = model(torch.from_numpy(c).float()).numpy().ravel()
        rq = model(torch.from_numpy(d).float()).numpy().ravel()
    scores = dict(test_native=rq, cal_native=rc,
                  test_raw1nn=knn(x, d), cal_raw1nn=knn(x, c),
                  test_lat1nn=knn(ht, hq), cal_lat1nn=knn(ht, hc))
    ts = np.load(TRAE / 'cic' / 'data' / 'cicids2017.npz')['ts_test']
    return scores, y.astype(np.int64), cats, ts


LOADERS = {'ton': ton, 'unsw': unsw, 'cic': cic_}

if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ton', 'unsw', 'cic']
    for name in names:
        print(f'== {name} ==', flush=True)
        scores, y, cats, ts = LOADERS[name]()
        payload = {k: np.asarray(v, dtype=np.float32) for k, v in scores.items()}
        payload.update(y_test=y.astype(np.int64), cats_test=cats.astype(str))
        if ts is not None:
            payload['ts_test'] = ts.astype(np.int64)
        out = RES / f'scores_{name}_s42.npz'
        np.savez_compressed(out, **payload)
        print(f'{name}: test={len(y)} normals={(y == 0).sum()} -> {out}', flush=True)
