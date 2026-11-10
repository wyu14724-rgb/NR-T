"""Deep SVDD baseline without bias terms, following Ruff et al. (2018).

The earlier baseline used an MLP with biases, which the original paper advises
against (bias terms make the hypersphere collapse trivial). This script re-runs
the baseline with bias=False on every layer, keeping the rest of the protocol
identical to `baselines_dev_sel.py`: configuration selected on the development
half, evaluated on the shared evaluation half.

UNSW-NB15 uses the MinMax reconstruction (same as `unsw_fix_2025.py`).

Usage: python baselines_dsvdd_nobias.py --dataset ton|unsw|cic [--seeds 42,43,44]
Outputs: results/dsvdd_nobias_<name>.json / .csv
         (multi-seed: results/dsvdd_nobias_<name>_seeds.csv)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import get_data, macro_metrics  # noqa: E402

SEED = 42  # split seed, identical to the split used for NR-T; training seeds come from --seeds


class DSVDDNoBias(nn.Module):
    def __init__(self, d_in, d_lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128, bias=False), nn.ReLU(),
                                 nn.Linear(128, 64, bias=False), nn.ReLU(),
                                 nn.Linear(64, d_lat, bias=False))
        self.c = None

    def forward(self, x):
        return self.net(x)


def fit(x, d_lat, epochs=30, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DSVDDNoBias(x.shape[1], d_lat)
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        m.c = m(xt[:1000]).mean(0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-6)
    n, bs = len(xt), 512
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            z = m(xt[perm[i:i + bs]])
            loss = ((z - m.c) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


def score(m, x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 65536):
            z = m(torch.tensor(x[i:i + 65536], dtype=torch.float32))
            out.append(((z - m.c) ** 2).sum(1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ton', 'unsw', 'cic'])
    ap.add_argument('--seeds', default='42',
                    help='comma-separated training seeds; the split always uses seed 42')
    args = ap.parse_args()
    name = args.dataset
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    t0 = time.time()

    if name == 'unsw':
        from unsw_fix_2025 import build, split
        x, _, d, y, cats = build()
        itr, ite = split(y)
    else:
        x, _, d, y, cats = get_data(name)
        rng = np.random.RandomState(SEED)
        tr, te = [], []
        for k in np.unique(y):
            ix = np.where(y == k)[0]
            rng.shuffle(ix)
            h = len(ix) // 2
            tr.append(ix[:h])
            te.append(ix[h:])
        itr, ite = np.concatenate(tr), np.concatenate(te)

    X_d, y_d, c_d = d[itr], y[itr], cats[itr]
    X_e, y_e, c_e = d[ite], y[ite], cats[ite]
    print(f'[{name}] fit_normals={x.shape} dev={X_d.shape} eval={X_e.shape}', flush=True)

    rows = []
    for seed in seeds:
        best = None
        for d_lat in (16, 32):
            t = time.time()
            m = fit(x, d_lat, seed=seed)
            sd, se = score(m, X_d), score(m, X_e)
            agg, _ = macro_metrics(y_d, c_d, sd)
            print(f'  seed={seed} DSVDD-nobias d_lat={d_lat} dev_tpr1={agg["macro_tpr_1"]:.4f} '
                  f'({time.time()-t:.1f}s)', flush=True)
            if best is None or agg['macro_tpr_1'] > best[0]:
                best = (agg['macro_tpr_1'], d_lat, sd, se)

        agg_d, _ = macro_metrics(y_d, c_d, best[2])
        agg_e, _ = macro_metrics(y_e, c_e, best[3])
        row = {'dataset': name, 'model': 'DeepSVDD_nobias', 'seed': seed, 'd_lat': best[1],
               'dev_tpr1': round(agg_d['macro_tpr_1'], 6),
               'dev_auroc': round(agg_d['macro_auroc'], 6),
               'eval_tpr1': round(agg_e['macro_tpr_1'], 6),
               'eval_auroc': round(agg_e['macro_auroc'], 6)}
        rows.append(row)
        print(f'{name} seed={seed} DeepSVDD-nobias dev={row["dev_tpr1"]:.4f} '
              f'eval={row["eval_tpr1"]:.4f} auroc={row["eval_auroc"]:.4f}', flush=True)

    out = HERE.parent / 'results'
    suffix = '' if len(seeds) == 1 else '_seeds'
    (out / f'dsvdd_nobias_{name}{suffix}.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / f'dsvdd_nobias_{name}{suffix}.csv').open('w', encoding='utf-8') as f:
        f.write('dataset,model,seed,d_lat,dev_tpr1,dev_auroc,eval_tpr1,eval_auroc\n')
        for row in rows:
            f.write(f'{name},DeepSVDD_nobias,{row["seed"]},{row["d_lat"]},'
                    f'{row["dev_tpr1"]:.6f},{row["dev_auroc"]:.6f},'
                    f'{row["eval_tpr1"]:.6f},{row["eval_auroc"]:.6f}\n')

    vals = [r['eval_tpr1'] for r in rows]
    if len(vals) > 1:
        print(f'{name} eval_tpr1 mean={np.mean(vals):.4f} sd={np.std(vals):.4f} '
              f'min={min(vals):.4f} max={max(vals):.4f}', flush=True)
    print(f'[{name}] total {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
