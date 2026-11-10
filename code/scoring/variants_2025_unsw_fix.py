"""UNSW scoring-variant screen on the MinMax-correct inputs (2025, seed 42)."""
import os
import sys
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import numpy as np
import torch

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat'))
from baselines import macro_metrics  # noqa: E402
import variants_2025 as V  # noqa: E402
import unsw_fix_2025 as F  # noqa: E402


def main():
    x_tr, x_cal, x_te, y, cats = F.build()
    x_cal = x_cal[:2800]
    _, ite = F.split(y)
    m = torch.load(V.RUNS / 'final_minmax_seed42' / 'model.pt', map_location='cpu', weights_only=False)
    m.eval()
    with torch.no_grad():
        ht = m.encoder(torch.from_numpy(x_tr).float()).numpy()
        hc = m.encoder(torch.from_numpy(x_cal).float()).numpy()
        hq = m.encoder(torch.from_numpy(x_te).float()).numpy()
        rq = m(torch.from_numpy(x_te).float()).numpy().ravel()
    sc = V.variants(hq[ite], ht, hc)
    base = None
    rows = []
    for v, s in sc.items():
        agg, _ = macro_metrics(y[ite], cats[ite], s)
        rows.append((v, agg['macro_tpr_1'], agg['macro_auroc']))
        if v == 'd1':
            base = agg['macro_tpr_1']
    for v, t1, au in rows:
        print(f'unsw {v:7s} tpr1={t1:.4f} auroc={au:.4f} delta={100 * (t1 - base):+.2f}pp', flush=True)
    with (V.RES / 'variants_2025_unsw_fix.csv').open('w', encoding='utf-8') as f:
        f.write('variant,macro_tpr1,macro_auroc\n')
        for v, t1, au in rows:
            f.write(f'{v},{t1:.6f},{au:.6f}\n')


if __name__ == '__main__':
    main()
