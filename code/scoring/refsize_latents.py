"""Build latents that match the paper's checkpoints, for the reference-size ablation.

The shared cache in cache/ was written before two checkpoint changes:
UNSW-NB15 was switched to the min-max checkpoint (final_minmax_seed42) and
CIC-IDS2017 was retrained with the ICLR 2025 implementation. Their cached
latents therefore reproduce the superseded numbers (0.4902 / 0.3981), not the
ones in the paper (0.4949 / 0.4706). This script recreates the correct latents
and stores them as {name}_s42_2025.npz. ToN-IoT is unaffected and still uses the
original cache.

Usage: python refsize_latents.py --dataset unsw|cic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
CACHE = TRAE / 'cache'


def unsw():
    from unsw_fix_2025 import RUNS, build
    x_tr, _, x_te, _, _ = build()
    model = torch.load(RUNS / 'final_minmax_seed42' / 'model.pt',
                       map_location='cpu', weights_only=False)
    model.eval()
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x_tr).float()).numpy()
        hq = model.encoder(torch.from_numpy(x_te).float()).numpy()
    np.savez(CACHE / 'unsw_s42_2025.npz', ht=ht, hq=hq)
    print('saved unsw_s42_2025.npz', ht.shape, hq.shape, flush=True)


def cic():
    import paired_2025_cic as P
    x, _, d, _, _ = P.cic.load()
    model = P.train(x, 42)
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x).float()).numpy()
        hq = model.encoder(torch.from_numpy(d).float()).numpy()
    np.savez(CACHE / 'cic_s42_2025.npz', ht=ht, hq=hq)
    print('saved cic_s42_2025.npz', ht.shape, hq.shape, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['unsw', 'cic'])
    a = ap.parse_args()
    {'unsw': unsw, 'cic': cic}[a.dataset]()
