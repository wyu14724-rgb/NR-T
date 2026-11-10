"""Exact-duplicate audit across the three data partitions.

Reviewer concern: ToN-IoT has only 2,549 unique latent vectors among 30,007
training normals, so the same record may appear in train / calibration / test.
For a 1-NN score this matters twice: a test normal can match a training vector
at distance zero, and duplicated attacks can cross the split.

The audit answers this in the space that decides the score (frozen latent
codes), for all three datasets, in four parts:

  1. unique rows in the reference set (ht), calibration set (hc) and query
     pool (hq), plus exact cross-set overlaps;
  2. share of query normals / attacks with an exact duplicate in the reference
     set (distance zero = leakage at the score level);
  3. NR-T mTPR@1% with the full reference set vs with duplicated reference
     rows dropped (same seed-42 evaluation half as the paper);
  4. raw-feature duplicates for ToN-IoT, the dataset where the concern is
     raised (train vs calibration vs test pool).

Output: results/dup_audit.csv
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
from baselines import get_data  # noqa: E402

CACHE = TRAE / 'cache'
RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}
ALPHA = 0.01
SEED = 42
FILES = {'ton': 'ton_s42.npz', 'unsw': 'unsw_s42_2025.npz', 'cic': 'cic_s42_2025.npz'}


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


def unique_rows(a):
    return np.unique(a, axis=0)


def in_ref(q, ref_ids_hashes=None, ref=None):
    """Boolean mask: row of q appears in ref (exact)."""
    allr = np.vstack([ref, q])
    _, inv = np.unique(allr, axis=0, return_inverse=True)
    a, b = inv[:len(ref)], inv[len(ref):]
    return np.isin(b, np.unique(a))


def macro_tpr1(norm, cls):
    t = []
    for s in cls:
        sc = np.r_[norm, s]
        lb = np.r_[np.zeros(len(norm)), np.ones(len(s))]
        fpr, tpr, _ = roc_curve(lb, sc)
        ok = fpr <= ALPHA
        t.append(tpr[ok].max())
    return float(np.mean(t))


def nn_scores(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=-1).fit(ref)
    d, _ = nn.kneighbors(q)
    return (d[:, 0].astype(np.float64)) ** 2


def main():
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        x_tr, x_cal, x_test, y, cats = get_data(name)
        y = np.asarray(y)
        cats = np.asarray(cats)
        assert len(x_test) == len(y), (name, len(x_test), len(y))
        z = np.load(CACHE / FILES[name])
        ht, hq = z['ht'], z['hq']
        assert len(hq) == len(y), (name, len(hq), len(y))

        _, te_idx = split(y)

        # ---- latent space ----
        hu = unique_rows(ht)
        hit = in_ref(hq, ref=ht)
        qnorm = y == 0
        atk_cats = sorted(set(cats.tolist()) - NORMAL)
        qatk = ~qnorm
        s_full = nn_scores(ht, hq[te_idx])
        s_u = nn_scores(hu, hq[te_idx])
        norm_mask_te = (y == 0)[te_idx]
        cls_masks = [(cats == c)[te_idx] for c in atk_cats]
        tpr_full = macro_tpr1(s_full[norm_mask_te], [s_full[m] for m in cls_masks])
        tpr_u = macro_tpr1(s_u[norm_mask_te], [s_u[m] for m in cls_masks])
        del s_full, s_u
        rows.append({
            'dataset': name, 'space': 'latent',
            'n_ref': len(ht), 'n_ref_unique': len(hu),
            'n_query': len(hq),
            'q_normal_with_exact_ref': float(hit[qnorm].mean()) if qnorm.any() else float('nan'),
            'q_attack_with_exact_ref': float(hit[qatk].mean()) if qatk.any() else float('nan'),
            'n_eval_normal_exact': int((hit[te_idx] & qnorm[te_idx]).sum()),
            'mTPR1_full': tpr_full, 'mTPR1_dedup': tpr_u,
        })
        print(f"[{name}] latent: ref {len(ht)} ({len(hu)} unique), "
              f"query normal dup {rows[-1]['q_normal_with_exact_ref']:.4f}, "
              f"attack dup {rows[-1]['q_attack_with_exact_ref']:.4f}, "
              f"mTPR1 {tpr_full:.4f} -> {tpr_u:.4f}", flush=True)
        del ht, hq, hu, hit, z

        # ---- raw space ----
        rt = unique_rows(x_tr)
        hit_r = in_ref(x_test, ref=x_tr)
        rows.append({
            'dataset': name, 'space': 'raw',
            'n_ref': len(x_tr), 'n_ref_unique': len(rt),
            'n_query': len(x_test),
            'q_normal_with_exact_ref': float(hit_r[qnorm].mean()) if qnorm.any() else float('nan'),
            'q_attack_with_exact_ref': float(hit_r[qatk].mean()) if qatk.any() else float('nan'),
            'n_eval_normal_exact': int((hit_r[te_idx] & qnorm[te_idx]).sum()),
            'mTPR1_full': float('nan'), 'mTPR1_dedup': float('nan'),
        })
        hit_c = in_ref(x_cal, ref=x_tr)
        rows.append({
            'dataset': name, 'space': 'raw_cal_vs_train',
            'n_ref': len(x_tr), 'n_ref_unique': len(rt),
            'n_query': len(x_cal),
            'q_normal_with_exact_ref': float(hit_c.mean()),
            'q_attack_with_exact_ref': float('nan'),
            'n_eval_normal_exact': int(hit_c.sum()),
            'mTPR1_full': float('nan'), 'mTPR1_dedup': float('nan'),
        })
        print(f"[{name}] raw: ref {len(x_tr)} ({len(rt)} unique), "
              f"test-normal dup {rows[-2]['q_normal_with_exact_ref']:.4f}, "
              f"cal dup {rows[-1]['q_normal_with_exact_ref']:.4f}", flush=True)
        del x_tr, x_cal, x_test, rt, hit_r, hit_c

    out = RES / 'dup_audit.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
