"""CIC-IDS2017 cross-day experiment with time-isolated preprocessing.

Same row selection and model hyper-parameters as cic_day_split.py, with one
change: every fitted preprocessing statistic (median fill, log1p column set,
zero-variance columns, min-max range) is estimated only from the first
nday-2 capture days of the training normals. The previous version reused the
scaler fitted on the original random split, which had seen 23,066 normals from
the held-out days.

Output: results/cic_day_split_clean.csv, results/cic_day_split_clean_meta.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
ROOT = TRAE / 'CICIDS2017_20261107'
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
from paired_2025_cic import train  # noqa: E402
from baselines import macro_metrics  # noqa: E402

RAW = ROOT / 'raw' / 'CICIDS_Flow.parquet'
RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}
NS = 1_000_000_000
SEED = 42
MIN_CLASS_TOTAL = 1500
CAP_TRAIN, CAP_CAL = 60000, 10000
CAP_TEST_NORMAL, CAP_TEST_ATTACK = 50000, 20000
DROP = ['flow_id', 'source_ip', 'destination_ip', 'Timestamp']


def day_ids(*ts_arrays):
    u = np.unique(np.concatenate(ts_arrays))
    cuts = (u[:-1][np.diff(u) > 8 * 3600 * NS] + u[1:][np.diff(u) > 8 * 3600 * NS]) // 2
    return [np.searchsorted(cuts, t) for t in ts_arrays], len(cuts) + 1


def knn(ref, q):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(q)[0][:, 0], 0))


def evaluate(s, cal, y, cats):
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    rec = [float(pred[cats == c].mean()) for c in sorted(set(cats.tolist()) - NORMAL)]
    return agg['macro_tpr_1'], agg['macro_auroc'], float(np.mean(rec)), float(pred[y == 0].mean())


def take(d, mask, cap, rng):
    d = d[mask]
    if len(d) > cap:
        d = d.iloc[rng.choice(len(d), cap, replace=False)]
    return d.sort_index()


def main():
    df = pd.read_parquet(RAW)
    print(f'loaded {len(df):,} x {df.shape[1]}', flush=True)

    # selection identical to prepare.py
    feat_all = [c for c in df.columns if c not in DROP]
    df = df.drop_duplicates(subset=feat_all).reset_index(drop=True)
    vc = df['attack_label'].value_counts()
    keep = vc[vc >= MIN_CLASS_TOTAL].index.tolist()
    df = df[df['attack_label'].isin(keep)].reset_index(drop=True)

    tr, rest = train_test_split(df, test_size=0.30, stratify=df['attack_label'],
                                random_state=SEED)
    ca, te = train_test_split(rest, test_size=2 / 3, stratify=rest['attack_label'],
                              random_state=SEED)
    rng = np.random.default_rng(SEED)
    tr = take(tr, tr['attack_label'].eq('BENIGN'), CAP_TRAIN, rng)
    ca = take(ca, ca['attack_label'].eq('BENIGN'), CAP_CAL, rng)
    te_n = take(te, te['attack_label'].eq('BENIGN'), CAP_TEST_NORMAL, rng)
    te_a = te[~te['attack_label'].eq('BENIGN')]
    te_a = pd.concat([take(g, np.ones(len(g), bool), CAP_TEST_ATTACK, rng)
                      for _, g in te_a.groupby('attack_label')])
    te = pd.concat([te_n, te_a]).sort_index()
    print(f'split: train {len(tr):,} cal {len(ca):,} test {len(te):,}', flush=True)

    proto = df['protocol'].astype('string').fillna('NA').str.upper()
    dummies = pd.get_dummies(proto, prefix='proto').astype(np.float32)
    for c in dummies.columns:
        df[c] = dummies[c].values

    def ts_of(d):
        return pd.to_datetime(d['Timestamp'], format='mixed',
                              dayfirst=True).astype('int64').values

    ts_tr, ts_ca, ts_te = ts_of(tr), ts_of(ca), ts_of(te)
    (d_tr, d_cal, d_te), nday = day_ids(ts_tr, ts_ca, ts_te)
    print('days:', nday, 'train', np.bincount(d_tr, minlength=nday),
          'cal', np.bincount(d_cal, minlength=nday),
          'test', np.bincount(d_te, minlength=nday), flush=True)
    early = nday - 2
    fit_tr, fit_ca = d_tr < early, d_cal < early
    held = d_te >= early
    print(f'fit normals={fit_tr.sum()}  cal normals={fit_ca.sum()}  '
          f'held-out rows={held.sum()} (normals={(te["attack_label"].eq("BENIGN").values[held]).sum()})',
          flush=True)

    # preprocessing fitted on the fit-day training normals only
    num = [c for c in df.columns if c not in DROP + ['attack_label', 'protocol']]
    X = pd.concat([df.loc[tr.index], df.loc[ca.index], df.loc[te.index]])[num].astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    fit_idx = tr.index[fit_tr]
    med = X.loc[fit_idx].median()
    X = X.fillna(med).fillna(0.0)
    nonneg = [c for c in num if X.loc[fit_idx, c].min() >= 0]
    X[nonneg] = np.log1p(X[nonneg].clip(lower=0))
    lo, hi = X.loc[fit_idx].min(), X.loc[fit_idx].max()
    const = [c for c in num if hi[c] <= lo[c]]
    use = [c for c in num if c not in const]
    print(f'zero-variance columns on fit days: {len(const)}; using {len(use)} features', flush=True)
    X = (X[use] - lo[use]) / (hi[use] - lo[use])
    X = X.clip(0.0, 1.0).astype(np.float32)

    x_ref = X.loc[tr.index].values[fit_tr]
    x_caln = X.loc[ca.index].values[fit_ca]
    xe = X.loc[te.index].values[held]
    classes = sorted(te['attack_label'].unique().tolist())
    y = te['attack_label'].map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    cats = np.asarray(classes, dtype=str)[y]
    ye, ce = y[held], cats[held]

    model = train(x_ref, SEED)
    with torch.no_grad():
        ht = model.encoder(torch.from_numpy(x_ref).float()).numpy()
        hc = model.encoder(torch.from_numpy(x_caln).float()).numpy()
        he = model.encoder(torch.from_numpy(xe).float()).numpy()
        rc = model(torch.from_numpy(x_caln).float()).numpy().ravel()
        re = model(torch.from_numpy(xe).float()).numpy().ravel()

    arms = {'native': (rc, re), 'raw1nn': (knn(x_ref, x_caln), knn(x_ref, xe)),
            'lat1nn': (knn(ht, hc), knn(ht, he))}
    n_held = np.where(ye == 0)[0]
    atk_mask = ye != 0
    out = RES / 'cic_day_split_clean.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('arm,macro_tpr1,macro_auroc,recall99,fpr,day_recal_recall,day_recal_fpr\n')
        for arm, (sc, sq) in arms.items():
            t1, au, r99, fpr = evaluate(sq, sc, ye, ce)
            recs, fprs = [], []
            for rep in range(20):
                rng2 = np.random.RandomState(500 + rep)
                perm = rng2.permutation(n_held)
                h = len(perm) // 2
                cut = float(np.quantile(sq[perm[:h]], .99, method='higher'))
                rec = [float((sq[atk_mask & (ce == c)] > cut).mean())
                       for c in sorted(set(ce.tolist()) - NORMAL)]
                recs.append(np.mean(rec))
                fprs.append(float((sq[perm[h:]] > cut).mean()))
            f.write(f'{arm},{t1:.6f},{au:.6f},{r99:.6f},{fpr:.6f},'
                    f'{np.mean(recs):.6f},{np.mean(fprs):.6f}\n')
            print(f'{arm:7s} tpr1={t1:.4f} auroc={au:.4f} recall99={r99:.4f} '
                  f'fpr={fpr:.4f} | day-recal recall={np.mean(recs):.4f} '
                  f'fpr={np.mean(fprs):.4f}', flush=True)

    meta = {'days': int(nday), 'early_days_for_fit': int(early),
            'n_fit_normals': int(fit_tr.sum()), 'n_cal_normals': int(fit_ca.sum()),
            'n_held_rows': int(held.sum()), 'n_held_normals': int((ye == 0).sum()),
            'n_features': len(use), 'dropped_zero_variance': len(const),
            'preprocess_fit': 'first nday-2 capture days of training normals only'}
    (RES / 'cic_day_split_clean_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
