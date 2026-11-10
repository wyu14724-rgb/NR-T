"""CIC-IDS2017 preparation for the DRL/normal-only protocol.

Protocol (a): stratified random 7:1:2 split, train/calibration contain BENIGN only,
test contains BENIGN + attack classes with >= 1500 total flows.
Everything (caps, seeds, thresholds) is frozen in protocol_cicids.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r'D:\实验\选题候选10.27\选题2\实验产物\TRAE接手\CICIDS2017_20261107')
RAW = ROOT / 'raw' / 'CICIDS_Flow.parquet'
DATA = ROOT / 'data'
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_CLASS_TOTAL = 1500
CAP_TRAIN = 60000
CAP_CAL = 10000
CAP_TEST_NORMAL = 50000
CAP_TEST_ATTACK = 20000
NORMAL = 'BENIGN'
DROP = ['flow_id', 'source_ip', 'destination_ip', 'Timestamp']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    df = pd.read_parquet(RAW)
    print(f'loaded {len(df):,} x {df.shape[1]}')

    # --- de-duplicate: identical flows leak across splits, inflating AUROC
    feat_all = [c for c in df.columns if c not in DROP]
    n0 = len(df)
    df = df.drop_duplicates(subset=feat_all).reset_index(drop=True)
    print(f'dedup: {n0:,} -> {len(df):,} ({n0 - len(df):,} removed)')

    vc = df['attack_label'].value_counts()
    keep = vc[vc >= MIN_CLASS_TOTAL].index.tolist()
    dropped = {k: int(v) for k, v in vc.items() if k not in keep}
    df = df[df['attack_label'].isin(keep)].reset_index(drop=True)
    print(f'classes kept {len(keep)}; dropped {dropped}')

    # --- protocol (a): stratified 7:1:2
    tr, rest = train_test_split(df, test_size=0.30, stratify=df['attack_label'],
                                random_state=SEED)
    ca, te = train_test_split(rest, test_size=2 / 3, stratify=rest['attack_label'],
                              random_state=SEED)
    rng = np.random.default_rng(SEED)

    def take(d, mask, cap):
        d = d[mask]
        if len(d) > cap:
            d = d.iloc[rng.choice(len(d), cap, replace=False)]
        return d.sort_index()

    tr = take(tr, tr['attack_label'].eq(NORMAL), CAP_TRAIN)
    ca = take(ca, ca['attack_label'].eq(NORMAL), CAP_CAL)
    te_n = take(te, te['attack_label'].eq(NORMAL), CAP_TEST_NORMAL)
    te_a = te[~te['attack_label'].eq(NORMAL)]
    te_a = pd.concat([take(g, np.ones(len(g), bool), CAP_TEST_ATTACK)
                      for _, g in te_a.groupby('attack_label')])
    te = pd.concat([te_n, te_a]).sort_index()
    print(f'split: train {len(tr):,} cal {len(ca):,} test {len(te):,}')

    # --- features
    proto = df['protocol'].astype('string').fillna('NA').str.upper()
    dummies = pd.get_dummies(proto, prefix='proto').astype(np.float32)
    for c in dummies.columns:
        df[c] = dummies[c].values
    num = [c for c in df.columns if c not in DROP + ['attack_label', 'protocol']]
    X = df[num].astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)

    tr_idx, ca_idx, te_idx = tr.index, ca.index, te.index
    med = X.loc[tr_idx].median()
    X = X.fillna(med).fillna(0.0)

    # log1p on non-negative heavy-tailed columns (fitted on train only)
    nonneg = [c for c in num if X.loc[tr_idx, c].min() >= 0]
    X[nonneg] = np.log1p(X[nonneg].clip(lower=0))

    lo, hi = X.loc[tr_idx].min(), X.loc[tr_idx].max()
    const = [c for c in num if hi[c] <= lo[c]]
    use = [c for c in num if c not in const]
    print(f'zero-variance columns dropped: {len(const)}; using {len(use)} features')
    X = (X[use] - lo[use]) / (hi[use] - lo[use])
    X = X.clip(0.0, 1.0).astype(np.float32)

    classes = sorted(te['attack_label'].unique().tolist())
    y = te['attack_label'].map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)

    np.savez_compressed(DATA / 'cicids2017.npz',
                        x_train=X.loc[tr_idx].values, x_cal=X.loc[ca_idx].values,
                        x_test=X.loc[te_idx].values, y_test=y,
                        classes=np.array(classes),
                        ts_train=pd.to_datetime(df.loc[tr_idx, 'Timestamp'], format='mixed', dayfirst=True).astype('int64').values,
                        ts_cal=pd.to_datetime(df.loc[ca_idx, 'Timestamp'], format='mixed', dayfirst=True).astype('int64').values,
                        ts_test=pd.to_datetime(df.loc[te_idx, 'Timestamp'], format='mixed', dayfirst=True).astype('int64').values)

    counts = te['attack_label'].value_counts().to_dict()
    print('\ntest counts:')
    for c in classes:
        print(f'  {c:28s} {counts[c]:>7,}')

    prot = {
        'script_sha256': sha(__file__),
        'raw_sha256': sha(RAW),
        'dataset': 'CIC-IDS2017 (CICFlowMeter features)',
        'source': 'huggingface.co/datasets/rdpahalavan/CIC-IDS2017 Network-Flows/CICIDS_Flow.parquet',
        'split': 'protocol (a): stratified random 7:1:2, seed 42; train/cal BENIGN only',
        'seed': SEED, 'min_class_total': MIN_CLASS_TOTAL,
        'caps': {'train': CAP_TRAIN, 'cal': CAP_CAL,
                 'test_normal': CAP_TEST_NORMAL, 'test_attack_per_class': CAP_TEST_ATTACK},
        'dropped_classes': dropped, 'dropped_zero_variance': const,
        'n_features': len(use),
        'preprocess': 'inf->NaN; NaN->train median; log1p on train-min>=0; min-max on train; clip[0,1]',
        'test_counts': counts,
        'rows': {'train': int(len(tr)), 'cal': int(len(ca)), 'test': int(len(te))},
    }
    (ROOT / 'protocol_cicids.json').write_text(json.dumps(prot, indent=2, ensure_ascii=False))
    print('\nwrote', DATA / 'cicids2017.npz')


if __name__ == '__main__':
    main()
