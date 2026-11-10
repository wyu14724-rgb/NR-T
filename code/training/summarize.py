"""Aggregate the frozen tail-objective runs (post-hoc, does not touch the protocol)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r'D:\实验\选题候选10.27\选题2\实验产物\TRAE接手\训练目标_20261107')
RES = ROOT / 'results'
REF = {'ton': 0.4824414551790954, 'unsw': 0.482809164081768}

d = pd.read_csv(RES / 'tail_runs.csv')
d = d.drop_duplicates(subset=['dataset', 'seed', 'lam_tail', 'lam_comp'], keep='last')
d['arm'] = d.apply(lambda r: 'pure_baseline' if r.lam_tail == 0 and r.lam_comp == 0
                   else ('comp_only' if r.lam_tail == 0 else f'tail_{r.lam_tail:g}_comp_{r.lam_comp:g}'), axis=1)
d.to_csv(RES / 'tail_runs_dedup.csv', index=False, encoding='utf-8-sig')

rows = []
for (ds, arm), g in d.groupby(['dataset', 'arm']):
    r = {'dataset': ds, 'arm': arm, 'n': len(g)}
    for m in ('knn_k1_macro_tpr1', 'knn_k5_macro_tpr1', 'native_macro_tpr1',
              'knn_k1_auroc', 'native_auroc', 'native_macro_recall99', 'h_norm_mean'):
        r[f'{m}_mean'] = g[m].mean()
        r[f'{m}_std'] = g[m].std(ddof=1)
    r['delta_vs_ref'] = g['knn_k1_macro_tpr1'].mean() - REF[ds]
    rows.append(r)
s = pd.DataFrame(rows).sort_values(['dataset', 'arm'])
s.to_csv(RES / 'tail_summary.csv', index=False, encoding='utf-8-sig')

base = d[(d.lam_tail == 0) & (d.lam_comp == 0)][['dataset', 'seed', 'knn_k1_macro_tpr1', 'native_macro_tpr1']]
base = base.rename(columns={'knn_k1_macro_tpr1': 'base_knn1', 'native_macro_tpr1': 'base_native'})
p = d.merge(base, on=['dataset', 'seed'], how='left')
p = p[p.arm != 'pure_baseline']
p['delta_knn1'] = p.knn_k1_macro_tpr1 - p.base_knn1
p['delta_native'] = p.native_macro_tpr1 - p.base_native
p[['dataset', 'seed', 'arm', 'knn_k1_macro_tpr1', 'base_knn1', 'delta_knn1',
   'native_macro_tpr1', 'base_native', 'delta_native']].to_csv(
    RES / 'tail_paired.csv', index=False, encoding='utf-8-sig')

print(s[['dataset', 'arm', 'n', 'knn_k1_macro_tpr1_mean', 'knn_k1_macro_tpr1_std',
         'delta_vs_ref', 'native_macro_tpr1_mean', 'h_norm_mean_mean']].to_string(index=False))
print()
print(p.groupby(['dataset', 'arm'])[['delta_knn1', 'delta_native']].agg(['mean', 'std']).to_string())
