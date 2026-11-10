"""Rebuild the archived CSV tables (T2 / T9 / T14 / T4) for the frozen paper.

Every number here is re-derived from on-disk artifacts only: baseline JSONs,
the neighbourhood-objective report, the dumped per-sample score archives and the
20-round half-split recalibration protocol of
`code/scoring/recal_2025.py`. No checkpoint is trained or loaded.

Outputs (results/):
  table2_supervised.csv        dataset,model,macro_tpr1,macro_auroc,n_test
  table9_contrastive_5seed.csv seed,base,method,delta_pp
  table14_alpha_sweep.csv      dataset,alpha,arm,macro_tpr
  table4_recalibration.csv     dataset,fpr_mean_pct,fpr_sd_pct,
                               recall_locked_threshold,n_rounds
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

TRAE = Path(__file__).resolve().parents[2]
RES = TRAE / 'results'
OUT = TRAE / 'results' / 'tables'
NB_REPORT = TRAE / 'results' / 'report_10_neighbour.csv'

DATASETS = ('ton', 'unsw', 'cic')
NORMAL = {'normal', 'Normal', 'BENIGN'}
ALPHAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
N_ROUNDS = 20
# documented normal counts of the evaluation half (paper §5.4 / §5.1)
EVAL_HALF_NORMALS = {'ton': 2500, 'unsw': 18500, 'cic': 25000}
PAPER_T14_A1 = {  # (residual, lat1nn) at alpha = 1%, paper §5.9
    'ton': (0.406, 0.481), 'unsw': (0.463, 0.495), 'cic': (0.109, 0.471)}
PAPER_T4 = {  # (fpr_mean_pct, fpr_sd_pct, recall), paper §5.4
    'ton': (0.99, 0.23, 0.482), 'unsw': (1.05, 0.12, 0.495), 'cic': (1.00, 0.13, 0.465)}
PAPER_T2 = {  # macro_tpr_1, paper §5.2
    ('ton', 'RF_sup'): 0.958, ('unsw', 'RF_sup'): 0.941, ('cic', 'RF_sup'): 0.998,
    ('ton', 'HGB_sup'): 0.902, ('unsw', 'HGB_sup'): 0.955, ('cic', 'HGB_sup'): 1.000}
PAPER_T9 = dict(sd_base_pp=1.96, sd_method_pp=0.85, F=5.3, p=0.068, mean_delta_pp=-0.1)


def eval_half(y):
    """Exactly the split of recal_2025.py: stratified halving, RandomState(42)."""
    rng = np.random.RandomState(42)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def load_scores(name):
    with np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(r) + '\n')
    print(f'  wrote {path.name}: {len(rows)} rows')


# ---------------------------------------------------------------- task 1 (T2)
def task2():
    rows = []
    for ds in DATASETS:
        with (RES / f'baselines_{ds}.json').open(encoding='utf-8') as f:
            data = json.load(f)
        by_model = {d['model']: d for d in data}
        for model in ('RF_sup', 'HGB_sup'):
            assert model in by_model, f'{ds}: {model} missing from baselines JSON'
            d = by_model[model]
            assert d['dataset'] == ds
            rows.append((ds, model, f"{d['macro_tpr_1']:.4f}",
                         f"{d['macro_auroc']:.4f}", str(d['n_test'])))
    write_csv(OUT / 'table2_supervised.csv',
              ('dataset', 'model', 'macro_tpr1', 'macro_auroc', 'n_test'), rows)
    print('  T2 vs paper (§5.2, macro TPR@1%):')
    for ds, model, tpr, _, n in rows:
        p = PAPER_T2[(ds, model)]
        print(f'    {ds:4s} {model:7s} reproduced={tpr} paper={p:.3f} '
              f'{"OK" if abs(float(tpr) - p) < 5e-4 else "MISMATCH"}  n_test={n}')
    return rows


# ---------------------------------------------------------------- task 2 (T9)
def task9():
    with NB_REPORT.open(encoding='utf-8', newline='') as fh:
        rows_in = list(csv.DictReader(fh))
    seeds = [int(r['seed']) for r in rows_in]
    assert seeds == [42, 43, 44, 45, 46], seeds
    base = np.array([float(r['base']) for r in rows_in])
    meth = np.array([float(r['method']) for r in rows_in])
    assert len(base) == len(meth) == 5
    md_delta = (meth - base) * 100

    rows = [(f'{s}', f'{b:.4f}', f'{m:.4f}', f'{(m - b) * 100:.4f}')
            for s, b, m in zip(seeds, base, meth)]
    write_csv(OUT / 'table9_contrastive_5seed.csv',
              ('seed', 'base', 'method', 'delta_pp'), rows)

    delta_pp = (meth - base) * 100
    sd_b = base.std(ddof=1) * 100
    sd_m = meth.std(ddof=1) * 100
    F = sd_b ** 2 / sd_m ** 2
    p = float(fdist.sf(F, 4, 4))
    print('  T9 parsed seeds:', seeds)
    print('  delta_pp per seed:', ' '.join(f'{v:+.2f}' for v in delta_pp),
          '| md:', ' '.join(f'{v:+.2f}' for v in md_delta))
    print(f'  mean diff = {delta_pp.mean():+.3f} pp  (paper {PAPER_T9["mean_delta_pp"]})')
    print(f'  sd base (ddof=1)   = {sd_b:.3f} pp  (paper 1.96, md 1.98)')
    print(f'  sd method (ddof=1) = {sd_m:.3f} pp  (paper 0.85)')
    print(f'  F(4,4) = {F:.3f}  (paper 5.317)   p = {p:.4f}  (paper 0.067)')
    print(f'  md table sd row: base 1.98 pp, method 0.85 pp')
    return rows, dict(sd_base_pp=sd_b, sd_method_pp=sd_m, F=F, p=p,
                      mean_delta_pp=float(delta_pp.mean()))


# --------------------------------------------------------------- task 3 (T14)
def threshold_upper(s_norm, alpha):
    n = len(s_norm)
    i = n - int(np.floor(alpha * n)) - 1
    assert 0 <= i < n, (n, alpha)
    return float(np.partition(s_norm, i)[i])


def task14():
    rows = []
    for ds in DATASETS:
        z = load_scores(ds)
        y, cats = z['y_test'], z['cats_test']
        assert len(y) == len(cats) == len(z['test_lat1nn']) == len(z['test_native'])
        _, ite = eval_half(y)
        yte, cte = y[ite], cats[ite]
        norm = yte == 0
        n_norm = int(norm.sum())
        assert n_norm == EVAL_HALF_NORMALS[ds], (ds, n_norm)
        classes = [c for c in np.unique(cte) if c not in NORMAL]
        assert len(classes) == (9 if ds in ('ton', 'unsw') else 10), (ds, len(classes))
        atk = ~norm
        for arm, key in (('residual', 'test_native'), ('lat1nn', 'test_lat1nn')):
            s = z[key][ite].astype(np.float64)
            s_norm = s[norm]
            for a in ALPHAS:
                tau = threshold_upper(s_norm, a)
                tpr = float(np.mean([float((s[atk & (cte == c)] > tau).mean())
                                     for c in classes]))
                rows.append((ds, f'{a:g}', arm, f'{tpr:.4f}'))
                if a == 0.01:
                    pr = PAPER_T14_A1[ds][0 if arm == 'residual' else 1]
                    flag = 'OK' if abs(tpr - pr) <= 5e-4 else 'MISMATCH'
                    print(f'  T14 {ds:4s} alpha=1% {arm:8s} reproduced={tpr:.4f} '
                          f'paper={pr:.3f} {flag} (n_norm={n_norm}, n_classes={len(classes)})')
    write_csv(OUT / 'table14_alpha_sweep.csv',
              ('dataset', 'alpha', 'arm', 'macro_tpr'), rows)
    return rows


# ---------------------------------------------------------------- task 4 (T4)
def task4():
    rows = []
    for ds in DATASETS:
        z = load_scores(ds)
        y, cats = z['y_test'], z['cats_test']
        _, ite = eval_half(y)
        s = z['test_lat1nn'][ite].astype(np.float64)
        sc = z['cal_lat1nn'].astype(np.float64)
        yte, cte = y[ite], cats[ite]
        n_idx = np.where(yte == 0)[0]
        atk = np.where(yte != 0)[0]
        assert len(n_idx) == EVAL_HALF_NORMALS[ds]
        classes = sorted(set(cte[atk].tolist()) - NORMAL)
        assert len(classes) == (9 if ds in ('ton', 'unsw') else 10)

        def macro_recall(cut):
            return float(np.mean([(s[atk][cte[atk] == k] > cut).mean() for k in classes]))

        # cross-distribution check (paper "cross-distribution calibration FPR")
        cut_src = float(np.quantile(sc, .99, method='higher'))
        fpr_src = float((s[n_idx] > cut_src).mean())
        rec_src = macro_recall(cut_src)

        fprs, recs = [], []
        for rep in range(N_ROUNDS):
            perm = np.random.RandomState(rep).permutation(n_idx)
            a, b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            cut = float(np.quantile(s[a], .99, method='higher'))
            fprs.append(float((s[b] > cut).mean()))
            recs.append(macro_recall(cut))
        fpr_mean = float(np.mean(fprs)) * 100
        fpr_sd = float(np.std(fprs)) * 100  # population sd, as in recal_2025.py
        rec = float(np.mean(recs))
        rows.append((ds, f'{fpr_mean:.4f}', f'{fpr_sd:.4f}', f'{rec:.4f}', str(N_ROUNDS)))
        pf, ps, pr = PAPER_T4[ds]
        ok = 'OK' if (abs(fpr_mean - pf) <= 0.01 and abs(fpr_sd - ps) <= 0.01
                      and abs(rec - pr) <= 5e-4) else 'CHECK'
        print(f'  T4 {ds:4s} recal FPR={fpr_mean:.3f}%+-{fpr_sd:.3f}% recall={rec:.4f} '
              f'| paper {pf:.2f}%+-{ps:.2f}% recall {pr} -> {ok}')
        print(f'     source-cal FPR={fpr_src:.3%} recall={rec_src:.4f}')
        if ds == 'unsw':
            # The npz stores cal_lat1nn for all 5,600 calibration normals, while
            # recal_2025_unsw_arms.py (the paper's 4.78%) used x_cal[:2800].
            # Hence only the source-threshold FPR differs; the 20-round target
            # numbers below are computed from test normals alone and are identical.
            print(f'     note: cal_lat1nn has {len(sc)} rows vs 2,800 in the paper, '
                  'so source-FPR is not comparable (target rows are)')
            assert abs(fpr_mean - 1.053) < 0.02 and abs(fpr_sd - 0.1174) < 0.02, \
                (fpr_mean, fpr_sd)
    write_csv(OUT / 'table4_recalibration.csv',
              ('dataset', 'fpr_mean_pct', 'fpr_sd_pct',
               'recall_locked_threshold', 'n_rounds'), rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('== T2 supervised baselines ==', flush=True)
    r2 = task2()
    print('== T9 contrastive five seeds ==', flush=True)
    r9, stats9 = task9()
    print('== T14 alpha sweep ==', flush=True)
    r14 = task14()
    print('== T4 recalibration (UNSW = protocol self-check) ==', flush=True)
    r4 = task4()
    print('== summary ==')
    for name, rows, cols in (
            ('table2_supervised.csv', r2, 'dataset,model,macro_tpr1,macro_auroc,n_test'),
            ('table9_contrastive_5seed.csv', r9, 'seed,base,method,delta_pp'),
            ('table14_alpha_sweep.csv', r14, 'dataset,alpha,arm,macro_tpr'),
            ('table4_recalibration.csv', r4,
             'dataset,fpr_mean_pct,fpr_sd_pct,recall_locked_threshold,n_rounds')):
        print(f'  {name}: {len(rows)} rows | columns: {cols}')
    print('  T9 recomputed stats:', json.dumps(stats9))


if __name__ == '__main__':
    main()
