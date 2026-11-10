"""Oracle-rate correction for a contaminated calibration batch.

Question: if the contamination rate eps were known exactly and we assumed every
mixed-in attack scores above the clean 99% quantile, how far could the operating
point be restored?  This is the best case for the family "take a lower quantile
of the mixture"; it is not a bound for every possible estimator.

Same data, split, caps and rng seeds as calib_stress.py, so the 'naive' rows
reproduce results/calib_stress.csv exactly.

Output: results/oracle_contam.csv
"""
import csv

import numpy as np

from calib_stress import CAP, REPS, RES, op_point, split

ARMS = ('native', 'raw1nn', 'lat1nn')
RATES = (0.01, 0.05, 0.10)
ALPHA = 0.01


def q(v, level):
    return float(np.quantile(v, level, method='higher'))


if __name__ == '__main__':
    rows = []
    for name in ('ton', 'unsw', 'cic'):
        z = np.load(RES / f'scores_{name}_s42.npz', allow_pickle=False)
        y, cats = z['y_test'], z['cats_test'].astype(str)
        _, ite = split(y)
        ye, ce = y[ite], cats[ite]
        attack_pool = np.where(ye != 0)[0]
        for arm in ARMS:
            sc, sq = z[f'cal_{arm}'][:CAP[name]], z[f'test_{arm}'][ite]
            honest = q(sc, 1 - ALPHA)
            r0, f0 = op_point(ye, ce, sq > honest)
            rows.append((name, arm, 0.0, 'honest', 0.0, 0.0, r0, f0, '', '', ''))
            for rate in RATES:
                acc = {k: [] for k in ('naive_r', 'naive_f', 'orc_r', 'orc_f',
                                       'ctl_r', 'ctl_f', 'share', 'micro')}
                for rep in range(REPS):
                    rng = np.random.RandomState(1000 + rep)
                    k = int(round(rate * len(sc)))
                    chosen = rng.choice(attack_pool, k, replace=False)
                    att = sq[chosen]
                    clean = rng.choice(len(sc), len(sc) - k, replace=False)
                    mix = np.concatenate([sc[clean], att])
                    keep = np.ones(len(ye), bool)
                    keep[chosen] = False
                    cut_n = q(mix, 1 - ALPHA)
                    cut_o = q(mix, (1 - ALPHA) * (1 - rate))
                    rn, fn = op_point(ye, ce, (sq > cut_n) & keep, keep)
                    ro, fo = op_point(ye, ce, (sq > cut_o) & keep, keep)
                    rc, fc = op_point(ye, ce, (sq > honest) & keep, keep)
                    micro = float(((sq > cut_o) & (ye != 0) & keep).sum()
                                  / max(1, ((ye != 0) & keep).sum()))
                    share = float((att > q(mix, 1 - ALPHA)).mean())
                    for key, val in (('naive_r', rn), ('naive_f', fn),
                                     ('orc_r', ro), ('orc_f', fo),
                                     ('ctl_r', rc), ('ctl_f', fc),
                                     ('share', share), ('micro', micro)):
                        acc[key].append(val)
                m = {k: float(np.mean(v)) for k, v in acc.items()}
                s = {k: float(np.std(v)) for k, v in acc.items()}
                pred = ALPHA + rate / (1 - rate) * (1 - m['micro'])
                for tag, rr, ff in (('naive', m['naive_r'], m['naive_f']),
                                    ('oracle_clip', m['orc_r'], m['orc_f'])):
                    rows.append((name, arm, rate, tag, s['orc_r'] if tag == 'oracle_clip' else s['naive_r'],
                                 s['orc_f'] if tag == 'oracle_clip' else s['naive_f'], rr, ff,
                                 m['micro'], pred, 100 * (ff - pred)))
                print(f'{name:>4s} {arm:7s} {rate:.0%} '
                      f'clean={m["ctl_r"]:.4f}/{m["ctl_f"]:.4f} '
                      f'naive={m["naive_r"]:.4f}/{m["naive_f"]:.4f} '
                      f'oracle={m["orc_r"]:.4f}/{m["orc_f"]:.4f} '
                      f'share={m["share"]:.2f} '
                      f'ident_fpr={pred:.4f} diff={100 * (m["orc_f"] - pred):+.2f}pp',
                      flush=True)

    out = RES / 'oracle_contam.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'arm', 'rate', 'scenario', 'recall_sd', 'fpr_sd',
                    'recall', 'fpr', 'micro_recall', 'identity_fpr', 'diff_pp'])
        w.writerows(rows)
    print(f'wrote {out}')
