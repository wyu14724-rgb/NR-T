"""Training-objective module: tail-risk (CVaR) + representation-compactness terms.

Everything except the training objective is held fixed:
  * PN architecture, prototype basis, initialisation: the official Model.py, imported unchanged.
  * optimisation: Adam(lr .05, wd 1e-5), ExponentialLR(.98), 200 epochs, batch 512, no shuffle.
  * base loss: mean squared latent residual + 0.1 cosine input-info + 0.06 separation.
  * readout / threshold / metric: identical to reports 03-06.

Added terms (per mini-batch, both scale-free):
  L_tail = CVaR_alpha(s) / mean(s)        , s = per-sample residual score
  L_comp = mean_i (||h_i||^2 / mean_j||h_j||^2 - 1)^2

Rationale: ROC and the 1% operating point depend only on the *relative* score
distribution, so both terms are deliberately scale-invariant; that removes the
degenerate "shrink everything" direction.

Decision rule (frozen):
  primary readout = kNN k=1 on the learned latent (same readout as reports 03-06)
  ADOPT iff, on BOTH datasets, kNN k=1 macro TPR@FPR<=1% is
      (a) no more than 1.0pp below the baseline kNN k=1, and
      (b) strictly above the baseline kNN k=1.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

BASE = STUDY
TRAE = BASE / 'artifacts' / 'current'
TON = BASE / 'artifacts' / 'ton'
UNSW = BASE / 'artifacts' / 'drl_unsw'
ROOT = TRAE / 'tail_objective'
RES = ROOT / 'results'
OFFICIAL = BASE / 'source' / 'DRL-7ff8fadada03564cfbc13f2ba66febc17cf4867d'
COMPAT = BASE / 'artifacts' / 'drl_baseline' / 'source_snapshot' / 'DRL2025_compat'
# The two existing baselines were trained with two different, upstream-pinned variants.
# Reproduce each dataset's own baseline rather than silently unifying them.
VARIANTS = {'ton': {'root': COMPAT, 'module': 'DRL_Model.Model', 'cls': 'DRL', 'softmax': True,
                    'native': 'forward'},
            'unsw': {'root': OFFICIAL, 'module': 'PrototypeNet.Model', 'cls': 'PN', 'softmax': False,
                     'native': 'mean_residual'}}

SEEDS = (42, 43, 44)
ALPHA = 0.05
EPOCHS = 200
LR = 0.05
GAMMA = 0.98
BATCH = 512
HIDDEN = 128
BASIS = 5
INFO_RATIO = 0.10
CL_RATIO = 0.06
TOL = 1.0  # pp
# baseline kNN k=1 references, from reports 03/04 (three-seed mean)
REF = {'ton': 0.4824414551790954, 'unsw': 0.482809164081768}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def at_fpr(y, s):
    f, t, _ = roc_curve(y, s, drop_intermediate=False)
    return float(t[f <= .01].max())


def evaluate(y, cats, s, cal):
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    normal = y == 0
    per = []
    for c in sorted(set(cats) - {'normal', 'Normal'}):
        m = cats == c
        pair = normal | m
        per.append({'category': c, 'tpr1': at_fpr(y[pair], s[pair]),
                    'recall99': float(pred[m].mean())})
    return {'auroc': float(roc_auc_score(y, s)),
            'macro_tpr1': float(np.mean([r['tpr1'] for r in per])),
            'macro_recall99': float(np.mean([r['recall99'] for r in per])),
            'fpr': float(pred[normal].mean())}, per


def model_class(name):
    import importlib
    v = VARIANTS[name]
    stub = types.ModuleType('ipdb')
    stub.set_trace = lambda *a, **k: None
    sys.modules.setdefault('ipdb', stub)
    sys.path.insert(0, str(v['root']))
    return getattr(importlib.import_module(v['module']), v['cls'])


def load(name):
    if name == 'ton':
        with np.load(TON / 'prepared' / 'minmax.npz', allow_pickle=False) as z:
            x, c, d = z['x_train'], z['x_calibration'], z['x_dev']
            y, cats = z['y_dev'].astype(np.int64), z['category_dev'].astype(str)
        cats = np.where(y == 0, 'normal', cats)
        return x, c, d, y, cats
    with np.load(UNSW / 'prepared' / 'data.npz', allow_pickle=False) as z:
        x, c, d = z['x_train'], z['x_validation'], z['x_test']
        y, cats = z['y_test'].astype(np.int64), z['attack_cat'].astype(str)
    cats = np.where(y == 0, 'Normal', cats)
    return x, c, d, y, cats


def config(name, dim, seed):
    base = {'data_dim': dim, 'epochs': EPOCHS, 'learning_rate': LR, 'sche_gamma': GAMMA,
            'mask_num': 15, 'lambda': 5, 'device': 'cpu', 'data_dir': 'Data/', 'runs': 1,
            'batch_size': BATCH, 'en_nlayers': 3, 'de_nlayers': 3, 'hidden_dim': HIDDEN,
            'z_dim': 128, 'mask_nlayers': 3, 'random_seed': seed, 'num_workers': 0,
            'diversity': True, 'plearn': False,
            'input_info': True, 'input_info_ratio': INFO_RATIO,
            'cl': True, 'cl_ratio': CL_RATIO, 'eval_each_epoch': False}
    if name == 'ton':
        return {**base, 'dataset_name': 'ToN_developer_archive2025', 'preprocess': 'minmax',
                'basis_vector_num': BASIS}
    return {**base, 'dataset_name': 'UNSW_NB15_official',
            'preprocess': 'numeric_standard_categorical_onehot', 'prototype_num': BASIS}


def cvar(v, alpha):
    k = max(1, int(math.ceil(alpha * v.shape[0])))
    return torch.topk(v, k, dim=0).values.mean()


def train_model(name, seed, lam_tail, lam_comp, verbose=True):
    Model = model_class(name)
    softmax = VARIANTS[name]['softmax']
    x_train, x_cal, x_test, y, cats = load(name)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if name == 'unsw':
        torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    model = Model(config(name, x_train.shape[1], seed)).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=GAMMA)
    xt = torch.from_numpy(x_train)
    n = len(xt)
    model.train()
    diag = {'mean_score': [], 'tail_ratio': [], 'compact': []}
    for epoch in range(EPOCHS):
        tot = 0.0
        ep = {'mean_score': 0.0, 'tail_ratio': 0.0, 'compact': 0.0, 'batches': 0}
        for i in range(0, n, BATCH):
            xb = xt[i:i + BATCH]
            s = model(xb)
            loss = s.mean()
            h = model.encoder(xb)
            x_tilde = model.decoder(h)
            loss = loss + INFO_RATIO * (1 - F.cosine_similarity(x_tilde, xb, dim=-1)).mean()
            h_ = model.phi(xb)
            if softmax:
                h_ = F.softmax(h_, dim=1)
            sel = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
            hs = h_[sel]
            matrix = hs @ hs.T
            mol = torch.sqrt(torch.sum(hs ** 2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(hs.T ** 2, dim=0, keepdim=True))
            matrix = matrix / mol
            eye = torch.eye(hs.shape[0])
            loss = loss + CL_RATIO * ((1 - eye) * matrix).sum() / hs.shape[0] / hs.shape[0]
            n2 = h.square().sum(1)
            r = n2 / (n2.mean().detach() + 1e-12)
            comp = ((r - 1.0) ** 2).mean()
            with torch.no_grad():
                tr = float(cvar(s.detach(), ALPHA) / (s.mean().detach() + 1e-12))
            if lam_tail:
                loss = loss + lam_tail * cvar(s, ALPHA) / (s.mean().detach() + 1e-12)
            if lam_comp:
                loss = loss + lam_comp * comp
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            ep['mean_score'] += float(s.mean().detach())
            ep['tail_ratio'] += tr
            ep['compact'] += float(comp.detach())
            ep['batches'] += 1
        sch.step()
        for key in ('mean_score', 'tail_ratio', 'compact'):
            diag[key].append(ep[key] / ep['batches'])
        if verbose and (epoch % 50 == 0 or epoch == EPOCHS - 1):
            print(f"  epoch {epoch} loss={tot / math.ceil(n / BATCH):.4f} "
                  f"mean_s={diag['mean_score'][-1]:.3e} cvar_ratio={diag['tail_ratio'][-1]:.2f} "
                  f"compact={diag['compact'][-1]:.3f}", flush=True)
    model.eval()
    return model, diag


def encode(model, x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            out.append(model.encoder(torch.from_numpy(x[i:i + 1024])).numpy())
    return np.concatenate(out)


def native(model, x, name):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.from_numpy(x[i:i + 1024])
            if VARIANTS[name]['native'] == 'forward':
                out.append(model(xb).reshape(-1).numpy())
            else:  # adapter readout of the UNSW baseline: mean squared residual
                h = model.encoder(xb)
                out.append((h - model.phi(xb) @ model.prototype).square().mean(-1).numpy())
    return np.concatenate(out)


def knn(ht, hc, hq, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='sqeuclidean', n_jobs=-1).fit(ht)
    dc, _ = nn.kneighbors(hc)
    dq, _ = nn.kneighbors(hq)
    dc = np.sqrt(np.maximum(dc, 0))
    dq = np.sqrt(np.maximum(dq, 0))
    return dc[:, 0], dq[:, 0], dc.mean(1), dq.mean(1)


def one_run(name, seed, lam_tail, lam_comp):
    x_train, x_cal, x_test, y, cats = load(name)
    model, diag = train_model(name, seed, lam_tail, lam_comp)
    ht, hc, hq = encode(model, x_train), encode(model, x_cal), encode(model, x_test)
    kc1, kq1, kc5, kq5 = knn(ht, hc, hq, 5)
    readouts = {'native': (native(model, x_cal, name), native(model, x_test, name)),
                'knn_k1': (kc1, kq1), 'knn_k5': (kc5, kq5)}
    row = {'dataset': name, 'seed': seed, 'lam_tail': lam_tail, 'lam_comp': lam_comp,
           'alpha': ALPHA, 'epochs': EPOCHS,
           'h_norm_mean': float((ht ** 2).sum(1).mean()),
           'h_norm_q99': float(np.quantile((ht ** 2).sum(1), .99)),
           'train_mean_score_end': diag['mean_score'][-1],
           'train_cvar_ratio_end': diag['tail_ratio'][-1],
           'train_compact_end': diag['compact'][-1]}
    per_rows = []
    for rn, (c, q) in readouts.items():
        met, per = evaluate(y, cats, q, c)
        for key, val in met.items():
            row[f'{rn}_{key}'] = val
        if seed == 42:
            for r in per:
                per_rows.append({'dataset': name, 'seed': seed, 'lam_tail': lam_tail,
                                 'lam_comp': lam_comp, 'readout': rn, **r})
        if name == 'ton':
            for half in (0, 1):
                m = np.arange(len(y)) % 2 == half
                mh, _ = evaluate(y[m], cats[m], q[m], c)
                row[f'{rn}_macro_tpr1_half{half}'] = mh['macro_tpr1']
    print(f"{name} seed{seed} lam_tail={lam_tail} lam_comp={lam_comp}: "
          f"native={row['native_macro_tpr1']:.4f} knn1={row['knn_k1_macro_tpr1']:.4f} "
          f"knn5={row['knn_k5_macro_tpr1']:.4f}", flush=True)
    return row, per_rows


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'code').mkdir(exist_ok=True)
    p = ROOT / 'protocol_tail.json'
    if p.exists():
        raise FileExistsError(p)
    ins = [TON / 'prepared' / 'minmax.npz', UNSW / 'prepared' / 'data.npz',
           TON / 'runs' / 'DRL2025' / 'seed42' / 'scores.npz',
           UNSW / 'runs' / 'baseline_seed42' / 'results' / 'scores.npz',
           OFFICIAL / 'PrototypeNet' / 'Model.py',
           TRAE / 'density_head' / 'results' / 'density_summary.csv']
    proto = {
        'purpose': 'Replace nothing but the training objective: add a CVaR tail-risk term and a scale-free representation-compactness term; architecture and readout unchanged.',
        'official_source': str(OFFICIAL),
        'architecture': {'en_nlayers': 3, 'de_nlayers': 3, 'hidden_dim': HIDDEN, 'prototype_num': BASIS,
                         'basis': 'fixed orthogonal random vectors, plearn=False, diversity=True'},
        'optimisation': {'optimizer': 'Adam', 'lr': LR, 'weight_decay': 1e-5, 'scheduler': f'ExponentialLR gamma={GAMMA}',
                         'epochs': EPOCHS, 'batch': BATCH, 'shuffle': False, 'torch_threads': 1,
                         'deterministic': True},
        'base_loss': 'mean residual score + 0.10*(1-cosine(decoder(h),x)) + 0.06*separation',
        'added_terms': {'tail': 'lam_tail * CVaR_alpha(s)/mean(s.detach())',
                        'compact': 'lam_comp * mean((||h||^2/mean||h||^2 - 1)^2)'},
        'alpha': ALPHA,
        'screen': {'dataset': 'ton', 'seed': 42, 'grid_lam_tail': [0.0, 0.001, 0.01, 0.05], 'lam_comp': 0.01,
                   'control': {'lam_tail': 0.0, 'lam_comp': 0.0,
                               'purpose': 'in-harness reference; also checked against the historical saved scores'},
                   'selection': 'highest knn_k1 macro TPR@1% on even-index dev rows of seed 42; odd-index rows held for confirmation'},
        'confirm': {'datasets': ['ton', 'unsw'], 'seeds': list(SEEDS),
                    'runs': 'selected lam_tail and lam_tail=0 (paired), lam_comp fixed at 0.01'},
        'rules': {'threshold': '99% quantile (method=higher) of calibration normals',
                  'metric': 'macro TPR at FPR<=1% over attack categories (threshold-free ROC per category)',
                  'readouts': ['native residual', 'kNN k=1 on latent', 'kNN k=5 on latent'],
                  'primary_readout': 'knn_k1'},
        'baseline_reference_knn_k1': REF,
        'harness_note': 'The historical ToN model is not bitwise reproducible in the current environment; a 1-epoch control is bitwise identical to the pinned compat Trainer, so lam_tail=0 is re-run in this harness as the paired reference.',
        'decision_rule': (f'ADOPT iff on BOTH datasets the 3-seed mean knn_k1 of the chosen lam is (i) within {TOL}pp of the '
                          'historical reference and (ii) strictly above the paired lam_tail=0 mean'),
        'inputs': {str(q): sha(q) for q in ins},
        'script_sha256': sha(Path(__file__)),
        'sealed_final_touched': False,
    }
    p.write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding='utf-8')
    print('frozen')


def append(name, rows, per_rows):
    RES.mkdir(parents=True, exist_ok=True)
    f = RES / 'tail_runs.csv'
    df = pd.concat([pd.read_csv(f)] if f.exists() else [], ignore_index=True) if f.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(f, index=False, encoding='utf-8-sig')
    if per_rows:
        pf = RES / 'tail_per_category.csv'
        pdf = pd.concat([pd.read_csv(pf)] if pf.exists() else [], ignore_index=True) if pf.exists() else pd.DataFrame()
        pdf = pd.concat([pdf, pd.DataFrame(per_rows)], ignore_index=True)
        pdf.to_csv(pf, index=False, encoding='utf-8-sig')


def control():
    """lambda=0 must reproduce each dataset's own baseline pipeline before any term is added."""
    out = []
    for name, path in (('ton', TON / 'runs' / 'DRL2025' / 'seed42' / 'scores.npz'),
                       ('unsw', UNSW / 'runs' / 'baseline_seed42' / 'results' / 'scores.npz')):
        seed = 42
        x_train, x_cal, x_test, y, cats = load(name)
        model, _ = train_model(name, seed, 0.0, 0.0)
        s = native(model, x_test, name)
        with np.load(path, allow_pickle=False) as z:
            base = z['development_scores'] if name == 'ton' else z['test_scores']
        res = {'dataset': name, 'seed': seed, 'max_abs_diff': float(np.abs(s - base).max()),
               'pearson': float(np.corrcoef(s, base)[0, 1]),
               'identical': bool(np.array_equal(s, base)),
               'macro_tpr1_recomputed': evaluate(y, cats, s, native(model, x_cal, name))[0]['macro_tpr1']}
        print(json.dumps(res), flush=True)
        out.append(res)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'control_lambda0.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'run', 'control'])
    ap.add_argument('--dataset', default='ton')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lam-tail', type=float, default=0.0)
    ap.add_argument('--lam-comp', type=float, default=0.0)
    a = ap.parse_args()
    if a.stage == 'freeze':
        freeze()
        return
    if a.stage == 'control':
        control()
        return
    proto = json.loads((ROOT / 'protocol_tail.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == proto['script_sha256'], 'script changed after freeze'
    for f, h in proto['inputs'].items():
        assert sha(f) == h, f
    row, per = one_run(a.dataset, a.seed, a.lam_tail, a.lam_comp)
    append(a.dataset, [row], per)


if __name__ == '__main__':
    main()
