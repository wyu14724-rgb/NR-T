"""CIC-IDS2017 baselines: raw-feature kNN vs DRL latent kNN.

The DRL arm reuses the frozen harness loss/optimiser exactly (see report 07):
  score = model(x); loss = mean(score) + 0.10*(1-cos(decoder(h),x)) + 0.06*separation
  Adam(lr .05, wd 1e-5), ExponentialLR(.98), 200 epochs, batch 512, no shuffle.
Readout: kNN (k=1, k=5) on the 128-d latent, macro TPR at FPR<=1% over attack classes.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

TRAE = Path(r'D:\实验\选题候选10.27\选题2\实验产物\TRAE接手')
ROOT = TRAE / 'CICIDS2017_20261107'
DATA = ROOT / 'data' / 'cicids2017.npz'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(TRAE / 'code' / 'training'))
import tail_objective as base  # noqa: E402

EPOCHS = 200
LR = 0.05
GAMMA = 0.98
BATCH = 512
BASIS = 5
INFO_RATIO = 0.10
CL_RATIO = 0.06
BATCH_SCORE = 1024


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    z = np.load(DATA, allow_pickle=True)
    return (z['x_train'], z['x_cal'], z['x_test'], z['y_test'],
            [str(c) for c in z['classes']])


def macro_tpr01(norm_scores, class_scores, classes=None):
    """Per attack class ROC against the normal scores; TPR at FPR<=1%; macro mean."""
    per = {}
    for c, s in class_scores.items():
        if s is None or len(s) == 0:
            continue
        y = np.r_[np.zeros(len(norm_scores)), np.ones(len(s))]
        sc = np.r_[norm_scores, s]
        fpr, tpr, _ = roc_curve(y, sc)
        m = fpr <= 0.01
        per[c] = {'tpr01': float(tpr[m].max()) if m.any() else 0.0,
                  'auroc': float(roc_auc_score(y, sc)), 'n': int(len(s))}
    macro = float(np.mean([v['tpr01'] for v in per.values()]))
    auroc = float(np.mean([v['auroc'] for v in per.values()]))
    return macro, auroc, per


def split_by_class(y, classes):
    idx = [np.where(y == i)[0] for i in range(len(classes))]
    return idx


def knn_scores(ht, hc, hq, k=5):
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ht)
    d, _ = nn.kneighbors(hq)
    return d[:, 0] if k == 1 else d.mean(1)


def embed(model, x):
    out = []
    with torch.no_grad():
        xt = torch.from_numpy(x)
        for i in range(0, len(xt), BATCH_SCORE):
            out.append(model.encoder(xt[i:i + BATCH_SCORE]).numpy())
    return np.concatenate(out)


def train_drl(x_train, seed, verbose=True):
    Model = base.model_class('unsw')
    cfg = base.config('unsw', x_train.shape[1], seed)
    cfg['prototype_num'] = BASIS
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    model = Model(cfg).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=GAMMA)
    xt = torch.from_numpy(x_train)
    n = len(xt)
    model.train()
    for epoch in range(EPOCHS):
        tot = 0.0
        for i in range(0, n, BATCH):
            xb = xt[i:i + BATCH]
            s = model(xb)
            loss = s.mean()
            h = model.encoder(xb)
            loss = loss + INFO_RATIO * (1 - F.cosine_similarity(model.decoder(h), xb, dim=-1)).mean()
            h_ = model.phi(xb)
            sel = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
            hs = h_[sel]
            matrix = hs @ hs.T
            mol = (torch.sqrt(torch.sum(hs ** 2, dim=-1, keepdim=True))
                   @ torch.sqrt(torch.sum(hs.T ** 2, dim=0, keepdim=True)))
            matrix = matrix / mol
            eye = torch.eye(hs.shape[0])
            loss = loss + CL_RATIO * ((1 - eye) * matrix).sum() / hs.shape[0] / hs.shape[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at epoch {epoch}')
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sch.step()
        if verbose and (epoch % 50 == 0 or epoch == EPOCHS - 1):
            print(f'  epoch {epoch} loss={tot / math.ceil(n / BATCH):.4f}', flush=True)
    model.eval()
    return model


def native_scores(model, x):
    out = []
    with torch.no_grad():
        xt = torch.from_numpy(x)
        for i in range(0, len(xt), BATCH_SCORE):
            xb = xt[i:i + BATCH_SCORE]
            h = model.encoder(xb)
            out.append((h - model.phi(xb) @ model.prototype).square().mean(-1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'freeze', 'summary'])
    ap.add_argument('--arm', default='drl', choices=['raw', 'drl'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    if a.cmd == 'freeze':
        out = {
            'script_sha256': sha(__file__),
            'data': str(DATA), 'data_sha256': sha(DATA),
            'data_protocol': str(ROOT / 'protocol_cicids.json'),
            'arm': 'raw = kNN on preprocessed features; drl = frozen harness (report 07 base loss) then kNN on 128-d latent',
            'model': {'class': 'official PN', 'prototype_num': BASIS, 'z_dim': 128,
                      'plearn': False, 'basis': 'fixed orthogonal random', 'diversity': True},
            'optimisation': {'optimizer': 'Adam', 'lr': LR, 'weight_decay': 1e-5,
                             'scheduler': f'ExponentialLR gamma={GAMMA}', 'epochs': EPOCHS,
                             'batch': BATCH, 'shuffle': False},
            'readouts': ['knn_k1', 'knn_k5', 'native_mean_residual'],
            'metric': 'per-class ROC vs test normals, TPR at FPR<=0.01, macro over attack classes',
        }
        (ROOT / 'protocol_cicids_run.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    x_train, x_cal, x_test, y, classes = load()
    idx = split_by_class(y, classes)
    norm = idx[classes.index('BENIGN')]
    rows = []
    perclass_out = {}

    if a.arm == 'raw':
        ht, xq = x_train, x_test
    else:
        model = train_drl(x_train, a.seed)
        ht = embed(model, x_train)
        xq = embed(model, x_test)
        nt = native_scores(model, x_test)
        macro, auroc, per = macro_tpr01(
            nt[norm], {c: nt[idx[i]] for i, c in enumerate(classes) if c != 'BENIGN'}, classes)
        rows.append(dict(arm='drl', seed=a.seed, readout='native_mean_residual',
                         macro_tpr01=macro, macro_auroc=auroc, n_features=ht.shape[1]))
        perclass_out['native_mean_residual'] = per

    for k in (1, 5):
        sc = knn_scores(ht, None, xq, k=k)
        macro, auroc, per = macro_tpr01(sc[norm],
                                        {c: sc[idx[i]] for i, c in enumerate(classes) if c != 'BENIGN'}, classes)
        rows.append(dict(arm=a.arm, seed=a.seed, readout=f'knn_k{k}',
                         macro_tpr01=macro, macro_auroc=auroc, n_features=ht.shape[1]))
        perclass_out[f'knn_k{k}'] = per

    import csv
    f = RES / 'cicids_runs.csv'
    new = not f.exists()
    with f.open('a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    (RES / f'perclass_{a.arm}_s{a.seed}{a.tag}.json').write_text(
        json.dumps(perclass_out, indent=2, ensure_ascii=False))
    for r in rows:
        print(f"{r['arm']:>3s} seed={r['seed']} {r['readout']:20s} "
              f"macro_tpr01={r['macro_tpr01']:.4f} auroc={r['macro_auroc']:.4f}", flush=True)


if __name__ == '__main__':
    main()
