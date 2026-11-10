"""CIC-IDS2017 paired scoring with the ICLR 2025 conference implementation.

UNSW already had 2025 checkpoints; CIC did not, so the 2025 model is retrained
here on the same training normals with the same hyper-parameters as the ToN
2025 baseline (200 epochs, lr .05, gamma .98, batch 512, 5 basis vectors,
softmax weights). The training loop mirrors tail_objective.train_model with
lam_tail = lam_comp = 0.
"""
import os

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

TRAE = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TRAE / 'code' / 'prepare'))
sys.path.insert(0, str(TRAE / 'code' / 'scoring'))
sys.path.insert(0, str(BASE / '实验产物' / 'DRL_基线对照_20261106' / 'source_snapshot' / 'DRL2025_compat'))
import run_cicids as cic  # noqa: E402
from baselines import macro_metrics  # noqa: E402
from DRL_Model.Model import DRL  # noqa: E402

RES = TRAE / 'results'
NORMAL = {'normal', 'Normal', 'BENIGN'}
EPOCHS, LR, GAMMA, BATCH, HIDDEN, BASIS = 200, .05, .98, 512, 128, 5
INFO_RATIO, CL_RATIO = .10, .06


def split(y):
    rng = np.random.RandomState(42)
    tr, te = [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        rng.shuffle(ix)
        h = len(ix) // 2
        tr.append(ix[:h])
        te.append(ix[h:])
    return np.concatenate(tr), np.concatenate(te)


def knn_scores(ref, points):
    nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='sqeuclidean',
                          n_jobs=-1).fit(ref)
    return np.sqrt(np.maximum(nn.kneighbors(points)[0][:, 0], 0))


def evaluate(s, cal, y, cats):
    agg, _ = macro_metrics(y, cats, s)
    cut = float(np.quantile(cal, .99, method='higher'))
    pred = s > cut
    rec = [float(pred[cats == c].mean()) for c in sorted(set(cats.tolist()) - NORMAL)]
    return agg['macro_tpr_1'], agg['macro_auroc'], float(np.mean(rec)), float(pred[y == 0].mean())


def train(x, seed):
    cfg = {'data_dim': x.shape[1], 'hidden_dim': HIDDEN, 'basis_vector_num': BASIS,
           'en_nlayers': 3, 'de_nlayers': 3, 'diversity': True, 'plearn': False}
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    np.random.seed(seed)
    model = DRL(cfg).to('cpu')
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=GAMMA)
    xt = torch.from_numpy(x)
    model.train()
    for _ in range(EPOCHS):
        for i in range(0, len(xt), BATCH):
            xb = xt[i:i + BATCH]
            s = model(xb)
            loss = s.mean()
            h = model.encoder(xb)
            x_tilde = model.decoder(h)
            loss = loss + INFO_RATIO * (1 - F.cosine_similarity(x_tilde, xb, dim=-1)).mean()
            hs = F.softmax(model.phi(xb), dim=1)
            sel = np.random.choice(hs.shape[0], int(hs.shape[0] * 0.8), replace=False)
            hs = hs[sel]
            matrix = hs @ hs.T
            mol = torch.sqrt(torch.sum(hs ** 2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(hs.T ** 2, dim=0, keepdim=True))
            eye = torch.eye(hs.shape[0])
            loss = loss + CL_RATIO * ((1 - eye) * (matrix / mol)).sum() / hs.shape[0] / hs.shape[0]
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
    model.eval()
    return model


def main():
    x, c, d, y, classes = cic.load()
    cats = np.asarray(classes, dtype=str)[y] if y.dtype.kind in 'iu' else np.asarray(classes, dtype=str)
    y = y.astype(np.int64)
    _, ite = split(y)
    raw_c, raw_q = knn_scores(x, c), knn_scores(x, d)
    rows = []
    for seed in (42, 43, 44):
        model = train(x, seed)
        with torch.no_grad():
            ht = model.encoder(torch.from_numpy(x).float()).numpy()
            hc = model.encoder(torch.from_numpy(c).float()).numpy()
            hq = model.encoder(torch.from_numpy(d).float()).numpy()
            rc = model(torch.from_numpy(c).float()).numpy().ravel()
            rq = model(torch.from_numpy(d).float()).numpy().ravel()
        lat_c, lat_q = knn_scores(ht, hc), knn_scores(ht, hq)
        for arm, (sc, sq) in {'native': (rc, rq), 'raw1nn': (raw_c, raw_q),
                              'lat1nn': (lat_c, lat_q)}.items():
            t1, au, r99, fpr = evaluate(sq[ite], sc, y[ite], cats[ite])
            rows.append((seed, arm, t1, au, r99, fpr))
            print(f'cic2025 s{seed} {arm:7s} tpr1={t1:.4f} auroc={au:.4f} '
                  f'recall99={r99:.4f} fpr={fpr:.4f}', flush=True)
    out = RES / 'paired_2025_cic.csv'
    with out.open('w', encoding='utf-8') as f:
        f.write('seed,arm,macro_tpr1,macro_auroc,recall99,fpr\n')
        for r in rows:
            f.write(','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in r) + '\n')
    a = np.array([r[2] for r in rows if r[1] == 'lat1nn'])
    b = np.array([r[2] for r in rows if r[1] == 'native'])
    print(f'lat1nn mean={a.mean():.4f} native mean={b.mean():.4f} '
          f'delta range={(a - b).min() * 100:.1f}..{(a - b).max() * 100:.1f}pp')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
