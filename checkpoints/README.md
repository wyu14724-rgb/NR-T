# Frozen checkpoints

The method studied in the manuscript is defined **on top of an unchanged
detector**. The detector is therefore an input, not a contribution, and its
weights are not redistributed here: they are the authors' reproduction of the
DRL 2025 prototype-based detector (ICLR 2025), trained with the seeds recorded in
`results/protocol_{ton,unsw,cic}.json`.

## What the scripts actually load

| Dataset | Detector source | Stored files |
| --- | --- | --- |
| ToN-IoT | rebuilt at run time from `data/raw/minmax.npz` by `code/training/tail_objective.py` at seed 42 | none |
| CIC-IDS2017 | rebuilt at run time by the scoring scripts through `code/prepare/run_cicids.py` | none |
| UNSW-NB15 | loaded from disk, `final_minmax_seed{42,43,44}/model.pt` | 5 files, hashed below |

So only UNSW-NB15 has checkpoint files to verify. Place them as

```
checkpoints/unsw/final_minmax_seed42/model.pt
checkpoints/unsw/final_minmax_seed43/model.pt
checkpoints/unsw/final_minmax_seed44/model.pt
checkpoints/unsw/screen_minmax_seed42/model.pt
checkpoints/unsw/screen_standard_seed42/model.pt
```

and run

```bash
python tools/make_manifest.py --check checkpoints/
```

## What each file is

- `final_minmax_seed4x` — the three seeds behind Table 4, Table 6 and the paired
  comparison (Fig. 2). Min-max preprocessing, prototype count `K = 5`.
- `screen_minmax_seed42`, `screen_standard_seed42` — the two preprocessing
  screens used for the preprocessing-switch ablation (§5.7).

Each checkpoint exposes the encoder output `h(x)` (128 dimensions here) and the
prototype basis `B` in the orientation used by the manuscript: `B in R^{K x d}`,
combination point `B^T w(x)`.

## Why they are not included

The upstream code and weights carry the original authors' licence, which does not
permit us to redistribute the weights. What we can and do release is everything
downstream of them: the latent-code caches, the score dumps and every derived
number in the paper.
