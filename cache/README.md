# Latent-code caches

The latent-code caches are the only large inputs the verification scripts need.
They are **not** committed to the repository: eleven files, about 584 MB in total,
of which CIC-IDS2017 accounts for roughly 340 MB.

Place them as

```
cache/ton_s42.npz        cache/ton_s43.npz        cache/ton_s44.npz
cache/unsw_s42.npz       cache/unsw_s43.npz       cache/unsw_s44.npz
cache/cic_s42.npz        cache/cic_s43.npz        cache/cic_s44.npz
cache/unsw_s42_2025.npz  cache/cic_s42_2025.npz
```

The two `*_2025.npz` files are the caches written with the 2025 checkpoints; they
reproduce the published scores exactly, while the plain `*_s42.npz` files carry
the superseded truncation for UNSW-NB15 and CIC-IDS2017 (see
`code/scoring/refsize_latents.py`).

## What they contain

For each dataset: the 128-dimensional encoder output `h(x)` for the training
normal samples (the reference set `H`) and for the evaluation samples, in the
order of the cleaned split. No labels and no attack categories are stored, which
is why `verify_cic_scores.py` reads the score dump for labels instead.

With these files present, `refsize_ablation.py`, `refsize_latents.py`,
`dup_audit.py` and `verify_cic_scores.py` reproduce the reference-set-size
ablation, the duplicate audit, the five-fold uniqueness figures of §5.9 and the
CIC macro average, without a checkpoint.

## Distribution

On acceptance the caches are deposited on Zenodo alongside this repository and
listed in `MANIFEST.sha256`. Until then they are available from the corresponding
author.
