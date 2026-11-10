# Code

| Folder | Contents | Runs as shipped? |
| --- | --- | --- |
| `tables/` | `export_tables.py` rebuilds the four table files and the five-seed statistics from `results/` only; `verify_cic_scores.py` re-derives the CIC-IDS2017 macro average from the score dump | yes, `verify_cic_scores.py` needs `cache/` |
| `figures/` | `paper_figs.py` redraws all six figures from `results/` into `figures/` | yes |
| `scoring/` | one script per experiment: NR-T scoring, recalibration, the baselines, reference-set-size ablation, duplicate audit, independent re-partition, bootstrap, calibration stress test | partly, see below |
| `training/` | the tail-objective training module the checkpoints were produced with | needs datasets and checkpoints |
| `prepare/` | CIC-IDS2017 per-day cleaning and split construction | needs the raw capture |
| `readout/` | encoder readout used to write the latent caches | needs checkpoints |

## Entry points

```bash
python code/tables/export_tables.py     # four table files, byte-identical across runs
python code/figures/paper_figs.py      # six figures into figures/
```

Both read only `results/`. `tables/verify_cic_scores.py`,
`scoring/refsize_ablation.py` and `scoring/dup_audit.py` additionally need the
latent caches described in `../cache/README.md`.

## Scripts that need the full study tree

`nr_seeds.py`, `refsize_latents.py`, `dump_scores.py`, `paired_2025*.py`,
`recal_2025.py`, `unsw_fix_2025.py`, `variants_*.py`, `baselines_dsvdd_nobias.py`,
`tail_objective.py` and everything under `prepare/` load a checkpoint or a raw
dataset, so they need the study tree that produced the published numbers. They are
released for transparency, not as entry points.

That tree is located through one environment variable, `NRT_STUDY_ROOT`, which
defaults to the repository root. The layout the scripts expect is

```
$NRT_STUDY_ROOT/
├─ artifacts/
│  ├─ current/                 study working tree
│  │  ├─ tail_objective/       training module, checkpoints, density head
│  │  ├─ cic/                  CIC-IDS2017 capture, per-day splits, runs
│  │  └─ density_head/
│  ├─ drl_unsw/                UNSW-NB15 prepared arrays and runs
│  ├─ drl_baseline/            2025 baseline runs and compatible source snapshot
│  └─ ton/                     ToN-IoT prepared arrays and runs
└─ source/
   └─ datasets/UNSW-NB15/      UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
```

Set it before running those scripts, for example

```bash
NRT_STUDY_ROOT=/path/to/study/tree python code/scoring/dump_scores.py
```

## Paths

Every script resolves its own location with
`Path(__file__).resolve().parents[2]`, which is the repository root, so the tree
can be cloned anywhere.
