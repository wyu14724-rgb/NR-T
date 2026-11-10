# Code

| Folder | Contents | Runs without checkpoints? |
| --- | --- | --- |
| `tables/` | `export_tables.py` rebuilds the four table files and the five-seed statistics from `results/` only; `verify_cic_scores.py` re-derives the CIC-IDS2017 macro average from the score dump | yes, but `verify_cic_scores.py` needs the caches in `cache/` |
| `figures/` | `paper_figs.py` redraws all six figures from `results/` into `figures/` | yes |
| `scoring/` | one script per experiment: NR-T scoring, target-domain recalibration, the eight baselines, the reference-set-size ablation, the duplicate audit, the independent re-partition check, the bootstrap and the calibration stress test | only the ones listed below |
| `training/` | the tail-objective training module the checkpoints were produced with, and its summariser | needs datasets + checkpoints |
| `prepare/` | CIC-IDS2017 per-day cleaning and split construction | needs the raw captures |
| `readout/` | encoder readout used to write the latent caches | needs checkpoints |

## Entry points that run from the repository as shipped

```bash
python code/tables/export_tables.py     # four table files, byte-identical across runs
python code/figures/paper_figs.py      # six figures into figures/
```

## Scripts that need more than this repository

`nr_seeds.py`, `refsize_latents.py`, `dump_scores.py`, `paired_2025*.py` begin by
importing the training modules and loading a checkpoint, so they need the full
study tree (`checkpoints/` plus the raw datasets) to run. They are included for
transparency: they document exactly how each number was produced, but they are
not entry points for a reader who only wants to check the tables.

`refsize_ablation.py` and `dup_audit.py` need the latent caches (`cache/README.md`).

## Paths

All scripts resolve their paths from their own location
(`Path(__file__).resolve().parents[2]` is the repository root), so the tree can be
cloned anywhere. The directory names used during the study map to this repository
as follows:

| Study directory | Repository path |
| --- | --- |
| `基线对照_20261107/results` | `results/` |
| `基线对照_20261107/code` | `code/scoring/` |
| `表格存档_20261110/results` | `results/tables/` |
| `表格存档_20261110/code` | `code/tables/` |
| `模态读出_20261107/cache` | `cache/` |
| `模态读出_20261107/code` | `code/readout/` |
| `训练目标_20261107/code` | `code/training/` |
| `CICIDS2017_20261107/code` | `code/prepare/` |
| `表示_20261107/实验报告_10_邻域目标.md` | `results/report_10_neighbour.md` |
