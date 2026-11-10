# NR-T: neighbourhood scoring on frozen deep-representation checkpoints

Code, results and verification scripts for the manuscript

> **From Subspace to Sample: Neighborhood Scoring for Low False-Positive Unsupervised Intrusion Detection**

NR-T keeps an existing prototype-based anomaly detector completely unchanged —
the encoder, the prototype basis and the network-predicted weights — and replaces
only the scoring object: instead of the residual to a predicted prototype
combination it uses the nearest-neighbour distance in the latent space of the
training-set normal samples.

## What this repository can and cannot reproduce

The scripts reproduce **every reported table value and derived statistic** from
the archived outputs, without training or loading a checkpoint. They do **not**
retrain the neural networks; re-training requires the checkpoints
(`checkpoints/README.md`) and the raw datasets (`data/README.md`).

This boundary is stated in Appendix A of the manuscript and is deliberate: the
paper's claim is about the scoring object, not about the representation.

## Layout

```
code/scoring/    NR-T scoring, calibration, baselines, ablations and statistics
code/tables/     table provenance (export_tables.py) and the CIC score re-derivation
code/figures/    paper_figs.py, which redraws all six figures from results/
results/         CSV/JSON/TXT/NPZ outputs of every script above
results/tables/  the four table files rebuilt by code/tables/export_tables.py
data/            how to obtain the raw datasets (not redistributed)
checkpoints/     how to obtain the frozen checkpoints (not redistributed)
cache/           latent-code caches (not included; see cache/README.md)
tools/           manifest generator
```

## Quick start

```bash
git clone https://github.com/wyu14724-rgb/NR-T.git
cd NR-T
python -m pip install -r requirements.txt

python reproduce.py          # rebuild the four table files and the six figures
```

`reproduce.py` runs `code/tables/export_tables.py` and `code/figures/paper_figs.py`.
The former reads only files under `results/`, and two consecutive runs produce
byte-identical output. `code/tables/verify_cic_scores.py` additionally needs the
latent-code caches (`cache/README.md`).

The six figures are shipped as vector PDFs in `figures/`, and `results/MAPPING.md`
maps every manuscript table to the file that backs it.

## Results files

| File | Content |
| --- | --- |
| `results/table2_supervised.csv` | supervised upper-bound reference |
| `results/table4_recalibration.csv` | three scoring arms after target-domain recalibration |
| `results/table9_contrastive_5seed.csv` | five-seed statistics |
| `results/table14_alpha_sweep.csv` | false-positive-budget sweep (18 cells, 36 values) |
| `results/paired_2025_unsw_minmax.csv` | paired three-arm comparison on UNSW-NB15 |
| `results/refsize_{ton,unsw,cic}.csv` | reference-set-size ablation |
| `results/calib_stress.csv`, `results/oracle_contam.csv` | calibration stress test and its closed form |
| `results/dup_audit.csv` | duplicate audit on the latent codes |
| `results/indep_splits.csv` | independent re-partition re-verification |
| `results/bootstrap_ci.csv` | stratified bootstrap intervals |
| `results/scores_*_s42.npz` | per-sample scores for all three datasets (seed 42) |

The `table*_*.csv` names follow the numbering used during the study and do not
always coincide with the table numbers in the final manuscript; Appendix A of
the manuscript records which file backs which table.

## Integrity

`MANIFEST.sha256` lists the SHA-256, size and modification time of every file in
this repository. Regenerate it with

```bash
python tools/make_manifest.py
```

## Citing

See `CITATION.cff`.

## Licence

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`DATA_LICENSE.md`).
The upstream datasets and checkpoints keep their own licences.
