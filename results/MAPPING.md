# Manuscript table to file mapping

The `table*_*.csv` names follow the internal numbering used while the study was
being run; they do not always match the numbering of the submitted manuscript.
This table is the authoritative mapping.

| Manuscript item | Backing file(s) | Produced by |
| --- | --- | --- |
| Table 4 (three scoring arms, recalibrated) | `table4_recalibration.csv` | `code/tables/export_tables.py` |
| Table 5 (paired three-arm comparison, UNSW-NB15) | `paired_2025_unsw_minmax.csv` | `code/scoring/paired_2025.py` |
| Table 6 (supervised upper bound) | `table2_supervised.csv` | `code/tables/export_tables.py` |
| Tables 12–13 (reference-set-size ablation) | `refsize_{ton,unsw,cic}.csv` | `code/scoring/refsize_ablation.py` |
| Table 16 (false-positive-budget sweep) | `table14_alpha_sweep.csv` | `code/tables/export_tables.py` |
| §5.5 five-seed statistics | `table9_contrastive_5seed.csv`, from `report_10_neighbour.csv` | `code/tables/export_tables.py` |
| §5.8 duplicate audit | `dup_audit.csv` | `code/scoring/dup_audit.py` |
| §5.8 independent re-partition | `indep_splits.csv` | `code/scoring/indep_splits.py` |
| Calibration stress test | `calib_stress.csv`, `oracle_contam.csv` | `code/scoring/calib_stress.py` |
| Bootstrap intervals | `bootstrap_ci.csv` | `code/scoring/bootstrap_ci.py` |
| Per-sample scores (all statistics) | `scores_{ton,unsw,cic}_s42.npz` | `code/scoring/dump_scores.py` |

Tables not listed here (dataset composition, hyper-parameters, feature counts and
the like) are descriptive and are not generated from an artefact.

Two consecutive runs of `export_tables.py` produce byte-identical files; the
SHA-256 of each is recorded in `../MANIFEST.sha256`.
