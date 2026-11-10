# Results

Every file here is produced by a script under `code/`, or is an input that the
table-rebuild script reads.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `table2_supervised.csv` | `code/tables/export_tables.py` | manuscript Table 6 |
| `table4_recalibration.csv` | `code/tables/export_tables.py` | manuscript Table 4 |
| `table9_contrastive_5seed.csv` | `code/tables/export_tables.py` | §5.5 five-seed statistics |
| `table14_alpha_sweep.csv` | `code/tables/export_tables.py` | manuscript Table 16 |
| `report_10_neighbour.csv` | five-seed run log, transcribed | `export_tables.py` (input) |
| `paired_2025_unsw_minmax.csv` | `code/scoring/paired_2025.py` | manuscript Table 5 |
| `refsize_{ton,unsw,cic}.csv` | `code/scoring/refsize_ablation.py` | manuscript Tables 12–13 |
| `calib_stress.csv` | `code/scoring/calib_stress.py` | calibration stress test |
| `oracle_contam.csv` | `code/scoring/calib_stress.py` | closed-form contamination check |
| `dup_audit.csv` | `code/scoring/dup_audit.py` | duplicate audit of §5.9 |
| `indep_splits.csv` | `code/scoring/indep_splits.py` | independent re-partition check |
| `bootstrap_ci.csv` | `code/scoring/bootstrap_ci.py` | stratified bootstrap intervals |
| `baselines_{ton,unsw,cic}.json` | `code/scoring/baselines.py` | `export_tables.py` (input) |
| `protocol_{ton,unsw,cic}.json` | run configuration | seeds, splits and environment |
| `scores_{ton,unsw,cic}_s42.npz` | `code/scoring/dump_scores.py` | every statistic above |
| `tables/` | `export_tables.py` (output copy) | byte-identical reruns |

See `MAPPING.md` for the manuscript table numbering.
