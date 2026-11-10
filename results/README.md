# results 目录口径说明

本目录混有不同时期的产物。**现行口径以 `scores_*_s42.npz` 为准**（一次性落盘的测试样本三分值 + 标签 + 类别名），
论文 §5.3 及配图 F2/F3/F6 均从该 npz 现算。以下几项已作废，仅留作早期对照。

## 已作废 / 仅部分有效

| 文件 | 状态 | 现行替代 |
|---|---|---|
| `paired_three_arms.csv` | 仅 `dataset == 'ton'` 的 9 行有效；`unsw` / `cic` 各 9 行为旧预处理（如 unsw native 0.102，现行 0.463） | `paired_2025_unsw_minmax.csv`、`paired_2025_cic.csv` |
| `perclass_*.json`（24 个） | 整套为旧口径（perclass_ton_NR 的 ddos 为 0.298，现行 0.49 级）；OCSVM/DeepSVDD/NR 三臂尤甚 | 由 `scores_*_s42.npz` 现算（见 `code/paper_figs.py` 的 `per_class_tpr`） |
| `calib_diag.json` 的 UNSW 条目 | 旧预处理 | `recal_2025_unsw_minmax.txt`（论文附录已注明） |

## 版本变更

- `oracle_contam.csv`（2026-11-09）：追加 `micro_recall`、`identity_fpr`、`diff_pp` 三列，用于逐格核对论文 §5.4 的解析式误差。
  原有 7 列数值不变（脚本种子固定，重跑逐位一致）。`diff_pp` 仅对 `scenario == 'oracle_clip'` 有意义。

## 注意

`paired_three_arms.csv` 若整表复算，会得到 unsw +38.8pp 这类错误结论。取数前先按 `dataset` 过滤。
