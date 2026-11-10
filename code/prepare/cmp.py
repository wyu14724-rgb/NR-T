import json
from pathlib import Path

r = Path(r'D:\实验\选题候选10.27\选题2\实验产物\TRAE接手\CICIDS2017_20261107\results')
a = json.loads((r / 'perclass_raw_s42.json').read_text())
b = json.loads((r / 'perclass_drl_s42.json').read_text())
print(f"{'class':26s}{'raw_k1':>8s}{'drl_k1':>8s}{'delta':>8s}{'drl_nat':>9s}{'raw_auc':>9s}{'drl_auc':>9s}{'n':>7s}")
for c in a['knn_k1']:
    ra, rb = a['knn_k1'][c], b['knn_k1'][c]
    print(f"{c:26s}{ra['tpr01']:8.3f}{rb['tpr01']:8.3f}"
          f"{rb['tpr01'] - ra['tpr01']:+8.3f}{b['native_mean_residual'][c]['tpr01']:9.3f}"
          f"{ra['auroc']:9.3f}{rb['auroc']:9.3f}{ra['n']:7d}")
