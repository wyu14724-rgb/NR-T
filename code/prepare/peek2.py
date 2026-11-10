import pandas as pd
from pathlib import Path

p = Path(r'D:\实验\选题候选10.27\选题2\实验产物\TRAE接手\CICIDS2017_20261107\raw\CICIDS_Flow.parquet')
df = pd.read_parquet(p, columns=['Timestamp', 'attack_label', 'Flow Duration', 'Total Fwd Packets'])
print('rows', len(df))
print('\n=== label counts ===')
vc = df['attack_label'].value_counts()
print(vc.to_string())
print('\nn_classes', df['attack_label'].nunique())
ts = pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed')
print('\ntimestamp parse ok', ts.notna().sum(), 'of', len(ts))
print('min', ts.min(), 'max', ts.max())
print('\nby date:')
print(ts.dt.date.value_counts().sort_index().to_string())
