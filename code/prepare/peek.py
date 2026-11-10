import pandas as pd
from pathlib import Path

p = STUDY / 'artifacts' / 'current' / 'cic' / 'raw' / 'CICIDS_Flow.parquet'
df = pd.read_parquet(p)
print('shape', df.shape)
print('dtypes head')
print(df.dtypes.to_string()[:2000])
print('\ncolumns:')
print(list(df.columns))
for c in df.columns:
    if df[c].dtype == object or str(df[c].dtype) == 'category':
        print('\nvalue_counts', c, df[c].nunique())
        print(df[c].value_counts().head(20).to_string())
print('\nhead')
print(df.head(3).to_string()[:1500])
