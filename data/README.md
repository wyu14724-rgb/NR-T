# Raw datasets

The three benchmark datasets are **not** redistributed here. They are public and
should be obtained from their original providers.

| Dataset | Source | Used for |
| --- | --- | --- |
| ToN-IoT (network) | UNSW Canberra / Cyber Range release | tabular network-flow benchmark |
| UNSW-NB15 | UNSW Canberra, official train/test split | the dataset on which the three scoring arms are compared in full |
| CIC-IDS2017 | Canadian Institute for Cybersecurity (UNB); this study reads the per-day CSV capture, cleaned as described in §4 | large-scale, heavily duplicated benchmark |

The cleaned, model-ready files expected by `code/scoring/` are the per-day and
per-split artefacts whose names are recorded in the `*_meta.json` and
`baselines_*.json` files under `results/`. Column order and label encoding must
match the originals: the scripts index columns positionally after the cleaning
step described in the manuscript.

## Exact files used

Place them under `data/raw/` with these names; `data/SHA256SUMS` verifies them.

| Path | Origin | Size | SHA-256 |
| --- | --- | --- | --- |
| `raw/minmax.npz` | ToN-IoT, prepared min-max arrays | 0.55 MB | `cdbf07d2…3485c3` |
| `raw/data.npz` | UNSW-NB15, prepared arrays (official train/test split) | 6.21 MB | `179801ac…4d8560` |
| `raw/CICIDS_Flow.parquet` | CIC-IDS2017 flow capture, `rdpahalavan/CIC-IDS2017 Network-Flows` | 352.68 MB | `e64f6adf…0dda35` |

The first two are the cleaned, model-ready arrays the training code reads; the
third is the unmodified flow capture, cleaned by `code/prepare/prepare.py`.

## Verifying what you obtained

```bash
python tools/make_manifest.py --check data/
```

Every line of `data/SHA256SUMS` must report `ok`. If a file differs, the table
values will differ too.

## Why they are not included

Redistributing ToN-IoT, UNSW-NB15 or CIC-IDS2017 would duplicate multi-gigabyte
archives under licences we do not control. The derived quantities that the
manuscript actually reports — scores, latent codes and table files — are released
instead, which is what the claims rest on.
