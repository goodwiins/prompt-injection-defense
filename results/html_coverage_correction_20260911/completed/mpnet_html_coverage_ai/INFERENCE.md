# AI-reviewed HTML correction weight packages

Three full MPNet encoder + classifier packages from the six-epoch A100 run. Each package contains checkpoint.pt, inference_config.json, mpnet_inference.py and inference_assets/. Keep them together. These are research classifiers; a negative decision does not establish safety or authorize tool access.

| Seed | Selected epoch | Package | Checkpoint SHA256 |
|---|---:|---|---|
| 17 | 5 | [finetuned_17](finetuned_17/INFERENCE.md) | `1840bd9d7b8ee128e1ef25bfb3f3ad35ca1d9cd10da7ebb1a9332aff86ab935f` |
| 42 | 6 | [finetuned_42](finetuned_42/INFERENCE.md) | `015595c507e551c8fc010dca9ec60df5c0077b7048ef18a36cd7844de4e33917` |
| 73 | 6 | [finetuned_73](finetuned_73/INFERENCE.md) | `e3cda270e20657e946feb85416594ff13b03a90bf2028c7d7779b5f5cfb67c5b` |

All three packages reproduced all 207 saved validation decisions on A100. No final test was evaluated. Matched frozen heads are in frozen_17/, frozen_42/ and frozen_73/. The previous baseline packages remain unchanged.

For a local float32 CPU prediction, from this directory:

```sh
python finetuned_17/mpnet_inference.py predict --package finetuned_17 --input /path/to/page.html --device cpu
```

CPU scores may differ slightly from the A100 float16 reference. The threshold and exact package hashes are stored in inference_config.json. Report all seeds; do not choose a seed from these validation results and call it independent evaluation.
