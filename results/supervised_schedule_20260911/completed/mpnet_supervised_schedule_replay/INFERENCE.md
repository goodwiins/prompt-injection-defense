# Revised MPNet research weights

All three validation-selected epoch-six packages are retained. None was selected using holdout performance. Keep each checkpoint, inference_config.json, mpnet_inference.py and inference_assets/ together.

| Seed | Threshold | Checkpoint SHA256 |
|---|---:|---|
| [17](finetuned_17/INFERENCE.md) | 0.6299245953559875 | `f921300f51fe3436be4b64bf2d0ebca1683d096dfa187d6b312e58becdae2f54` |
| [42](finetuned_42/INFERENCE.md) | 0.6299117803573608 | `e2f987b76caabfab2e8429642f00bb66f78d68656871c96d217eedba3e6fe397` |
| [73](finetuned_73/INFERENCE.md) | 0.8691009283065796 | `76eb593feab6ed4b513a4664b8f35e998a2784a767e293cfdcb45b496ace3f26` |

Each checkpoint is 438,015,770 bytes. The package loads offline, processes all raw-HTML windows, and returns a detector score and decision. The weights contain no optimizer state.

From the repository root, for example:

```bash
python training/mpnet_inference.py predict \
  --package results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay/finetuned_17 \
  --input page.html --device cuda
```

Use `--device cpu` for float32 inference. GPU reference: A100, torch 2.9.1, transformers 4.57.3, sentence-transformers 5.1.2, numpy 2.3.5. CPU numerical equivalence is not guaranteed near the threshold.

All three packages passed 192-page A100 validation replay with identical decisions. This is research evidence, not security authorization. See [the report](../../../../paper/SUPERVISED_SCHEDULE.md).
