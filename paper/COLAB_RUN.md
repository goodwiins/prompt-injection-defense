# Colab audit experiment — 2026-09-10

Follow-up completed September 11: validation error analysis and a three-classifier-seed windowing experiment on 919 fresh BrowseSafe rows are recorded in [VALIDATION_DIAGNOSIS.md](VALIDATION_DIAGNOSIS.md). Windowing alone did not provide a useful improvement; the base remains a research reference.

Notebook: https://colab.research.google.com/drive/1hnI9NUTIdHjk1TBAG6TbeOitbRQYfFxh

Status: **all three variants completed; downloaded evidence verification passed on September 11, 2026.** The evidence and both trained encoders are retained under `results/colab_audit_20260911/`. All 53 recorded artifact checksums matched. Saved classifiers reproduced the saved test scores from stored embeddings; split isolation, token hashes, training-pair membership, validation-only thresholds, confusion counts, and latency percentiles passed independent checks. Encoder inference and training were not repeated locally.

| Variant | Accuracy | Attack recall | False-positive rate | Official NotInject FPR |
|---|---:|---:|---:|---:|
| Base MPNet + XGBoost | 69.03% | 45.55% | 11.28% | 29.50% |
| Uniform pairs + XGBoost | 62.55% | 48.30% | 25.50% | 57.82% |
| Doubled injection/trigger pairs + XGBoost | 63.67% | 64.20% | 36.78% | 83.78% |

The common test set has 2,315 inputs (1,056 attacks and 1,259 benign). Doubling the pair type improves attack recall by 15.91 percentage points over uniform pairing, but raises false positives by 11.28 points. Its official NotInject false positives are 284/339, versus 196/339 for uniform and 100/339 for base MPNet. These operating points do **not** support a general reduction in over-defense. Base MPNet has the highest pooled accuracy and lowest false-positive rate in this experiment. Thresholds were independently selected on validation: base 0.880826, uniform 0.772123, double 0.999148; this is not a comparison at matched test recall or FPR.

Measured single-input P50/P95 latencies on Tesla T4 with CPU XGBoost were 17.31/42.05 ms (base), 20.90/49.44 ms (uniform), and 17.24/45.18 ms (double), each from 150 inputs after 10 warmups. These include tokenization, embedding, and classifier inference, but exclude HTML preprocessing, network/queue time, and application work. They are not production-service latency.

Evidence archive SHA256: `f1c5c966e3cf747181c0121393683aad78178a216be48f8390702aeb948f91f1`. Verification summary: `results/colab_audit_20260911/verification.json`.

Re-run the check with:

```sh
.venv/bin/python scripts/verify_audited_run.py results/colab_audit_20260911
```


Source: `training/retrain_audited.py`, SHA256 `06c908c571486b62313af603406e2c3544184cc28f57e304e97d4491a06c9a07`. Portable notebook: `notebooks/colab_retrain_audited.ipynb`, built by `scripts/build_audited_colab.py`. The live notebook's preparation cell applies a hash-checked correction to the initially embedded script; the portable local notebook directly embeds the corrected source. Both execute the same script hash.

The actual source check loaded 6,370 nonempty rows and recorded one rejected empty LLMail submission. It retained 5,343 unique nonconflicting inputs after 1,027 exclusions: 2,286 train, 742 validation, 2,315 test. The source revisions, raw inputs, rejected rows, duplicate/label exclusions, normalized text hashes, 384-token input hashes, known team/template groups, and final split memberships are persisted. All 339 official NotInject prompts and all 247 historical local synthetic prompts are test-only. The 200-row TensorTrust cache contributes 121 retained unique test inputs; its upstream sampling provenance is unknown.

Both encoder fine-tuning and classifier fitting consume the same frozen training IDs. The classifier's threshold maximizes validation balanced accuracy (highest threshold breaks ties), then freezes before test inference. Base, uniform-pair, and doubled-pair variants use one seed; trained variants each use 14,000 pairs, four epochs, batch size four, explicit learning rate 5e-5, and 100 warmup steps. The run saves actual pairs, classifiers/checkpoints, validation and test scores, test embeddings, nominal Wilson intervals, synchronized single-input timing samples, and artifact checksums.

This is a capped exploratory experiment, not a complete reproduction or a guarantee against semantic near-duplicates. Known LLMail team groups and synthetic template families are isolated; unknown SaTML provenance and semantic similarity remain limitations. Per-source metrics combine official test rows and generated held-out groups; use `source_split` in the manifest to distinguish them. BrowseSafe remains extracted-text classification. No downstream attack-success test or runtime authentication fix is included.

The evidence archive and a separate encoder-checkpoint archive were downloaded; nothing was published to Hugging Face. The active paper now adds these bounded experimental results while retaining the withdrawal of unsupported historical claims. The old training entry points and old evaluation artifacts remain historical.

Offline checks passed: `tests/test_audited_retraining.py` (transitive group isolation, token-truncation duplicates, conflicting labels, pair membership/counts, validation threshold, absent-class metrics), notebook schema and Python-cell parsing, embedded-source equality, existing paper metric checks, and `git diff --check`. The user's pre-existing staged patch hash remained `404de3b634a62eb9fceb16b48da2a054c4ae3edcbb1a5b665c2affbfb40bcd46`.
