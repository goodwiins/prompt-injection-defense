# Paper claim audit — September 10, 2026

**September 11 follow-up:** the bounded Colab experiment completed and downloaded evidence verification passed. See [COLAB_RUN.md](COLAB_RUN.md) for the three-variant comparison and limits. The active manuscript/PDF includes these new results; the audit below records the earlier historical-evidence assessment.

**Verdict: the historical results do not support the paper's generalization, mechanism, latency, or security conclusions.** The manuscript now distinguishes implementation facts from historical observations and unperformed experiments. This is an editorial and evidence repair, not a new model evaluation.

Audited checkout: `paper-fix-april-2026`, base `7b8af34dd7e5dfb90746e8183a8a19fa0ed7b901`, including the pre-existing April paper edits. Also inspected the later leakage patch `3d1bba8` and its merge `origin/main` at `1be03c1`. The patch changes source, not the saved December 2025 model/results. It has not been merged into this dirty checkout during this audit.

Scope: active `paper.tex`, its included tables/figure producers and bibliography; the two earlier Markdown manuscripts; the annotated historical copy; relevant encoder/classifier training, loaders, benchmark timing, model metadata, and saved results. No inference or training was run. Exact overlap sizes for remote datasets were not remeasured, and absence of an artifact in the inspected repository does not prove it never existed elsewhere.

## Findings and disposition

| Severity | Claim or issue | Evidence | Correction / remaining work |
|---|---|---|---|
| Critical | Near-perfect scores represent unseen-input performance | `training/finetune_mpnet_embeddings.py:70` loads benchmark sources; `:198` builds training pairs from all pools and `:210` validation pairs from their tails. The later evaluation reloads these sources at `:346`. | Withdraw held-out interpretation and headline superiority. Retrain both stages with independently verified text/group splits. |
| Critical | The classifier independently validates low over-defense | `training/train_injection_aware_classifier.py:157` loads the same local synthetic trigger set; `:377` calls `train`, whose implementation at `src/detection/embedding_classifier.py:487` fits every row. | All **247/247** local synthetic evaluation prompts occur in the generated classifier training pool. No independent over-defense result follows from 0/247 errors. |
| High | The merged leakage fix establishes a clean trained model | At `1be03c1`, `training/finetune_mpnet_embeddings.py:84` omits local synthetic NotInject from its holdout, and the separate classifier training script is unchanged. `scripts/verify_no_leak.py:64` subtracts evaluation text before intersecting the result with that same set at `:67`. | That intersection is empty by construction; it does not verify actual training membership or checkpoint lineage. Deduplicate before splitting and verify persisted training manifests independently. |
| High | Thousands of training rows imply thousands of distinct prompts | `training/train_injection_aware_classifier.py:131` repeats generated variations. Executing the repository's pure generation functions produced **2,667 rows / 160 normalized unique safe texts** and **570 rows / 502 unique trigger texts**. | Describe metadata as row counts. Quantify duplicates and split source/template groups before retraining. Duplicates can also invalidate nominal independence assumptions for intervals. |
| High | Latency CDF, boxplot, and P50/P95 are measured performance | `paper/generate_latency_charts.py:17` generates seeded lognormal samples; it reads no timing artifact. `benchmarks/runner.py:296` times a batch and `:325` divides its duration among items. | Remove the simulated plots and production-latency claims. Measure single-request latency separately from amortized batch throughput. |
| High | All earlier timers excluded embedding generation | The inspected runner calls `predict_proba`; `src/detection/embedding_classifier.py:414` calls `embed` inside that operation. The later branch's prose makes a broader assertion than this source supports. | Do not copy that assertion or the replacement CPU numbers without producer evidence. This audit establishes batch amortization, not universal XGBoost-only timing. |
| High | 97.3% overall accuracy is a reproducible aggregate | `results/latest_benchmark_results.txt` says 99.0%, while the count-bearing December record contains `(4081 + 590) / 4793 = 97.4546%`. The old table combines different slices/model generations and an unsupported TensorTrust row. | Generate a clearly labelled historical table from one count-bearing file. Display its pooled record separately; do not sum it with constituent slices or infer missing per-dataset counts. |
| High | Zero FPR on the official 339-item NotInject benchmark | `results/injection_aware_mpnet_evaluation.json:48` records 247 local synthetic examples; `benchmarks/benchmark_datasets.py:289` generates that set. The official loader is distinct at `:574`. | Name the local set explicitly. Remove the untraceable official result. The encoder historically also loads official NotInject for training, so a new run on that checkpoint would not resolve contamination. |
| High | IDR 2.31 → 0.67 proves semantic intent separation | No model-matched centroid/embedding calculation or artifact was located. These same values appear in the older BIT Markdown draft under different model names. | Withdraw the values and causal conclusion. Run uniform-vs-doubled-pair ablation with identical unique data and held-out predictions. A centroid ratio alone does not certify individual clustering. |
| High | TensorTrust achieves 83.2% | No supporting MPNet prediction/count record was located. The untracked evaluation script is not evidence of execution. Contrary to the old critique, the **current cache has 200 texts**, not zero, and the requested limit of 500 would not establish a 500-sample run. | Remove the row and limitation that cites this number. Use actual loaded counts if a model-bound evaluation is later run. Do not label the number fabricated based only on missing evidence. |
| High | BrowseSafe demonstrates cross-modality generalization | `benchmarks/benchmark_datasets.py:503` removes script/style and extracts text; `:535` applies this before classification. BrowseSafe is in the encoder training pool. | Recast as historical classification of extracted text. It is not rendered-browser, CSS, visual, held-out-domain, or downstream-action evidence. Withdraw conflicting 98.2/96.7 versus 98.0/94.1 claims from performance conclusions. |
| High | Authenticated OVON messaging / multi-agent protection is established | `src/coordination/messaging.py:22` uses an unkeyed hash; `:64` accepts the supplied `security_verified` flag when `source_tag_hash` is present, with no cryptographic verification. | Remove authenticated-metadata and empirical multi-agent guarantees. This is a separate runtime security defect; it is not fixed by editing the paper. |
| Medium | Threshold 0.764 was optimized on a validation split | `training/train_injection_aware_classifier.py:368` hard-codes it. Its `train` call has no split. The separate threshold-optimization result identifies the base MPNet model and recommends 0.85. | Describe 0.764 as the recorded operating threshold; remove the nonexistent threshold appendix and validation-recall criterion. |
| Medium | Class-ratio weighting and an 80/20 split were used | `models/injection_aware_mpnet_classifier_metadata.json:5` lists parameters without `scale_pos_weight`. The called training method fits all rows; a different method uses 90/10 at `src/detection/embedding_classifier.py:118`. | Correct the method description. Do not infer execution from an unused helper's capabilities. |
| Medium | All reported metrics have Wilson intervals and paired significance tests | The original included table had neither counts nor CIs. A McNemar mention is not a paired prediction result. | Add nominal Wilson intervals to count-backed accuracy/recall/FPR; use N/A for absent classes. Remove claims that a paired significance test was performed. These intervals cannot repair contamination. |
| Medium | Spearman 0.904 is unsupported, or proves a plateau-triggered early stop | `models/injection_aware_mpnet/README.md:511` records 0.9040 at step 12,500 / epoch 3.5714. The card records learning rate 5e-5 at `:373`; the script does not explicitly pass a rate or implement the claimed plateau stopping rule. | Retain the logged score as historical; correct both the earlier critique's missing-evidence statement and the unsupported stop explanation. Validation overlap still limits interpretation. |
| Medium | ROC/PR and qualitative examples establish perfect generalization | Saved ROC/PR JSON contains aggregate AUCs, not scores or a split manifest; qualitative text is truncated without a linked prediction record. | Exclude these assets from empirical claims until checkpoint, texts, split, and predictions are tied together. |
| Medium | 36 GCG/AutoDAN examples establish adaptive robustness of MPNet | `results/gcg_evaluation.json:58` identifies `models/bit_xgboost_model.json`. Its bypass counts are detector decisions, not downstream attack success. | Do not restore this section as an MPNet experiment. Explicitly distinguish detector bypass from agent ASR. |

Line references above are to unchanged files in this checkout unless an explicit remote commit is given. Original manuscript/table claims were captured before editing; rewritten manuscript line numbers are intentionally not used as evidence of the original text.

## Bibliography and external claims

Primary source pages were checked on September 10, 2026. The active bibliography now corrects:

- **PromptArmor:** first author is Tianneng Shi, not Shengyun Si; the full author list replaces the incorrect attribution. The source supports the qualitative preprocessing approach and its own sub-1% AgentDojo FPR/FNR result, not this repository's superiority or a generic 200 ms latency. [Primary paper](https://arxiv.org/abs/2507.15219).
- **InjecGuard:** authors are Hao Li and Xiaogeng Liu, not Liang; the 339-item official benchmark is separate from local templates. [Primary paper](https://arxiv.org/abs/2410.22770).
- **SecAlign:** the listed paper has six authors and does not include Bo Li. [Primary paper](https://arxiv.org/abs/2410.05451).
- **BrowseSafe:** use the named authors rather than a corporate author. Its existence does not substantiate our classifier's claimed cross-modality transfer. [Primary paper](https://arxiv.org/abs/2511.20597).

The active related-work summaries also match [StruQ](https://arxiv.org/abs/2402.06363), [Prompt Infection](https://arxiv.org/abs/2410.07283), [PeerGuard](https://arxiv.org/abs/2505.11642), [Ignore Previous Prompt](https://arxiv.org/abs/2211.09527), and [indirect prompt injection](https://arxiv.org/abs/2302.12173). The DefensiveToken numerical claim was removed: its previous citation points to an attack paper, not evidence for that defense. The StruQ percentage comparison was also removed because it cited JailbreakBench rather than an appropriate StruQ experiment. This is not an exhaustive literature review.

## Historical counts retained

Source: `results/injection_aware_mpnet_evaluation.json`, December 16, 2025, 03:24:39, model identifier `injection_aware_mpnet`, threshold 0.764.

| Record | n | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|
| deepset benign | 343 | 0 | 343 | 0 | 0 |
| deepset injections | 203 | 203 | 0 | 0 | 0 |
| Local synthetic triggers | 247 | 0 | 247 | 0 | 0 |
| Separately stored pooled record | 4,793 | 4,081 | 590 | 0 | 122 |

The first three slices total 793 rows; the pooled record also covers 4,000 SaTML/LLMail rows according to its sampling metadata. Individual confusion counts for those two datasets are absent. Do not reconstruct per-dataset recall from the combined false-negative count. No model hash, sample manifest, or post-fix provenance makes these fresh held-out results.

## Repairs delivered

- Rewrote the active manuscript around supported method details and an explicit results audit; removed the unsupported performance abstract/conclusion, simulated figures, IDR result, baseline ranking, qualitative confidence table, and contradictory dataset summary.
- Rebuilt `tables/per_dataset_metrics.tex` from validated confusion counts, with nominal 95% Wilson intervals and correct N/A handling. The generator rejects inconsistent/negative/noninteger counts, mismatched sample totals, nonfinite or inconsistent defined metrics, and unexpected dataset scope.
- Added `--check` so an overwritten/stale table fails verification instead of silently changing the paper's claims.
- Marked both older Markdown manuscripts as historical. `paper_annotated.tex` remains the historical reviewer copy, not a current manuscript. Unreferenced legacy tables and generators remain historical material; do not use them to repopulate current results.
- Corrected primary-source bibliography attributions and the stale handoff status. Preserved the user's pre-existing staging, model deletion, evaluation script, and result-file moves.

## Validation and limits

Run from the repository root:

```sh
python3 tests/test_paper_metrics.py
python3 paper/generate_per_dataset_metrics_table.py --check
git diff --check
mkdir -p /tmp/paper-claim-audit-build
tectonic --outdir /tmp/paper-claim-audit-build --keep-logs paper/paper.tex
```

The offline metric checks cover Wilson boundary values, absent classes, a real zero recall, pooled-count arithmetic, invalid evidence, and the historical provenance label. Dataset generation was checked by executing the existing pure loader/generator functions extracted through Python AST, avoiding their ML imports and training entry points. Normalization was whitespace collapse plus lowercase. That check does not load remote datasets or run a model.

Fresh training and inference remain unperformed. The current local encoder directory has no `model.safetensors` (its deletion was already staged before this task), and the inspected artifacts do not provide a post-fix checkpoint. Recovering an old checkpoint alone would not make the historical benchmark independent. The next experimental work is to complete a shared, deduplicated split for both training entry points, persist provenance, retrain, freeze the threshold on validation, and then regenerate results. Runtime authentication requires separate implementation and testing before any related paper claim is restored.

Validation outcome: both offline checks and `git diff --check` passed. Tectonic compiled the revised PDF with no undefined references/citations or overfull boxes; the remaining inputenc notice is the expected UTF-8-engine compatibility warning. The rendered results table was visually checked. `paper/paper.pdf` now matches the revised source. The pre-existing staged diff is byte-for-byte unchanged.
