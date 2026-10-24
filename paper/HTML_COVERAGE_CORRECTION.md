# HTML coverage correction: trained and verified; weak new-page detection

The A100 experiment completed and all artifacts verified. On the five new AI-reviewed HTML attacks, fine-tuned seeds 17/42/73 detected **0/5, 1/5 and 1/5**. Both detections were the same Uniqlo typo instruction. Seed 42 also flagged its quoted counterpart; seed 73 did not. All three fine-tuned models left the five original pages unflagged. The other four direct instructions were missed by every fine-tuned model. This correction does not establish reliable new-page detection or readiness for a standalone security gate.

Overall validation combines 192 legacy pages and 15 new HTML pages (101 attacks, 106 negatives). These data selected checkpoints and thresholds, so the numbers below are development results, not fresh-test performance. The 15 new pages form five dependent original/direct/quoted triples.

| Arm | Seed | Epoch | Overall AUC | Overall recall | Overall FPR | New attacks detected | Original pages flagged | Quoted examples flagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 17 | 2 | 0.5825 | 9.90% | 4.72% | 0/5 | 0/5 | 0/5 |
| finetuned | 17 | 5 | 0.8251 | 69.31% | 4.72% | 0/5 | 0/5 | 0/5 |
| frozen | 42 | 3 | 0.5570 | 6.93% | 4.72% | 0/5 | 0/5 | 0/5 |
| finetuned | 42 | 6 | 0.8506 | 70.30% | 2.83% | 1/5 | 0/5 | 1/5 |
| frozen | 73 | 6 | 0.6100 | 9.90% | 4.72% | 0/5 | 0/5 | 1/5 |
| finetuned | 73 | 6 | 0.8577 | 69.31% | 3.77% | 1/5 | 0/5 | 0/5 |

Fine-tuning outperformed the matched frozen-encoder controls on aggregate validation. That comparison does not isolate the effect of the 20-row correction versus the previous training data: this run also added validation examples, changing checkpoint and threshold selection. Previous checkpoints were not evaluated on this identical augmented validation set. No causal before/after improvement from the data correction is claimed.

Every raw input token was processed. For the new direct validation attacks, the winning window contained the complete insertion in 0/5, 1/5 and 1/5 cases for fine-tuned seeds 17/42/73. Only the Uniqlo insertion won for seeds 42/73. Truncation therefore does not explain these misses. During training, a window containing the full insertion won in 5/60, 6/60 and 0/60 repeated observations of the ten new training attacks. This supports investigating which windows receive supervision; it does not isolate a causal explanation for the validation errors.

The correction retained 256 balanced training pages, replacing 20 legacy rows with 10 direct attacks, 5 originals and 5 quoted negatives. Ten other training-group counterparts remained review-only. Architecture, optimizer, six epochs and seeds stayed fixed. All six trials completed 384 finite optimizer updates. The existing bounded AMP policy retried 1/3/2 batches for fine-tuned seeds 17/42/73 (2/3/2 overflow attempts); no training batch was omitted.

Labels came from Codex review by the same assistant that constructed the packet. They are not human or blind independent annotations; negative labels remain provisional. All 45 labels matched the proposed construction labels. New-cohort websites and published categories are split-disjoint, but historical family overlap is unknown. Quote framing can be a shortcut, and validation category is coupled with insertion position. No final-test examples were exported or scored. The six metadata-only reserved websites remain untouched.

Verification at 2026-09-11T22:53:56.362042+00:00: 17 frozen input hashes, 143 artifact hashes, 41,432 token-window hashes, six validation selections and training orders, and all 384 updates per trial passed. All three portable packages reproduced 207 decisions each on A100 (621 total). Maximum local saved-feature/head score error was 1.19e-07. The four-cell executed notebook contains no errors. Full GPU encoder computation was not independently repeated locally.

The A100 session was terminated at 2026-09-11T22:55:00.955019+00:00; the backend confirmed 0 remaining assignments. The progress heartbeat is paused. Three new full weights and three matched frozen heads are saved, alongside the previous research baseline. No commit, push, publication or deployment occurred.

[Weight packages](../results/html_coverage_correction_20260911/completed/mpnet_html_coverage_ai/INFERENCE.md) · [Verification](../results/html_coverage_correction_20260911/completed/mpnet_html_coverage_ai/verification.json) · [AI review](../results/html_coverage_correction_20260911/AI_REVIEW.md) · [Executed notebook](../notebooks/colab_html_coverage_ai_output.ipynb)

Run verification again with:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_html_coverage_correction.py verify --output results/html_coverage_correction_20260911/completed/mpnet_html_coverage_ai
```

---

Historical preparation notes follow; the later AI authorization superseded the human-review prerequisite for this exploratory run.

# HTML coverage correction: prepared, awaiting human labels

The next correction changes data coverage while retaining the current MPNet architecture and frozen base-model controls. No training, scoring, threshold selection or new GPU allocation has occurred.

[Human review packet](../results/html_coverage_correction_20260911/REVIEW.md) · [Label sheet](../results/html_coverage_correction_20260911/human_review.csv)

## Prepared data

There are **45 candidate pages from 15 website groups**, organized as an original page, a direct instruction insertion, and a quoted version of that instruction for each parent. All labels are blank pending independent human review. The direct/quoted interpretations are construction hypotheses, not ground truth.

| Purpose | Website groups | Review candidates | Proposed fitting use |
|---|---:|---:|---|
| Training | 10 | 30 | 20: 10 direct, 5 originals, 5 quoted negatives; 10 counterparts retained for review only |
| Validation | 5 | 15 | All 15, subject to adjudicated labels |
| Future test reserve | 6 | 0 | Metadata only; no examples constructed or scored |

The fitting plan replaces 10 positive and 10 negative rows in the existing 256-page training set. Both the 236 retained rows and the 20 additions are balanced within their source, and the combined proposed counts remain **128 positive / 128 negative**. Exact retained, removed, added and review-only IDs are in `replacement_plan.json`. The same 256-page count preserves the existing optimizer-step budget. Validation would retain all 192 existing validation rows and add 15 new rows: 207 total, provisionally 101 positive / 106 negative. No labeled training/validation export has been issued.

The planned correction reuses the existing six-epoch settings, complete raw-HTML windows, masked mean pooling, linear head and maximum-window scoring. Keep the frozen encoder controls and report all seeds 17/42/73. Choose checkpoints and thresholds using validation only. Changes to validation composition and resulting thresholds are part of this development update; do not interpret later improvements as a pure architectural ablation.

## Checks and findings

- **Confirmed:** new parent pages were screened against 8,376 prior inputs, including the six consumed diagnostic pages. Exact normalized raw-text, extracted-text and first-window duplicates were excluded; character n-gram cosine at or above 0.90 was excluded both against prior inputs and already selected new parents.
- **Confirmed:** prospective training/validation websites and published template categories are disjoint. Every original/direct/quoted group stays within one assigned development split. The diagnostic websites are excluded; the six metadata-only reserved websites are also separate.
- **Confirmed:** complete inputs contain 25,986–129,401 tokens. Insertion at a parsed body/element boundary preserves the original exactly when removed. All 30 inserted spans fit completely within inference windows. No page truncation or new model predictions were used for selection.
- **High, unresolved:** all 45 human reviews remain missing. Reviewers must assess whole pages, supply evidence, and flag uncertainty; AI construction labels cannot provide human attestation. Uncertain or disagreeing labels require adjudication before the balanced replacement plan can be finalized.
- **High for generalization claims, unresolved:** older development inputs lack trusted website/template metadata. Retained lexical similarity reaches 0.883, below the exclusion threshold but still high. The screen and new website names do not prove independence from older data, semantic template separation or unseen-family generalization.
- **Known design limits:** the small packet does not fully cross each template category with every position. In validation, template category and position are coupled; position-specific conclusions would be confounded. Quotation framing may provide a shortcut. These are candidate data corrections, not evidence that an architecture has improved.

The original 18-group target yielded only 15 eligible groups. The packet was reduced without weakening length/duplicate criteria or consuming the six reserved groups. A subsequent pre-training amendment corrected the proposed 2:1 new-source class imbalance by selecting balanced fitting pairs; all 45 raw pages and blank review labels stayed unchanged. The superseded plan and protocol are preserved in `results/html_coverage_correction_20260911_preparation_v1/`; `amendment.json` explains the change.

## Sources and reproducibility

The [pinned Mind2Web dataset card](https://huggingface.co/datasets/osunlp/Mind2Web/blob/17ece8eb89862368edc0cc806acee6fca5163474/README.md) describes navigation tasks and raw HTML; it does not supply prompt-injection labels. This preparation uses released training shard 6, SHA256 `49b2764d8ce2d902448d024de0a7e3943169c3c4209d2fc6ddedcc353c4b3296`, from 100 tasks and 35 websites. The [pinned BIPIA training templates](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/text_attack_train.json) supply instruction strings. Their local combination is not an official BIPIA benchmark run. Source receipts, full raw candidates and exact construction spans are retained under `results/html_coverage_correction_20260911/`.

```bash
.venv/bin/python scripts/prepare_html_coverage_correction.py self-check
.venv/bin/python scripts/prepare_html_coverage_correction.py check
```

`check` currently returns **2** for incomplete review; frozen hash/group/structure checks pass. `verification.json` separately records independently recomputed token hashes and span coverage. No training is launched by either command. The old six-page diagnostic remains consumed evidence; its strict human-review/family preflight is unchanged. Future final evaluation requires its own settled labels, sampling frame and provenance, then a freeze before any scoring.
