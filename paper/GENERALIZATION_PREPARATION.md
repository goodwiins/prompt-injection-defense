# Independent generalization evaluation — preparation

**Status: prepared for human review; evaluation has not started.** No new detector metrics exist. The previous A100 results remain unchanged. No GPU was allocated, no model was trained, and no threshold was changed.

The user approved freezing the three models and testing a new, human-reviewed holdout from unseen website/template families. The provenance audit found that the available data cannot currently establish that stronger form of independence.

## Confirmed findings

- All **448 training/validation records** lack trusted website and template IDs. None contains a canonical URL or `og:url`. The per-page audit, including IDs, raw hashes and available titles, is in `training_provenance.jsonl`.
- Missing metadata is not proof of overlap. Conversely, a different dataset name, different title, absent exact duplicates or low lexical similarity cannot establish unseen website/template families.
- Three fine-tuned models and their three matching frozen-encoder heads are fixed at their existing validation selections. Local package/checkpoint/configuration hashes passed. Freeze time: **2026-09-11T19:42:17.245121+00:00**. The historical XGBoost classifier and its existing threshold are retained as context.
- Base MPNet remains pinned to revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Its upstream `model.safetensors` LFS SHA256 is `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`. The base weights are not locally cached; verify downloaded bytes against this hash before any future evaluation. No new weights are needed now.

## Concrete review packet

[Open the blinded review packet](../results/generalization_20260911/REVIEW.md), then enter actual human labels in [human_review.csv](../results/generalization_20260911/human_review.csv).

The packet contains **six unscored raw pages in three matched pairs**, from the upstream website groups `kohls`, `sports.yahoo`, and `travelzoo`. It is a small review-workflow sample, **not the final benchmark** and not sufficient for a generalization claim. Each pair contains an original page and a locally constructed version with one published instruction inserted. Provisional labels are by construction; final labels are blank. AI review has not been represented as human annotation.

Pages come from the smallest released [Mind2Web shard](https://huggingface.co/datasets/osunlp/Mind2Web/tree/17ece8eb89862368edc0cc806acee6fca5163474/data/train), under CC-BY-4.0. For each of its three website groups, task choice is deterministic by SHA256, using the first action's complete raw HTML. Mind2Web's crowdsourced navigation annotations do **not** provide human prompt-injection labels. Its source `train` split is external to this detector's training data; that alone does not prove website-family separation.

The inserted instruction comes from a pinned [BIPIA template file](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/text_attack_test.json), under MIT. One template each uses task automation, translation and answer reversal. Only escaped text inside an `aside` is inserted; removing that exact span recovers the original page bytes. Nothing is executed or sent to a live website. These are our constructed combinations, not an official BIPIA benchmark run.

The three original parent pages passed normalized raw/extracted/first-window exact screening and character-TF-IDF cosine <0.90 against **8,370 prior inputs**, including earlier consumed holdouts solely for overlap rejection. Nearest-parent similarities are **0.238–0.284**. The inserted phrases have no exact normalized matches in the 448 development inputs. Neither check establishes semantic or template-family novelty. The parent-page overlap record is shared by its constructed pair; it is not a separate similarity measurement of the inserted page.

Candidate lengths are **57,468–189,369 tokens**. The full raw inputs are saved as inert `.txt` files, without truncation. Reviewers must inspect complete context; human review of a short excerpt cannot silently stand in for page-level review. Original/inserted pairs and pages from the same website must not be counted as independent observations.

## What is required before scoring

1. A real human reviewer must record labels (`injection`, `non-injection`, or `uncertain`), identity, dated attestation, supporting evidence and rationale. The assistant leaves these fields blank. Ambiguity is recorded rather than guessed. The user has been asked who will review.
2. Resolve the missing source/template provenance. The user has been asked whether to retain the strict unseen-family requirement or authorize a narrower external-dataset evaluation that explicitly reports unknown family overlap. Human labels alone do not resolve provenance.
3. Freeze a broader final sampling frame and final reviewed labels before scoring. Retain all three seeds and validation thresholds. Report paired fine-tuned versus matched frozen-head results with recall, FPR, AUC and uncertainty that accounts for available website/page groups. Do not choose a winning seed from test results.

The current preflight deliberately returns exit **2**, with two unmet requirements: six incomplete human reviews and unverified unseen-family provenance. This is an expected readiness result, not a failed training run. No label or provenance flag is set to complete merely because preparation finished.

## Reproduce and inspect

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_generalization_review.py self-check
HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_generalization_review.py check
```

- `results/generalization_20260911/models_frozen.json`: existing model/threshold choices and file hashes.
- `preparation.json`: immutable candidate inputs, provenance counts, source checksums and sampling limits.
- `preflight.json`: current unmet requirements and hash-check result.
- `training_provenance.jsonl`: all 448 development records, with missing metadata explicit.
- `template_audit.json`: literal-template occurrence screen, with limited scope.
- `candidates.jsonl`, `pages/`, `human_review.csv`, `REVIEW.md`: human review inputs; no predictions.
- `construction_reference.jsonl`: construction details; keep separate from the initial independent label pass.

The self-check passed raw-byte preservation, escaped insertion, incomplete-review and missing-family blocking, and duplicate-ID rejection. `git diff --check` passed; original staging SHA256 remains `404de3b634a62eb9fceb16b48da2a054c4ae3edcbb1a5b665c2affbfb40bcd46`. No commit, push, publication or deployment was performed. The completed-experiment monitor remains paused.
