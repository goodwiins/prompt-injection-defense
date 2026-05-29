# Fix Train/Eval Data Leakage, Retrain, Re-benchmark, Update Paper

## Context

The "Injection-Aware MPNet" detector in `prompt-injection-defense` reports near-perfect
results (SaTML 98.8%, deepset/LLMail 100%, NotInject-HF **0.0% FPR**, 97.3% overall).
Investigation traced this to **train/eval data leakage**:

- The producing script `training/finetune_mpnet_embeddings.py` builds its encoder
  fine-tuning pools from **deterministic first-N streaming pulls** of HuggingFace
  datasets (SaTML, deepset, LLMail, BrowseSafe, NotInject-HF).
- The benchmark (`benchmarks/run_benchmark.py --paper`) pulls the **same sources with the
  same first-N logic and no holdout/dedup**. Eval sets are therefore subsets of the
  encoder's training pool. For NotInject-HF the overlap is **100%** (`:100` loads all 339;
  `:348` evaluates the same 339) — this is why FPR is exactly 0.0%.
- A second internal leak: validation pairs are built from `pool[-200:]` of pools that
  remain fully in the training pairs (`finetune_mpnet_embeddings.py:209-216`).

The fine-tuned encoder — the paper's headline contribution — has seen the exact eval
prompts. All headline numbers are inflated. (Note: an earlier "0% NotInject overlap" check
was wrong; it compared a local synthetic file the script never uses, not the HF source.)

**Goal:** eliminate the leak, verify zero overlap, retrain, re-benchmark for honest numbers,
and update the paper (replace inflated numbers + add a Threats to Validity subsection).

User decision: **do both** (fix + re-run AND document). Fix method chosen below.

## Recommended approach: Approach (A) — holdout eval IDs at training time

Compute the exact evaluated text set (same loaders + same limits the benchmark uses),
normalize, and subtract those texts from every training pool before pair construction and
before XGBoost training. **Eval code is untouched**, so numbers stay directly comparable to
the original and any metric drop is attributable purely to leak removal.

Rejected: global disjoint 80/20 split (Approach B) — methodologically purer but touches all
seven benchmark loaders, re-pins every paper sample count, and would shrink the NotInject-HF
benchmark below the full official 339, weakening the central over-defense claim. Documented
as future work.

## Tasks

### 1. Add eval-holdout subtraction — `training/finetune_mpnet_embeddings.py`
- Extend loader imports (`:31-38`) to add `load_deepset_injections_only`, `load_tensortrust_dataset`.
- After `:67`, build `EVAL_HOLDOUT` = normalized texts from the loaders at the **exact eval
  limits**: satml=300, deepset_full=400, deepset_injections=203, notinject_hf=339, llmail=200,
  browsesafe=500, tensortrust=1000. Normalizer: `" ".join(t.split()).strip().lower()`.
- After each pool is built (`injection_samples` ~`:84`, `safe_samples` ~`:96`,
  `benign_trigger_samples` ~`:107`), filter out any sample whose normalized text is in
  `EVAL_HOLDOUT`. Add asserts that each pool keeps >100 samples.
- Effect: `benign_trigger_samples` loses all 339 NotInject-HF entries but keeps the synthetic
  NotInject pool, preserving benign-trigger contrastive signal.

### 2. Fix the internal `[-200:]` validation leak — same file `:198-216`, `:301-303`
- Shuffle each pool (seed 42) and carve a **disjoint** `(train_part, val_part)` tail (200, or
  10% if pool too small). Build train pairs from `*_train`, val pairs from `*_val`.
- Change XGBoost source arrays (`:301-303`) from the full pools to `inj_train/safe_train/bt_train`
  so the classifier is also clean of encoder-val and eval texts.

### 3. Verification script — new `scripts/verify_no_leak.py`
- Rebuild eval sets and training pools (with holdout) using the same normalizer; print
  per-dataset `leaked_into_train` counts and `assert` empty intersection.
- **Gate:** must print `PASS: zero overlap` before any retrain. CPU-only, no GPU.

### 4. Retrain encoder + XGBoost — `python training/finetune_mpnet_embeddings.py`
- **Runtime warning: CPU-only box, no GPU. Canonical run (14k pairs × 4 epochs, batch 4) is
  ~4-9h wall-clock** incl. first-time dataset downloads (satml/notinject_hf/browsesafe not cached).
- **Back up** current artifacts first (`models/injection_aware_mpnet/` →
  `..._PRELEAKFIX/`, both classifier JSONs) to quantify inflation later.
- Run in background and poll. Optional fast smoke run (EPOCHS=2, BATCH_SIZE=16, ~<1h) to
  confirm the pipeline before the canonical run; **paper numbers must come from the canonical
  4-epoch/batch-4 config** for comparability.
- Artifacts overwritten: `models/injection_aware_mpnet/` + `injection_aware_mpnet_classifier.json`
  (+ metadata; auto-reloaded by the benchmark via metadata `model_name`).

### 5. Re-run benchmark — leakage-free numbers
- `python -m benchmarks.run_benchmark --paper --model models/injection_aware_mpnet_classifier.json --threshold 0.764 --output-format json --output results/latest_benchmark.json`
- Second run with `--datasets satml deepset deepset_injections notinject notinject_hf llmail browsesafe tensortrust` → `results/latest_benchmark_full.json` for the full table rows.
- CPU benchmark ~5-15 min.

### 6. Update paper — `paper/paper.tex` + `paper/tables/per_dataset_metrics.tex`
- Regenerate the table via `paper/generate_per_dataset_metrics_table.py` (reads
  `results/latest_benchmark.json`); verify its key→display-name mapping covers
  `notinject_hf`/`browsesafe`/`tensortrust` and the N/A logic for benign-only rows — hand-edit
  the changed numbers if the generator doesn't cover all rows.
- Replace headline numbers in `paper.tex`: abstract (`:48`), contributions (`:67`),
  threshold/recall sentences (`:169`, `:541`), results (`:280`), conclusion bullets (`:534-536`),
  deepset "perfect AUC" caption (`:378` — soften if recall drops).
- Add **`\subsection{Threats to Validity}`** (~`:503`, before Conclusion): document the original
  leakage (first-N overlap; 100% on NotInject-HF; the `[-200:]` internal leak), the remedy
  (holdout subtraction + disjoint val + `verify_no_leak.py`), and note global 80/20 split as
  future work.

### 7. Sequencing & contingency
- Order: edits (1,2) → `verify_no_leak.py` PASS (3) → [optional smoke run] → backup + canonical
  retrain (4) → re-benchmark (5) → table + paper edits (6).
- **Numbers will likely drop** (especially NotInject-HF FPR rising above 0%, SaTML/deepset
  recall falling). Report honest post-fix numbers; do **not** re-tune in any way that
  reintroduces leakage. Re-select the operating threshold only on the **internal disjoint
  validation set**, never on the eval set.
- Soften absolute "perfect"/"0.0%"/"100%" claims to measured values. Keep the pre-fix backup so
  the paper can state "leak-inflated X% vs. leak-free Y%". If results miss the paper's stated
  targets, revise the claims to match reality — do not change the `--paper` exit gate to pass.

## Verification
1. `python scripts/verify_no_leak.py` → `PASS: zero overlap`, every `leaked_into_train=0`.
2. After retrain: `injection_aware_mpnet_classifier_metadata.json` shows `is_trained: true`,
   `model_name: models/injection_aware_mpnet`.
3. Benchmark JSON produced; compare `overall_accuracy`/`overall_fpr` against the backed-up
   pre-fix run to quantify inflation.
4. Paper compiles (needs a LaTeX toolchain — none on this box; `tectonic paper.tex` after install).

## Critical files
- `training/finetune_mpnet_embeddings.py` — leak fix + `[-200:]` fix
- `scripts/verify_no_leak.py` — new audit (gate)
- `benchmarks/benchmark_datasets.py` — loader limits defining the holdout (read-only ref)
- `paper/paper.tex`, `paper/tables/per_dataset_metrics.tex` (+ generator) — numbers + Threats to Validity
- Back up: `models/injection_aware_mpnet/`, `injection_aware_mpnet_classifier.json` (+ metadata)
