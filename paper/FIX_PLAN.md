# Fix Plan — "A Multi-Layer Defense System Against Prompt Injection in Multi-Agent LLMs"

Companion to `CRITIQUE.md`. Every item here cites the critique section it resolves so the two documents stay in lockstep.

**Estimated total effort:** ~5–7 focused workdays for P0+P1+P2. No model retraining required if Decision 3 stays at "keep θ=0.764" and Decision 2 is "remove TensorTrust." If Decision 3 is "re-tune threshold" add ~1 day. If Decision 1 is "restore three-layer framework," add ~3–5 days for missing evaluations.

---

## 0. Decisions to make first (≈30 min)

These five choices determine what's in scope. Make them before touching anything.

| # | Decision | Options | Default if you don't decide |
|---|---|---|---|
| **D1** | Paper framing | (a) Keep three-layer (Detection + Coordination + Response) framing → must restore commented-out §2.5, §7.5, §7.5.x, §7.4 ablation → ~3–5 extra days. (b) Re-scope to Detection-only → rewrite abstract / §1 / §4 to drop Coordination & Response → ~2 hours. | **(b) re-scope** — fastest, honest, defensible undergraduate output. |
| **D2** | TensorTrust | (a) Run the eval (cache is already populated with 200 strings) → ~1–2 hours including writing the script. (b) Remove all TensorTrust references from abstract, §1, §6, §7.1 row, §8 limitations bullet → ~30 min. | **(a) run the eval** — it's cheap and gives you an honest "where the system struggles" data point. |
| **D3** | Threshold θ=0.764 | (a) Keep, but add a validation-curve plot from the new MPNet model. (b) Re-tune on the new MPNet validation set, pick a fresh θ → ~1 day plus all downstream numbers re-run. | **(a) keep + justify** unless you're willing to re-run all evals. |
| **D4** | Author list | (a) Solo "Abdelghafour El Bikha" — confirm Jennifer Marrero (named in `docs/PROJECT_FEEDBACK_REPORT.md`) is not a co-author. (b) Add Jennifer. | Confirm with Jennifer directly; default to (b) only if she contributed to the writing or experiments. |
| **D5** | Code release | (a) Release on submission (paper currently says "released upon publication"). (b) Withhold until accepted. | (a). Gap analysis already promised this; aligns with norms. |

Write your choices on this line so they survive context resets:

> **My choices:** D1 = ____ , D2 = ____ , D3 = ____ , D4 = ____ , D5 = ____

The rest of the plan assumes the **defaults**: D1=(b) re-scope, D2=(a) run, D3=(a) keep, D4=(confirm), D5=(a) release.

---

## Phase 0 — Preflight (≈30 min)

P0a. **Snapshot the repo before edits.**
```bash
cd /Users/goodwiinz/development/prompt-injection-defense
git status   # make sure nothing else is mid-flight
git checkout -b paper-fix-april-2026
git add -A && git commit -m "snapshot before fix-plan edits"
```

P0b. **Move stale results out of the way** (so a reviewer cloning the repo can't pull contradictory numbers).
```bash
mkdir -p results/legacy
git mv results/comprehensive_evaluation.json     results/legacy/
git mv results/benchmark_results_updated.json    results/legacy/
git mv results/benchmark_results.json            results/legacy/
git mv results/balanced_model_evaluation_v2.json results/legacy/
git mv results/balanced_v2_evaluation.json       results/legacy/
git mv results/corrected_balanced_evaluation.json results/legacy/
git mv results/quick_balanced_evaluation_v2.json results/legacy/
git mv results/distilbert_evaluation.json        results/legacy/
git mv results/optimal_threshold_evaluation.json results/legacy/
echo "These files come from prior model versions (pre-Injection-Aware-MPNet) and are kept for provenance only. Paper numbers come from results/injection_aware_mpnet_evaluation.json and results/latest_benchmark_results.txt." > results/legacy/README.md
```
**Acceptance:** `grep -lE 'bit_xgboost_theta_764|MiniLM' results/*.json` returns nothing.

P0c. **Pin the canonical results files** at the top of the paper directory.
```bash
ln -sf ../results/injection_aware_mpnet_evaluation.json paper/canonical_per_dataset.json
ln -sf ../results/latest_benchmark_results.txt          paper/canonical_summary.txt
ln -sf ../results/statistical_analysis.json             paper/canonical_stats.json
```
Now every fix below references one of these three files when it cites a number.

---

## Phase 1 — P0 fixes (must ship)

Order matters — do them top-to-bottom; later fixes depend on earlier ones.

### P0-1. Rebuild `paper/references.bib` (CRITIQUE §D.1) — ~2 hours

**File:** `paper/references.bib`. Replace 8 entries entirely.

| Bib key | Replace arXiv ID with | Replace authors with | Add venue |
|---|---|---|---|
| `lee2025prompt` | **arXiv:2410.07283** | Donghyun Lee, Mo Tiwari | (also rename the bib key to `lee2024prompt`; year is 2024) |
| `chen2024struq` | **arXiv:2402.06363** | Sizhe Chen, Julien Piet, Chawin Sitawarin, David Wagner | USENIX Security '25 |
| `chen2025secalign` | **arXiv:2410.05451** | Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Bo Li, Chuan Guo | CCS '25 |
| `defensivetoken2024` | (search arXiv for "defensive token prompt injection"; if not found, replace this citation in §2.3 with a more general test-time-defense reference such as `Liu et al. 2023 "Prompt Injection attack against LLM-integrated Applications" arXiv:2306.05499`) | — | — |
| `promptarmor2024` | **arXiv:2507.15219** | Shengyun Si, Xinfeng Li, et al. | (workshop / preprint) |
| `liang2024injecguard` | **arXiv:2410.22770** (rename key to `li2024injecguard`) | Hao Li, et al. | ACL '25 (cite as Findings until you confirm main vs Findings) |
| `wang2024peerguard` | **arXiv:2505.11642** (rename key to `fan2025peerguard`) | Fan, Li, et al. | (preprint) |
| `zhan2025jailbreakbench` | If you wanted JailbreakBench: **arXiv:2404.01318**, Patrick Chao et al., NeurIPS '24. If you wanted the NAACL'25 IPI eval paper: replace with that. | depends on which | — |

After editing, verify:
```bash
cd paper && grep -E "^@" references.bib | wc -l         # should equal 11 (or whatever the new total is)
grep -E "00001|00002|00003" references.bib              # should print nothing
grep -i "anonymous" references.bib                      # should print nothing
```

**Acceptance:** every `\citep{...}` in `paper.tex` resolves to a real, citable paper with named authors.

### P0-2. Reconcile internal numbers (CRITIQUE §B.1, B.2, B.3, B.4, B.6) — ~1 hour

**Source of truth from now on:** `results/injection_aware_mpnet_evaluation.json` for per-dataset metrics; `results/latest_benchmark_results.txt` for the headline summary; `results/statistical_analysis.json` for CIs.

Open `paper/paper.tex` and `paper/tables/per_dataset_metrics.tex`. Make a single pass with these substitutions:

| Where | Old | New (canonical) |
|---|---|---|
| Abstract line 48 | `97.2\% accuracy` | `97.3\% accuracy` |
| Abstract line 48 | `P95 latency of \textbf{38.1ms}` | `P95 latency of \textbf{43.3ms}` (or whatever you choose; make it match Conclusion + §7.4) |
| Abstract line 48 | `BrowseSafe (97.8\% accuracy, 0.4\% FPR)` | `BrowseSafe (98.2\% accuracy, 0.5\% FPR)` (matches §7.6 + Conclusion + Table 2) |
| Abstract line 48 | `98.8\% recall on SaTML` | Either `96.0\% recall on SaTML` (matches `latest_benchmark_results.txt`) **or** re-run the SaTML eval and quote the new number with its CI. **Do not leave 98.8 % unless you can point to a results file that produces it.** |
| Conclusion line 513 | `97.3\% overall accuracy across 8 benchmark datasets` | If you remove TensorTrust per D2(b), change to `7 benchmark datasets`. If you keep it per D2(a), keep "8" but make sure the count is right (Table 1 currently has 7 rows). |
| Conclusion line 514 | `0.0\% FPR on the NotInject over-defense benchmark (both local and HuggingFace versions)` | Run the HF variant separately (loader exists at `benchmarks/benchmark_datasets.py:load_notinject_hf_dataset`) **or** drop "(both local and HuggingFace versions)". |

After editing, run this sanity check:
```bash
cd paper && python3 - <<'PY'
import re, pathlib
tex = pathlib.Path("paper.tex").read_text()
for tag in ["97.2", "97.3", "38.1", "43.3", "96.0", "98.8", "97.8", "98.2", "0.4\\%", "0.5\\%"]:
    n = len(re.findall(re.escape(tag), tex))
    print(f"  {tag:8s}  appears {n} times in paper.tex")
PY
```
**Acceptance:** the percentages you keep appear consistently; no "old" number lingers anywhere.

### P0-3. TensorTrust — execute D2 (CRITIQUE §B.5) — ~2 hours if D2=(a), ~30 min if D2=(b)

**If D2 = (a) run the evaluation:**

Create `evaluation/evaluate_mpnet_on_tensortrust.py`:
```python
#!/usr/bin/env python3
"""Evaluate Injection-Aware MPNet on TensorTrust (cached locally)."""
import json, time, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from benchmarks.benchmark_datasets import load_tensortrust_dataset
from src.detection.embedding_classifier import EmbeddingClassifier

ds = load_tensortrust_dataset(limit=500, use_cache=True)
texts, labels = list(ds.texts), list(ds.labels)
clf = EmbeddingClassifier(model_name="models/injection_aware_mpnet", threshold=0.764)
clf.load_model("models/injection_aware_mpnet_classifier.json")

t0 = time.time()
preds = clf.predict(texts)
dur = (time.time() - t0) * 1000

tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
recall = tp / (tp + fn) if (tp + fn) else 0
n = len(labels)
result = {
    "dataset": "TensorTrust", "model": "injection_aware_mpnet",
    "threshold": 0.764, "samples": n, "tp": tp, "fn": fn,
    "recall": recall, "accuracy": recall, "duration_ms": dur,
}
out = Path("results/tensortrust_mpnet_evaluation.json")
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
```
Run it: `cd /Users/goodwiinz/development/prompt-injection-defense && PYTHONPATH=. .venv/bin/python evaluation/evaluate_mpnet_on_tensortrust.py`

Then update Table 2 row to use the actual number, with a Wilson CI footnote.

**If D2 = (b) remove TensorTrust:**

```bash
cd paper && grep -nE "TensorTrust|tensortrust" paper.tex tables/*.tex
```
Delete every match. Specifically:
- Abstract: drop "TensorTrust" from the dataset list.
- §1 contributions: drop "TensorTrust" from the 8-dataset list; change "8" to "7" or the actual remaining count.
- `tables/per_dataset_metrics.tex`: delete the TensorTrust row.
- `tables/dataset_summary.tex`: delete the TensorTrust row; recompute the Total row.
- §8 Limitations: delete the bullet "TensorTrust evaluation shows 83.2% accuracy …".

**Acceptance:** `grep -i tensortrust paper/paper.tex paper/tables/*.tex` returns the same as `grep -i tensortrust results/*.json` — either both have it (option a) or neither does (option b).

### P0-4. Rewrite §5.2 training-data section (CRITIQUE §C.1, C.2) — ~1 hour

**File:** `paper/paper.tex` lines 198–252.

Replace the "Training Data Sources" itemize and the "XGBoost Classifier Training" itemize with the *actual* numbers from `models/injection_aware_mpnet_metadata.json` and the actual `limit=` arguments in `training/finetune_mpnet_embeddings.py` and `training/train_injection_aware_classifier.py`.

Suggested rewrite:
> **Contrastive fine-tuning data.** We construct 14,000 sentence pairs from
> SaTML CTF (limit 2,000), deepset (1,000 safe + injections), LLMail (limit 2,000),
> BrowseSafe (limit 2,000), NotInject HuggingFace (limit 500), and NotInject
> synthetic (1,000 generated samples). Pairs are sampled per Table~3 with the
> doubled injection–benign-trigger weighting.
>
> **XGBoost classifier training.** The fine-tuned 768-dim embeddings are used
> as features for an XGBoost classifier trained on 3,650 examples drawn from
> the contrastive corpus: 2,667 safe-benign, 570 benign-with-trigger, and 413
> malicious. Class imbalance is handled by XGBoost's `scale_pos_weight`
> (2.96 ≈ negative/positive ratio). Hyperparameters: `n_estimators=300`,
> `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.

If you would prefer the bigger dataset described in the *current* paper (2,800 / 2,800 / 1,400), then you must **retrain the XGBoost classifier** on a larger malicious set drawn from the actual benchmark loaders — that's a separate work item, ~2 hours, plus re-running every downstream evaluation. The shortest path is to honestly report the smaller dataset.

**Acceptance:** the three counts in §5.2 match `models/injection_aware_mpnet_metadata.json` exactly.

### P0-5. Multi-layer framing — execute D1 (CRITIQUE §F.1) — ~2 hours if D1=(b), ~3–5 days if D1=(a)

**If D1 = (b) re-scope to Detection-only (recommended):**

Edit `paper/paper.tex`:
- **Title:** change to *"Injection-Aware Sentence Embeddings: Suppressing Over-Defense in Prompt Injection Detection"* (or similar).
- **Abstract:** delete "comprehensive three-layer defense framework (Detection, Coordination, Response) designed specifically for multi-agent environments." Replace with single-sentence description of the detector.
- **§1 contributions:** delete the "Three-Layer Architecture" bullet entirely. Keep "Ensemble Detection," "Injection-Aware MPNet," "Comprehensive Benchmarking."
- **§4 System Design:** delete §4.2 (Coordination Layer) and §4.3 (Response Layer). Renumber.
- **§7 intro:** delete the bulletted list that forward-references §7.5.
- **§8.0.1 ("Limited Effectiveness Over Long Workflows"):** delete (depends on Guard Agent number that no longer has a source).
- **Conclusion:** drop "multi-layer" wording; lead with the embedding contribution.

**If D1 = (a) restore three-layer evaluation:**

You'll need to:
- Uncomment lines 401–463 (§7.5 multi-agent attacks, OVON, quarantine, PeerGuard).
- Re-run the underlying experiments — most don't have results files. Check `tests/test_integration.py` for what's actually executable.
- Restore the BIT-vs-MOF positioning section (lines 106–136). You can keep the honest "complementary not superior" framing.
- Generate the missing tables (`tables/multi_agent_attacks.tex`, `tables/guard_bypass_testing.tex`, `tables/ovon_security.tex`, `tables/quarantine_effectiveness.tex`, `tables/peerguard_comparison.tex`).

This path is realistic only if those evaluations were genuinely run earlier and the numbers exist somewhere outside the repo. If they have to be produced from scratch, budget 3–5 days.

**Acceptance:** §1 contributions, §4 sections, §7 evaluations, and §9 conclusion all describe the same scope. No subsection forward-references a section that doesn't exist.

---

## Phase 2 — P1 fixes (major weaknesses)

### P1-1. Add confidence intervals everywhere (CRITIQUE §E.1) — ~3 hours

**Files:** `paper/tables/per_dataset_metrics.tex`, `paper/tables/baseline_comparison.tex`, abstract, conclusion.

Use Wilson score for proportions. Already-computed CIs live in `results/statistical_analysis.json`:
```
accuracy:  0.980 [0.968, 0.992]
precision: 0.993 [0.978, 1.000]
recall:    0.943 [0.904, 0.977]
f1:        0.967 [0.945, 0.984]
```

Per-dataset Wilson CIs you'll need to compute. Quick helper:
```python
# paper/scripts/wilson_ci.py
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    denom = 1 + z**2/n
    centre = p + z**2/(2*n)
    half = z * ((p*(1-p) + z**2/(4*n))/n)**0.5
    return ((centre - half)/denom, (centre + half)/denom)

# Examples:
print("NotInject FPR (0/247):", wilson(0, 247))    # (0.000, 0.0153)
print("BrowseSafe recall:", wilson(int(0.967*227), 227))
print("SaTML recall:", wilson(...))
```

In Table 2, change the column from `Recall` to `Recall (95% CI)` and report e.g. `100.0 [98.2, 100.0]`.

In §6 (Statistical Methodology), add: *"All proportions are reported with Wilson 95 % confidence intervals; differences are tested with McNemar's test on paired predictions when comparing models on the same dataset."*

**Acceptance:** every percentage in Tables 2, 5, and abstract has a CI in brackets. No bare percentage left except in qualitative prose.

### P1-2. Implement IDR computation script (CRITIQUE §B.7) — ~1 hour

**New file:** `evaluation/compute_idr.py`.

```python
#!/usr/bin/env python3
"""Compute Intra-class Distance Ratio (IDR) for baseline vs Injection-Aware MPNet."""
import numpy as np, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sentence_transformers import SentenceTransformer
from benchmarks.benchmark_datasets import (
    load_deepset_dataset, load_notinject_dataset, load_satml_dataset,
)

def centroid(texts, model):
    embs = model.encode(texts, normalize_embeddings=True)
    return embs.mean(axis=0)

def idr(model_path, label):
    m = SentenceTransformer(model_path)
    benign = [t for t, y in load_deepset_dataset() if y == 0][:200]
    triggers = [t for t, _ in load_notinject_dataset(limit=200)]
    injections = [t for t, _ in load_satml_dataset(limit=200)]
    c_b, c_t, c_i = centroid(benign, m), centroid(triggers, m), centroid(injections, m)
    d_tb = np.linalg.norm(c_t - c_b)
    d_ti = np.linalg.norm(c_t - c_i)
    print(f"{label}: d(trigger,benign)={d_tb:.4f}  d(trigger,injection)={d_ti:.4f}  IDR={d_tb/d_ti:.4f}")

idr("sentence-transformers/all-mpnet-base-v2", "baseline")
idr("models/injection_aware_mpnet",            "fine-tuned")
```

Run, paste the actual numbers into the §7.4.5 IDR table.

**Acceptance:** the table cites a script that produces those exact numbers.

### P1-3. Restore adaptive-attack preliminary results (CRITIQUE §E.3) — ~30 min

**File:** `paper/paper.tex` lines 481–493 (currently commented out).

Uncomment, and add a clear preamble:
> **Preliminary adaptive-attack results (n=36 — small sample).** As a sanity check we ran a limited GCG and AutoDAN sweep against the embedding classifier; full-scale adaptive evaluation is out of scope for this paper but preliminary results are reported here with their wide confidence intervals.

Add Wilson CIs to the existing percentages.

**Acceptance:** the section appears in the rendered PDF, with CIs and a "preliminary, n=36" caveat.

### P1-4. Justify θ=0.764 on the new MPNet model (CRITIQUE §C.3) — ~30 min

**File:** new appendix or a new figure in §5.

`evaluation/evaluate_theta_764_model.py` already runs a threshold sweep. Just save the output as a plot and reference it.

```bash
cd /Users/goodwiinz/development/prompt-injection-defense
PYTHONPATH=. .venv/bin/python evaluation/evaluate_theta_764_model.py > paper/figures/threshold_sweep.txt
# Then use paper/generate_ablation_charts.py (already exists) to render a plot.
```

In §5.1 add: *"Figure X shows the validation precision/recall trade-off across thresholds; θ=0.764 is the smallest threshold for which validation recall exceeds 0.98."*

**Acceptance:** there is a figure backed by a results file showing the sweep.

### P1-5. NotInject HuggingFace evaluation (CRITIQUE §B.4) — ~30 min

**Loader exists** at `benchmarks/benchmark_datasets.py:load_notinject_hf_dataset` (referenced from `finetune_mpnet_embeddings.py`).

Quick eval:
```bash
cd /Users/goodwiinz/development/prompt-injection-defense
PYTHONPATH=. .venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.append('.')
from benchmarks.benchmark_datasets import load_notinject_hf_dataset
from src.detection.embedding_classifier import EmbeddingClassifier
ds = load_notinject_hf_dataset()
texts, labels = list(ds.texts), list(ds.labels)
clf = EmbeddingClassifier(model_name="models/injection_aware_mpnet", threshold=0.764)
clf.load_model("models/injection_aware_mpnet_classifier.json")
preds = clf.predict(texts)
fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
n = len(labels)
print(f"NotInject HF (n={n}): FP={fp}, TN={tn}, FPR={fp/(fp+tn):.4f}")
Path("results/notinject_hf_mpnet_evaluation.json").write_text(json.dumps({
  "dataset": "NotInject HF", "samples": n, "fp": fp, "tn": tn, "fpr": fp/(fp+tn) if (fp+tn) else 0
}))
PY
```

Update Table 2 row "NotInject (HuggingFace)" with the real number + CI.

**Acceptance:** there's a `results/notinject_hf_mpnet_evaluation.json` file backing the row.

---

## Phase 3 — P2 polish (≈half-day)

| ID | Task | File | Concrete change |
|---|---|---|---|
| P2-1 | Fix subsection numbering (CRITIQUE §F.2) | `paper/paper.tex` line 340 | Promote §7.4.5 "Embedding Space Analysis" from `\subsubsection` to `\subsection`, OR put it under a new `\subsection{Embedding Space Geometry}` parent. |
| P2-2 | Define threat-model jargon (CRITIQUE §F.3) | `paper/paper.tex` lines 154–158 | Either define "Model extraction attacks", "Feedback-based attacks", "Distributed attacks" in one sentence each, or delete them. |
| P2-3 | Untruncate qualitative examples (CRITIQUE §F.7) | `paper/tables/qualitative_examples.tex` | Replace truncated prompts with full text or a note "(truncated for space; full prompt in supplementary)". |
| P2-4 | Author list (D4) | `paper/paper.tex` lines 39–40 | Confirm with Jennifer; if she contributed, add `\author[1,2]{Jennifer Marrero}` block. |
| P2-5 | Remove unused citations (CRITIQUE §F.5) | `paper/paper.tex` | If D1=(b), `wang2024peerguard` and `lee2025prompt` only appear in commented-out sections — remove those bib entries OR add a one-sentence reference in §2 so `natbib` doesn't warn. |
| P2-6 | Fix file name typo | `paper/my_papper.md` | `git mv my_papper.md paper.md` (and update any references). |
| P2-7 | State raw AUC (CRITIQUE §B.8) | Figure 3 caption | Change "AUC = 1.00" to "AUC = 0.998" (the actual number from `injection_aware_mpnet_evaluation.json`). |
| P2-8 | Define ensemble weights (CRITIQUE §F.6) | `paper/paper.tex` Eq. (1) | Either give numerical values for $w_{emb}, w_{pattern}$ or note that the pattern detector is currently set to weight zero. |
| P2-9 | Tighten IDR equation prose (CRITIQUE §F.6) | `paper/paper.tex` §6 metrics paragraph | Change "mean Euclidean distance between class centroids" to "ratio of Euclidean distances between the benign-trigger centroid and the benign / injection centroids". |
| P2-10 | Mark vendor numbers in baseline table (CRITIQUE §D.2, F.7) | `paper/tables/baseline_comparison.tex` | Add a `Source` column or split the table into "Direct measurement (this work)" and "Vendor-reported / external benchmark" sections. |
| P2-11 | Qualify BrowseSafe sample size (CRITIQUE §D.3) | `paper/paper.tex` §7.6 first sentence | "BrowseSafe \citep{...}, of which we evaluate a 500-sample subset." |
| P2-12 | Bibliography compile check | `paper/` | After all bib edits, run `pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex`; the warning list at the end should have no "Citation undefined" or "There were undefined references" lines. |

---

## Phase 4 — Final verification checklist (≈1 hour)

Run the entire checklist before re-submitting anywhere. Everything below should be answerable "yes" or "n/a."

**Numbers:**
- [ ] Every percentage in the abstract appears in `results/injection_aware_mpnet_evaluation.json`, `results/latest_benchmark_results.txt`, or a new results file produced in this fix cycle.
- [ ] No number in the conclusion contradicts the abstract.
- [ ] Every cell in `paper/tables/per_dataset_metrics.tex` has a 95 % CI.
- [ ] No `tables/*.tex` row references a dataset whose results file is in `results/legacy/`.

**Bibliography:**
- [ ] `grep -E "00001|00002|00003" paper/references.bib` returns nothing.
- [ ] `grep -i anonymous paper/references.bib` returns nothing.
- [ ] Every `\citep{...}` in `paper.tex` resolves; no LaTeX warning "Citation undefined."
- [ ] Every entry in `references.bib` is cited at least once; no LaTeX warning "There were unused references."

**Scope coherence:**
- [ ] §1 contributions, §4 sections, §7 results, and §9 conclusion describe the same set of layers.
- [ ] Every forward-reference (`Section 7.5`, `Figure X`) resolves to a section/figure that exists.
- [ ] Every `\input{tables/...}` succeeds.

**Repository hygiene:**
- [ ] `results/legacy/README.md` explains why the legacy files are kept.
- [ ] Every script referenced in the paper (`evaluation/compute_idr.py`, `evaluation/evaluate_mpnet_on_tensortrust.py`, etc.) runs end-to-end on a fresh checkout.
- [ ] `tests/test_detection.py` and `tests/test_integration.py` pass: `cd /Users/goodwiinz/development/prompt-injection-defense && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`.

**Submission readiness:**
- [ ] PDF compiles with no LaTeX warnings.
- [ ] Author list confirmed (D4).
- [ ] Code release plan stated in conclusion (D5).
- [ ] `CRITIQUE.md` and `FIX_PLAN.md` deleted from the submitted bundle (they're for internal use).

---

## What this plan deliberately does NOT include

- **Adaptive-attack research at scale** (full GCG/AutoDAN evaluation per `paper_gap_analysis.md` §1). Doing this properly is a separate ~2-week effort and changes the paper's scope. The P1-3 task adds the *preliminary* n=36 result back; full evaluation is future work.
- **XGBoost adversarial robustness analysis** (`paper_gap_analysis.md` §3). Same reasoning — a separate paper.
- **Multi-turn attack evaluation** (`paper_gap_analysis.md` §6, §8). Out of current scope; should be acknowledged in §8 Limitations.
- **Multilingual evaluation** (`paper_gap_analysis.md` §10). Out of current scope.
- **RBAC / role-based permissions** (`paper_gap_analysis.md` §9). Out of current scope.

These are all legitimate future-work items; trying to address them now would turn a 1-week revision into a 6-month project.

---

## Suggested working order (one-pass execution)

If you're sitting down to do all of this in one stretch:

```
Day 1 morning   : D1–D5 decisions; P0a/b/c snapshot; P0-1 bibliography rebuild
Day 1 afternoon : P0-3 TensorTrust (run or remove); P0-4 §5.2 rewrite
Day 2 morning   : P0-2 reconcile internal numbers; P0-5 framing edit
Day 2 afternoon : P1-1 add CIs everywhere; P1-2 IDR script
Day 3 morning   : P1-3 GCG/AutoDAN restore; P1-4 threshold justification; P1-5 NotInject HF
Day 3 afternoon : Phase 3 polish (P2-1 through P2-12)
Day 4 morning   : Phase 4 verification checklist; final compile; second-pair-of-eyes review
```

End state: a paper whose every quantitative claim is reproducible from a script in the repo, whose bibliography contains real papers, and whose framing matches the actual evaluations.
