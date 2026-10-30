# Reviewer-Style Critique: "A Multi-Layer Defense System Against Prompt Injection in Multi-Agent LLMs"

**Reviewer:** Independent (claim-and-code audit)
**Source files audited:** `paper/paper.tex` (526 lines), `paper/tables/*.tex`, `paper/references.bib`, `results/*.json`, `results/latest_benchmark_results.txt`, `evaluation/*.py`, `training/*.py`, `models/injection_aware_mpnet*`, `data/`, `tests/`.
**Date:** 2026-04-26
**Recommendation:** **Major revision required before any external submission.** A workshop or undergraduate venue may accept a revised version; the work is **not** in shape for IEEE SaTML, USENIX Security, NDSS, CCS, or ACL Findings as currently written.

---

## 0. Executive summary

There is real engineering work behind the paper: a fine-tuned MPNet model exists on disk, an XGBoost classifier on top of it has been trained and evaluated, and several benchmark loaders and evaluation scripts are in the repo. The core idea — using contrastive fine-tuning of sentence embeddings with deliberate emphasis on "injection vs benign-with-trigger-word" pairs to suppress over-defense — is reasonable and well-aligned with the literature.

But the **paper as written cannot be reproduced from the code** and contains numbers that contradict the repository's own results files. In several places the paper appears to report numbers that no script in the repo produces (TensorTrust 83.2 %, IDR 0.67/2.31, validation Spearman 0.904, P95 38.1 ms). The bibliography is built almost entirely on fake placeholder arXiv IDs. About 40 % of the paper is `%`-commented out, including everything that would substantiate the abstract's claim of a "three-layer" framework — Sections 4.2 (Coordination) and 4.3 (Response) are introduced but never evaluated in the live text.

The four most serious issues, in order:

1. **Numbers in the abstract, body, conclusion, and tables disagree with each other and with the repo's results files** (Section B).
2. **The training data described in §5.2 is not the data the model was actually trained on** (Section C).
3. **TensorTrust results appear fabricated**: the dataset cache is empty, no script evaluates the new model on TensorTrust, and the reported 83.2 % accuracy is suspiciously identical to an old SaTML run (Section B.5).
4. **The bibliography is mostly placeholder arXiv IDs (2402.00001, 2403.00001, …) attributed to "Anonymous"** despite all referenced works being real, well-known, and easy to cite (Section D).

The good news: every single one of these is fixable with editing and one or two additional evaluation runs. None require new modeling work.

---

## A. Reviewer-style summary (the kind of paragraph a PC member would write)

**Strengths.** The over-defense framing is timely and the contrastive-pair construction (especially the doubled "injection vs benign-with-trigger" pairs) is a clean, defensible methodological choice. Reporting on NotInject specifically is appropriate. The repository contains a working model, a unified benchmark harness across multiple datasets, and a quick-start evaluation script. The qualitative-examples table is a nice touch.

**Weaknesses.**
- The abstract advertises a *three-layer* (Detection / Coordination / Response) framework, but the body only evaluates the Detection layer — every figure or table that would back the Coordination and Response layer claims is commented out (lines 106–136, 311–338, 401–463, 481–493 of `paper.tex`).
- Headline numbers in the abstract, the per-dataset table, and the conclusion are mutually inconsistent (97.2 % vs 97.3 %, 38.1 ms vs 43.3 ms, BrowseSafe 97.8/0.4 vs 98.2/0.5).
- Per-dataset numbers in Table 2 (`per_dataset_metrics.tex`) cannot be reproduced from any single results file in the repo.
- The training-data and XGBoost-training counts in §5.2 contradict the model metadata file (`models/injection_aware_mpnet_metadata.json`).
- Statistical methodology paragraph (§6, "Wilson score CIs, McNemar's test") is *promised* but never used — no CI is reported anywhere in the paper.
- The bibliography is unusable: 8 of 11 entries have placeholder arXiv IDs and several are attributed to "Anonymous" or wrong authors.
- The TensorTrust row in Table 2 is unsupported by any results file and the cached dataset is empty.

**Decision.** Major revision. The work could become a credible undergraduate honors thesis or workshop paper after the issues in Sections B, C, and D below are resolved.

---

## B. Claim-vs-evidence audit (numbers in paper vs numbers on disk)

I audited every quantitative claim in the abstract, the per-dataset table, the baseline-comparison table, the IDR table, and the conclusion. Each row below pairs the claim with the file where its evidence should live.

### B.1 Internal inconsistencies (paper vs paper)

| # | Claim location 1 | Claim location 2 | Discrepancy |
|---|---|---|---|
| 1 | Abstract: **97.2 %** accuracy | Conclusion + Table 2 overall: **97.3 %** | ±0.1 pp; pick one |
| 2 | Abstract: P95 latency **38.1 ms** | §7.4 + Conclusion: **43.3 ms** | These are completely different numbers |
| 3 | §7.4 + Conclusion: P50 **35.5 ms**, P95 **43.3 ms** | Table 2 per-dataset max P95 = **31.6 ms** (BrowseSafe) | Where does 43.3 come from? |
| 4 | Abstract: BrowseSafe **97.8 %** acc, **0.4 %** FPR | §7.6 + Conclusion: BrowseSafe **98.2 %** acc, **0.5 %** FPR | Different numbers in the same paper |
| 5 | Abstract: "**98.8 % recall on SaTML**" | Table 2 row: SaTML acc 98.8 %, recall 98.8 % | Recall and accuracy on a single-class dataset are the same number; OK in spirit, but the abstract phrasing implies these are independent measurements |
| 6 | Abstract: "evaluated on … (SaTML, deepset, LLMail, NotInject, **TensorTrust**, BrowseSafe)" | Table 2 lists 8 rows including TensorTrust at 83.2 % | TensorTrust has *no evaluation results file in the repo* — see B.5 |
| 7 | §1 contributions: "comprehensive benchmarking … 8 datasets" | Table 1 lists **7 dataset rows** (deepset full + deepset injections-only is one dataset; NotInject local + NotInject HF is one dataset) | "8 datasets" is at most 6 distinct datasets |

### B.2 SaTML

| Source | Sample size | Recall reported | Accuracy reported |
|---|---|---|---|
| `paper.tex` Table 2 (`per_dataset_metrics.tex`) | 500 (per dataset_summary.tex) | **98.8 %** | 98.8 % |
| `results/latest_benchmark_results.txt` (2025-12-16 04:38) | unspecified | **96.0 %** | 96.0 % |
| `results/benchmark_results_updated.json` (2025-12-08) | 498 | **83.13 %** (414/498) | 83.13 % |
| `results/benchmark_results.json` (2025-12-04) | 100 | 97.0 % | 97.0 % |
| `results/comprehensive_evaluation.json` (2025-12-14) at θ=0.764, **OLD MiniLM model** (`bit_xgboost_theta_764`, not MPNet) | 1000 | **74.2 %** | — |
| `results/injection_aware_mpnet_evaluation.json` (2025-12-16 03:24) | "2000/136000 sampled" but **not broken out** in the JSON | — | — |

**Verdict:** No single file in the repository produces 98.8 % SaTML recall at the configuration the paper claims (`injection_aware_mpnet`, θ=0.764). The closest evidence is the 96.0 % in the 2025-12-16 04:38 benchmark run. Note: the 74.2 % row above is from the *previous-generation* MiniLM-based model (`bit_xgboost_theta_764`), not the MPNet model — it's listed for completeness because both models share the θ=0.764 threshold name.

### B.3 LLMail

| Source | Recall |
|---|---|
| Paper Table 2 | **100.0 %** |
| `latest_benchmark_results.txt` | 100.0 % ✓ |
| `benchmark_results_updated.json` | 98.8 % |
| `comprehensive_evaluation.json` (θ=0.764, OLD MiniLM model) | 66.3 % |

**Verdict:** Paper number is consistent with the latest benchmark run for the MPNet model. The 66.3 % row is from the *previous-generation* MiniLM model that happens to share the θ=0.764 threshold name; it is listed only because a reviewer cloning the repo will see both and ask which is canonical. The paper should clarify which run the headline numbers come from and either delete the older results or move them to `results/legacy/`.

### B.4 NotInject FPR

| Source | FPR |
|---|---|
| Paper abstract + Table 2 | **0.0 %** |
| `latest_benchmark_results.txt` | 0.0 % ✓ |
| `injection_aware_mpnet_evaluation.json` (NotInject local, n=247) | 0.0 % ✓ |
| `benchmark_results_updated.json` (older MiniLM model) | 3.24 % |

**Verdict:** This claim *is* reproducible for the MPNet model on the local NotInject (247 samples). Two caveats: (a) the paper also claims 0.0 % on the **HuggingFace** NotInject (339 samples) but the repo only has results for the **local** 247-sample variant; (b) at n=247 with 0 FPs, the 95 % Wilson CI is [0.0 %, 1.5 %] — the paper's own §6 says it computes Wilson CIs but never reports them.

### B.5 TensorTrust (most serious)

- Paper Table 2 row: **TensorTrust 83.2 % accuracy, 100 % precision, 83.2 % recall, 90.9 % F1, 0.0 % FPR, n=500.**
- `data/tensortrust_cache.json` exists and **contains 200 cached attack strings** (52 KB). The dataset is loadable.
- `grep -i tensortrust` over `results/` returns zero hits — no evaluation results file mentions TensorTrust.
- No script in `evaluation/` or `scripts/` evaluates the `injection_aware_mpnet` model on TensorTrust.
- The number **83.2 %** is suspiciously identical (within rounding) to `benchmark_results_updated.json` SaTML row (acc 0.8313, recall 0.8313, prec 1.0, fpr 0, n=498) — a result from a *different model* (older MiniLM-based classifier) on a *different dataset*.

**Verdict:** The dataset is loadable but the row was never produced. The number appears to have been copied from another row of an older result file. **It cannot be left in the paper as-is.** Either run the evaluation (the loader exists at `benchmarks/benchmark_datasets.py:load_tensortrust_dataset` and the cache is already populated), or remove the TensorTrust row and the corresponding sentences in §1, §6, §7.1, and §8.

### B.6 BrowseSafe

| Source | Acc | Recall | FPR |
|---|---|---|---|
| Abstract | 97.8 % | — | 0.4 % |
| §7.6 + Conclusion + Table 2 | 98.2 % | 96.7 % | 0.5 % |
| `latest_benchmark_results.txt` | 98.0 % | 94.1 % | 0.0 % |

**Verdict:** Two different (acc, recall, FPR) triples appear *inside the paper* for BrowseSafe (97.8/—/0.4 in the abstract, 98.2/96.7/0.5 in §7.6 / Conclusion / Table 2), and a third triple appears in the most recent benchmark run on disk. Pick one and propagate.

### B.7 IDR (Intra-class Distance Ratio)

- Paper §7.4.5: baseline `all-mpnet-base-v2` IDR = 2.31, Injection-Aware MPNet IDR = 0.67.
- `grep -ri "IDR\|intra.*distance"` over `src/`, `evaluation/`, `scripts/`, `training/`: **zero hits.**

**Verdict:** No code in the repository computes IDR, so neither value can be checked. Either add the script or remove the table.

### B.8 ROC/PR AUC = 1.00

- Figure 3 caption: "ROC AUC = 1.00, PR AUC = 1.00" on deepset.
- `injection_aware_mpnet_evaluation.json` records `auc: 0.0` for `deepset_benign` and `deepset_injections` individually (because each is single-class) and **AUC ≈ 0.9985** for the *Overall* combined evaluation.

**Verdict:** AUC 1.00 is rounding from 0.9985, which is fine if the paper says "AUC ≈ 1.00 (0.9985)". As written it claims a perfect classifier, which can be misleading and invites scrutiny. State the raw number.

### B.9 MPNet validation Spearman = 0.904

- Paper §5.2: "Validation Spearman correlation: 0.904."
- `models/injection_aware_mpnet/eval/similarity_evaluation_injection-detection-val_results.csv` records *epoch-end* values only:

  ```
  epoch,steps,cosine_pearson,cosine_spearman
  1.0,3500,0.6662,0.7108
  2.0,7000,0.8283,0.8327
  3.0,10500,0.9388,0.8890
  ```

  Last *full* epoch logged: 3.0, Spearman 0.889.
- The README at `models/injection_aware_mpnet/README.md` does record Spearman 0.9040 — but at **step 12500, epoch 3.5714** (line 511), and the README lists "spearman_cosine 0.904" as the headline metric (line 310, 204).

**Verdict:** The 0.904 is reachable from the repo but it's the *mid-epoch-3.57* value, not an end-of-epoch-4 value. Combined with the paper saying "Epochs: 4," this means training was effectively stopped roughly 89 % of the way through the planned schedule. State the actual stopping point (e.g., "trained for 3.57 epochs / 12,500 steps; final validation Spearman 0.904") and either drop the "Epochs: 4" line or note that training was halted early.

---

## C. Methodology misrepresentation

The paper's §5.2 description of the training data does not match what the training script actually does or what the model metadata records.

### C.1 Contrastive fine-tuning data sources

Paper §5.2 (lines 202–207):
> Injection samples: SaTML CTF (1,500), deepset attacks (500), LLMail (500), BrowseSafe injections (2,000)
> Safe samples: deepset benign (1,000), BrowseSafe benign
> Benign-trigger samples: NotInject HuggingFace (500), NotInject synthetic (1,000)

Actual `training/finetune_mpnet_embeddings.py`:
- `load_browsesafe_dataset(limit=2000)` ✓
- `load_deepset_dataset(limit=1000, include_safe=True, include_injections=False)` ✓ (1000 safe, not 500 attack)
- The script passes `samples_per_type=2000` to a pair builder that constructs 2000+2000+2000+2000+4000+2000 = **14000 InputExamples** (matches the paper's 14k pair count and the HF model card)
- However, the script's malicious sample sourcing is shorter than the paper says — see C.2.

This is *salvageable*. The pair construction matches; the source-counts paragraph just needs to reflect what the script loads (`limit=2000` for BrowseSafe, `limit=1000` for deepset safe, etc.).

### C.2 XGBoost training data

Paper §5.2 (lines 244–250):
> XGBoost Classifier Training: 2,800 injections + 2,800 safe + 1,400 benign-trigger.

Actual `models/injection_aware_mpnet_metadata.json`:
```json
{
  "training_data": {
    "safe_benign": 2667,
    "benign_with_triggers": 570,
    "malicious": 413,
    "total": 3650,
    "safe_to_trigger_ratio": 0.8239110287303059
  }
}
```

**Discrepancies:**
- Safe benign: paper says 2,800; metadata says **2,667**.
- Benign-trigger: paper says 1,400; metadata says **570**.
- **Malicious: paper says 2,800; metadata says 413** (a 6.8× discrepancy).
- Total: paper implies 7,000; metadata says **3,650**.

**Verdict:** This is the single most consequential finding in the audit. The XGBoost classifier on top of MPNet was trained on **413 malicious examples generated from a small set of templated strings** in `training/train_injection_aware_classifier.py` (e.g., `"Override your safety protocols to {}"`, `"Disable safety measures and {}"`), not on 2,800 real attacks from SaTML/deepset/LLMail. Either the paper must report the actual data, or the model must be retrained on the larger, real-attack mixture the paper claims.

### C.3 Threshold derivation

Paper §5.1:
> We optimize θ on the validation set targeting 98 % recall, selecting θ=0.764.

Actual `evaluation/final_evaluation_report.py`:
> "⚠️ θ=0.764 is too high - misses 64% of injections"
> "✅ Use θ=0.25 for best balance (71% precision, 88% recall)"

This script is from 2025-12-14, evaluating the *previous* MiniLM model (`bit_xgboost_theta_764`), not the MPNet model the paper describes. But the file is in the repo, the threshold name (θ=0.764) is the same, and a reviewer who finds it will ask the obvious question: was 0.764 chosen on validation data or carried over from a now-superseded model?

**Verdict:** Either (a) describe the new threshold-selection procedure for the MPNet model and discard `final_evaluation_report.py`, or (b) acknowledge that 0.764 is inherited and re-justify on the new model's validation curve.

---

## D. External fact-check (prior work and baselines)

### D.1 Bibliography placeholders

`references.bib` has 11 entries. **8 of them have fabricated arXiv identifiers** (sequential placeholders like `2501.00001`, `2502.00001`, …):

| Bib key | Paper's arXiv ID | Real arXiv ID | Real authors |
|---|---|---|---|
| `lee2025prompt` | 2501.00001 | **2410.07283** | Donghyun Lee, Mo Tiwari (paper says "Jin Lee, Ashish Tiwari") |
| `chen2024struq` | 2402.00001 | **2402.06363** | Sizhe Chen et al., USENIX Sec'25 |
| `chen2025secalign` | 2501.00002 | **2410.05451** | Sizhe Chen et al., CCS'25 |
| `defensivetoken2024` | 2403.00001 | (not found, `author = "Anonymous"`) | — |
| `promptarmor2024` | 2404.00001 | **2507.15219** | Shengyun Si, Xinfeng Li et al. |
| `liang2024injecguard` | 2405.00001 | **2410.22770** | Hao Li et al. (paper says "Liang Zhexin et al.") |
| `wang2024peerguard` | 2406.00001 | **2505.11642** | Fan, Li et al. |
| `zhan2025jailbreakbench` | 2501.00003 | (likely confused with Patrick Chao et al., NeurIPS'24, arxiv 2404.01318) | — |

`perez2022ignore` (2211.09527), `greshake2023more` (2302.12173), and `perplexity2025browsesafe` (real arxiv 2511.20597) are the only entries with plausible IDs.

**Verdict:** The bibliography looks like it was generated by an LLM from memory and never verified against arXiv. This will be caught instantly by any reviewer who clicks a link, and "Anonymous" authorship for two entries (PromptArmor, DefensiveToken) is a near-automatic desk-reject signal at most venues. **Replace every entry with the real metadata.**

### D.2 Baseline numbers

| System | Paper (Table 5) | Independent verification |
|---|---|---|
| **PromptArmor** | FPR 0.6 %, FNR 0.1 % | PromptArmor-GPT-4.1: 0.56 % FPR, 0.13 % FNR on AgentDojo (matches) |
| **StruQ** | "<2 % ASR for Llama-2 and Mistral" | StruQ paper: <2 % ASR for *Llama-7B* and *Mistral-7B* (not Llama-2 specifically) |
| **SecAlign** | "near-zero ASR on optimization-free attacks" | SecAlign paper: <10 % ASR even against sophisticated attacks (your wording is too strong) |
| **Lakera Guard** | acc 87.9 %, FPR 5.7 %, latency 66 ms | Lakera marketing: 98 %+ detection, <0.5 % FPR, sub-50 ms; PINT benchmark: 92.5 %. Source for 87.9 % unclear — cite it. |
| **PIGuard / InjecGuard MOF** | ~2.1 % FPR (in commented-out section) | Original paper claims state-of-the-art on NotInject "surpassing existing best by 30.8 %"; the 2.1 % number needs a page-and-line citation |
| **Glean AI / ActiveFence** | 97.8 % / 5.4 % FPR | These are vendor-reported, not from a comparable benchmark. Mark explicitly. |

The baseline-comparison footnote already says "Numbers from published benchmarks; may use different test sets" — that's appropriate, but the table's "98.5 % lower FPR than Lakera" hard claim in the footnote then directly contradicts it. Either qualify the comparison or drop it.

### D.3 NotInject and BrowseSafe sample counts

- **NotInject 339 benign samples**: ✓ matches Li et al. (arxiv 2410.22770).
- **BrowseSafe 14,719 samples**: ✓ matches Perplexity (arxiv 2511.20597) — but the paper only evaluates 500 (per `dataset_summary.tex`), so qualify the claim ("on a 500-sample subset of the 14,719-sample BrowseSafe-Bench").

---

## E. Statistical and scientific rigor

### E.1 The promised statistics are missing

Paper §6:
> All reported metrics include 95 % confidence intervals (CI) computed using the Wilson score method … Statistical significance for classifier comparison is assessed using McNemar's test (p < 0.05).

Search of `paper.tex`: zero CIs reported in Table 2, baseline_comparison, qualitative_examples, IDR table, BrowseSafe results, abstract, or conclusion. McNemar's test is mentioned only in commented-out sections.

`results/statistical_analysis.json` *does* contain CIs (recall 0.9427 [0.9042, 0.9770]). They simply weren't put into the paper. **Just paste them in.**

At the dataset sizes the paper uses, the CIs are not negligible:
- NotInject local (n=247), 0 FPs → Wilson 95 % CI on FPR = [0.0 %, 1.53 %].
- BrowseSafe (n=500 with claimed 96.7 % recall at 227 attacks) → CI on recall ≈ [93.4 %, 98.5 %].
- TensorTrust (n=500), if the row stays: 83.2 % accuracy → CI ≈ [79.5 %, 86.4 %].

### E.2 Results filter

The repo has an evaluation that ran θ=0.764 on the *previous* MiniLM-based model and showed catastrophic numbers (recall 36 % on deepset injections, FPR 13.4 % on deepset benign, recall 74.2 % on SaTML, recall 66.3 % on LLMail). Those numbers were not retracted; they live alongside the paper-supporting numbers. Any peer reviewer who clones the repo will find them. The paper should either explicitly explain why those results are not relevant (different model, different threshold-derivation), or the older results files should be moved into `results/legacy/`.

### E.3 Adaptive attack evaluation: missing as expected, but commented out

The repo's own `paper_gap_analysis.md` flags the absence of GCG/AutoDAN evaluation as the #1 critical gap. The paper has a section for adaptive-attack analysis but it is commented out (lines 481–493) with the note "preliminary results (n=36)." If those preliminary numbers exist, include them with the n=36 caveat and confidence intervals — that's better than silently removing the section.

### E.4 Multi-turn / cumulative bypass

§8.0.1 ("Limited Effectiveness Over Long Workflows") makes a quantitative claim — `>99 %` cumulative bypass over 50 messages at p=8.7 % per-message — but the 8.7 % per-message bypass rate is from the commented-out Guard-Agent bypass evaluation (`tables/guard_bypass_testing.tex`). The math is fine in isolation, but you can't cite a number whose source you've removed from the paper.

---

## F. Writing and structure

### F.1 Commented-out content

About **40 % of `paper.tex` is `%`-commented:** lines 106–136 (BIT vs MOF positioning), 303–338 (ablation, NotInject difficulty, trigger-word analysis, BIT component ablation), 401–463 (multi-agent attack evaluation, Guard bypass, OVON security, quarantine effectiveness, PeerGuard comparison), 481–493 (GCG/AutoDAN). The corresponding `tables/*.tex` files exist on disk and are not loaded.

The result: §4 ("System Design") introduces a Coordination Layer (Guard Agent, OVON, PeerGuard) and a Response Layer (Circuit Breaker, Quarantine), and §7 even has a sentence "Coordination Layer: Inter-agent trust exploitation and OVON protocol security (Section 7.5)" — but **§7.5 doesn't exist in the live paper**. The reader is told three layers will be evaluated; only one is.

**Either delete the layers from §1, the abstract, and §4 (and re-frame as a Detection paper), or restore the commented-out evaluations.** Mid-paper reframing inconsistencies will be caught immediately.

### F.2 Section numbering breaks

§7.4.5 "Embedding Space Analysis" appears as `\subsubsection` with no parent `\subsection` (because §7.4 — the ablation — is commented out). This will compile but will look like "Section 7.4.5" with no 7.4.x.

### F.3 Threat-model language

§3 "Out of Scope" lists "Model extraction attacks", "Feedback-based attacks", "Distributed attacks" without defining them. Two are jargon; one ("Feedback-based attacks") is non-standard. Either define them or omit them.

§3 also distinguishes "Input Control" (adversary can inject prompts via user chat or indirect sources) from "Compromised Agent" (adversary controls one agent), but the body never demonstrates the second capability — every evaluated attack is single-message direct injection. The Compromised Agent threat is only addressed in commented-out sections.

### F.4 Title and affiliation

- Title is fine but a bit long.
- Author block: `Abdelghafour El Bikha, John Jay College`. The internal `PROJECT_FEEDBACK_REPORT.md` lists "Abdel El Bikha, **Jennifer Marrero**" as the project group. If Jennifer contributed, she should be listed; if not, drop her name from the docs/.

### F.5 Citation hygiene

Every `\citep{}` in the paper resolves to a placeholder bib entry (see D.1). `\citep{wang2024peerguard}` and `\citep{lee2025prompt}` only appear in commented-out sections — `natbib` will warn about *unused* bib entries on every compile.

### F.6 Equations and notation

- Eq. (1) (ensemble score): defines $w_{emb}$ and $w_{pattern}$ but never reports their values.
- Eq. (2) (CosineSimilarityLoss): correctly stated but the symbol $\mathbf{e}_1, \mathbf{e}_2$ for the embeddings is not defined as the *output* of the fine-tuned encoder.
- IDR equation in §7.4.5 uses $\mu_{\text{benign-trigger}}$ etc. as the *centroid*, but the prose says "mean Euclidean distance between class centroids" — which would be a single number, not a ratio. The intended quantity is $d(\mu_a, \mu_b)$ (distance between centroids), not the mean of pairwise distances. Tighten the definition.

### F.7 Tables

- `qualitative_examples.tex`: text is truncated mid-sentence (`"What are the main differences between Python..."`). Untruncate or note the truncation.
- `baseline_comparison.tex`: mixes vendor-reported numbers and direct measurements without distinguishing methodologically — use a separate row group with explicit caveats.
- `dataset_summary.tex`: lists "deepset (full)" and "NotInject (local)" / "NotInject (HF)" as separate rows but also as a single dataset for accounting purposes. Be consistent.

### F.8 Typos and minor issues

- File name `paper/my_papper.md` has a typo ("papper").
- Abstract uses both 97.2 % and 97.3 %.
- §7.2 footnote ("our system achieves 98.5 % lower FPR than Lakera Guard and 46.3 % faster latency") combines a relative percentage with vendor-reported numbers — the comparison is mathematically OK but conceptually misleading.

---

## G. Novelty and contribution

The honest version of the contribution is:

> We construct a contrastive-pair training set that doubles the weight of "injection vs benign-with-trigger" pairs and use it to fine-tune `all-mpnet-base-v2`. We then train an XGBoost classifier on top of the resulting embeddings. The fine-tuned-embedding + XGBoost combination achieves competitive recall on standard prompt-injection benchmarks while keeping the false-positive rate on the trigger-heavy NotInject benchmark near zero, with sub-50 ms P95 latency on GPU.

That is a defensible *engineering* contribution and a reasonable undergraduate research output. It is **not** a "comprehensive three-layer defense framework for multi-agent LLMs" — that framing requires the Coordination and Response evaluations that have been commented out. Either (a) put those evaluations back and report them honestly even if the numbers are weaker, or (b) re-scope the paper to "An Injection-Aware Embedding Detector with Suppressed Over-Defense" and remove the multi-agent framing.

Compared to InjecGuard / PIGuard (Hao Li et al., arxiv 2410.22770), the actual delta is:
- A different base encoder (MPNet vs DeBERTa-v3).
- An ensemble decision (XGBoost on top of embeddings + regex patterns) rather than a single neural classifier.
- A pair-mining strategy that explicitly doubles the most-discriminative pair type.

This is a real engineering improvement worth writing up. It is *not* a fundamental paradigm shift, and the paper currently positions it as one (in the commented-out §2.5 BIT-vs-MOF section, which itself ends with the honest sentence "this work is **complementary** to InjecGuard rather than strictly superior").

---

## H. Recommended actions, priority-ordered

**P0 — must fix before submission anywhere:**
1. **Replace the bibliography** with real arXiv IDs and authors (D.1). Half a day with `arxiv.org` open in a browser.
2. **Decide what TensorTrust row says.** Either populate `data/tensortrust_cache.json`, run `evaluate_injection_aware_mpnet_full.py` on it, and put the real numbers in — or delete the row and every sentence that references it.
3. **Reconcile abstract / body / conclusion / Table 2 numbers.** Pick *one* run as canonical, paste its numbers everywhere.
4. **Rewrite §5.2's training-data paragraph** so it matches what `train_injection_aware_classifier.py` actually does and what `injection_aware_mpnet_metadata.json` records (413 malicious, 2667 safe, 570 trigger-benign).
5. **Either restore Sections 4.2/4.3 evaluations or delete the multi-layer framing** from the abstract, §1, §4, and §7.

**P1 — major weaknesses:**
6. Add CIs to every reported metric (E.1). The data is in `results/statistical_analysis.json`.
7. Remove or relocate stale results (`comprehensive_evaluation.json`, `benchmark_results_updated.json`) so a reviewer cloning the repo can't immediately produce numbers that contradict the paper.
8. Justify θ=0.764 on the new MPNet model with an actual validation curve (currently `evaluate_theta_764_model.py` runs a sweep — include the resulting plot or table).
9. Implement the IDR computation script if the IDR claim stays (§7.4.5).
10. Add the GCG/AutoDAN preliminary results back, with the n=36 caveat and CIs.

**P2 — polish:**
11. Fix the section numbering (F.2).
12. Define the threat-model "Out of Scope" terms (F.3).
13. Untruncate `qualitative_examples.tex` (F.7).
14. Decide the author list (F.4).
15. Either remove the unused PeerGuard / Lee-Tiwari citations or actually compare against them in §2.

**Estimated effort:** P0 + P1 ≈ 3–5 focused days of editing plus 1 evaluation run on TensorTrust. The model itself does not need to be retrained.

---

## I. What works well and should be kept

- The doubled-pair contrastive-fine-tuning idea (paper §5.2 contrastive-pair table) is genuinely the most interesting methodological choice in the paper. Lead with this.
- Using a quantitative over-defense metric (NotInject FPR) and the IDR concept (once it's actually computed) is a nicer framing than yet another accuracy bar chart.
- Reporting per-dataset metrics, latency distributions, and a baseline table is the right structure.
- The qualitative examples table communicates the over-defense story in one glance.
- The repository has unit tests and reproducible scripts — get them passing in CI and reference them in the paper.

---

## Appendix: files cross-referenced in this critique

- `paper/paper.tex` (lines cited inline)
- `paper/tables/per_dataset_metrics.tex`
- `paper/tables/baseline_comparison.tex`
- `paper/tables/dataset_summary.tex`
- `paper/tables/qualitative_examples.tex`
- `paper/references.bib`
- `paper/paper_gap_analysis.md` (existing internal review)
- `results/injection_aware_mpnet_evaluation.json`
- `results/latest_benchmark_results.txt`
- `results/benchmark_results_updated.json`
- `results/benchmark_results.json`
- `results/comprehensive_evaluation.json`
- `results/statistical_analysis.json`
- `models/injection_aware_mpnet/README.md`
- `models/injection_aware_mpnet/eval/similarity_evaluation_injection-detection-val_results.csv`
- `models/injection_aware_mpnet_metadata.json`
- `models/injection_aware_mpnet_classifier_metadata.json`
- `data/tensortrust_cache.json` (empty)
- `training/finetune_mpnet_embeddings.py`
- `training/train_injection_aware_classifier.py`
- `evaluation/evaluate_injection_aware_mpnet_full.py`
- `evaluation/final_evaluation_report.py`
- `evaluation/evaluate_theta_764_model.py`
- `benchmarks/benchmark_datasets.py`
- `docs/PROJECT_FEEDBACK_REPORT.md`
- `tests/BALANCED_MODEL_REPORT.md`

External sources used in the fact-check:
- Lee & Tiwari, "Prompt Infection" — https://arxiv.org/abs/2410.07283
- Chen et al., "StruQ" — https://arxiv.org/abs/2402.06363
- Chen et al., "SecAlign" — https://arxiv.org/abs/2410.05451
- Li et al., "InjecGuard / PIGuard" — https://arxiv.org/abs/2410.22770
- Si, Li et al., "PromptArmor" — https://arxiv.org/abs/2507.15219
- Fan, Li et al., "PeerGuard" — https://arxiv.org/abs/2505.11642
- Perplexity AI, "BrowseSafe" — https://arxiv.org/abs/2511.20597
- Lakera PINT benchmark — https://www.lakera.ai/blog/lakera-pint-benchmark
