# September 24: MPNet research decision

The T4 reviewed-span run completed and verified all six trials and three portable package replays. Fine-tuned seeds 17/42/73 scored 74/4, 4/2 and 71/5 TP/FP on the 207-page validation set; all three missed all five direct new-HTML attacks. Retain the verified A100 no-span HTML-correction run as the research comparison reference; do not promote the span-loss packages or claim a security gate. No application model pointer, deployment, or final holdout was changed. See `results/span_supervision_20260911/README.md` and `review_packet_20260924.md`.

The matched no-span T4 control inputs are frozen, but its runtime was terminated mid-run and a T4 retry at 19:57 UTC returned Service Unavailable after Colab showed a GPU usage limit. The partial log is not a control result. When T4 access returns, finish all six control trials, three package replays, and local verification before attributing any difference to the span loss.

# September 11: reviewed span auxiliary loss implemented; training pending

The user selected step 1: add targeted loss for known attack spans and quoted examples. `training/supervised_pilot.py` now supports optional, class-balanced auxiliary BCE on complete reviewed training insertions while retaining the page-max loss. It uses all covering windows with per-example averaging; other windows receive no invented labels. Zero coefficient preserves the original objective. Inference and validation-only selection are unchanged.

Prepared inputs: `results/span_supervision_20260911/inputs/`, coefficient 0.5, 10 positive and 5 quoted-negative spans, class weights 0.75/1.5. Train/validation data bytes match the completed HTML-correction run exactly. Existing six epochs and three seeds are retained. Reviews remain explicitly AI labels. Local gradient, zero-coefficient, class-balance, review/split/hash/context rejection and full-token preflight checks passed. See `results/span_supervision_20260911/implementation.json` and README. No new training, GPU allocation, weights or final-test evaluation occurred. The completed-run heartbeat stays paused.

Disk space is extremely low. Only one regenerated Python bytecode file and five fully verified redundant input archives were removed; all extracted research evidence and weights remain. Archive cleanup receipt: `results/html_coverage_correction_20260911/span_loss_archive_cleanup.json`. Older upload notebooks need their archives repackaged from preserved inputs and archive checksums updated before replay. Preserve all new weights durably before a future full GPU run. Dirty work and original staging are intact; no commit, push or publication.

# September 11: AI-reviewed HTML correction completed and verified

See `paper/HTML_COVERAGE_CORRECTION.md`. Fine-tuned seeds 17/42/73 detected 0/5, 1/5 and 1/5 new HTML validation attacks; both detections were the Uniqlo typo insertion. Seed 42 also flagged its quoted counterpart; all original pages were unflagged. Aggregate validation recall is 69.31/70.30/69.31%, FPR 4.72/2.83/3.77%, AUC 0.8251/0.8506/0.8577. These are validation-selected development results, not fresh-test evidence, and they do not demonstrate reliable attack detection or isolate a causal improvement from the data correction.

The authorized AI review is complete: 15 injection attempts and 30 provisional non-injections, same-assistant review, no human or blind-independent claim. All six trials completed 384 finite updates. The user-approved numerical overflow policy recovered every affected batch. Only 11/180 repeated observations of new training attacks selected a window containing the full insertion; supervision placement remains a possible failure mechanism. Every insertion was fully covered by the input windowing.

All 143 artifacts, 17 frozen inputs, 41,432 window hashes, selections and orders verified at 2026-09-11T22:53:56.362042+00:00. Three portable packages reproduced 621 validation decisions on A100; maximum local saved-feature/head error 1.19e-07. Full GPU encoder computation was not independently repeated locally. The executed four-code-cell notebook has no errors. Weights and package index: `results/html_coverage_correction_20260911/completed/mpnet_html_coverage_ai/INFERENCE.md`. No final test was included; six metadata-only reserved websites remain untouched. Historical family independence remains unknown.

A100 `mpnet-html-coverage-ai-a100` terminated at 2026-09-11T22:55:00.955019+00:00; backend confirmed zero assignments. The existing heartbeat is paused. No restart is needed for this completed run. Preserve dirty work and original staged diff `404de3b634a62eb9fceb16b48da2a054c4ae3edcbb1a5b665c2affbfb40bcd46`. No commit, push, publication or deployment.

# September 11: AI-reviewed HTML correction authorized and running

The user explicitly authorized Codex to review the 45 candidates and proceed. `results/html_coverage_correction_20260911/AI_REVIEW.md` and `ai_review.csv` record 15 injection attempts and 30 provisional non-injections, with no unresolved labels or disagreements. This is same-assistant AI review, not blind independent review or human sign-off. `ai_authorization.json` records the scope amendment; the old blank human sheet and stricter preflight remain historical records, not blockers for this authorized exploratory path.

Frozen run inputs: `results/html_coverage_correction_20260911/ai_run/`. Training stays balanced at 256 rows (128/128), replacing 20 legacy rows with 10 direct, 5 original and 5 quoted examples. Validation has 207 rows (101 attacks/106 negatives). All raw token hashes passed. Architecture and six-epoch schedule are unchanged, with seeds 17/42/73 and matched frozen controls. No final test is exported or scored; six reserved websites remain metadata only. Historical family overlap remains unknown and validation position/category are coupled.

A100-SXM4-40GB and pinned base weight hash were verified at 22:26 UTC; all six trials are running. A100 session: `mpnet-html-coverage-ai-a100`. Notebook: `notebooks/colab_html_coverage_ai.ipynb`; execution log: `results/html_coverage_correction_20260911/execution.log`. The notebook installs pinned packages, verifies base weights and A100 identity, trains all six trials, exports three complete packages, replays all 207 validation pages, then writes `artifact_manifest.json` and `completion.json`. Do not infer success from the CLI exit code alone. Retrieve all artifacts with `/Users/goodwiinz/.local/share/uv/tools/google-colab-cli/bin/python /tmp/retrieve_html_correction.py` (streams files, verifies hashes and checks space), run `scripts/run_html_coverage_correction.py verify --output <retrieved directory>`, inspect notebook errors, report every seed and new-page conditions, then stop this GPU and pause the existing heartbeat.

Storage is tight (~2.1 GiB free after removing only regenerable virtual-environment Python bytecode; cleanup receipt is `bytecode_cache_cleanup.json`). `lossless_storage_receipt.json` records byte/hash-preserving native compression and copy-on-write clones of existing evidence; nothing unique was deleted. Check remaining space before retrieval and avoid a second large archive. The original staged diff hash remains `404de3b634a62eb9fceb16b48da2a054c4ae3edcbb1a5b665c2affbfb40bcd46`. No commit, push, publication or deployment.

# September 11: HTML coverage correction prepared; human labels pending

See `paper/HTML_COVERAGE_CORRECTION.md`. The new review packet is `results/html_coverage_correction_20260911/REVIEW.md`; label sheet `human_review.csv`. There are 45 unscored candidates (original/direct/quoted triples) across 15 website groups: 30 training-group candidates and 15 validation candidates. The proposed fitting plan uses balanced pairs from training groups: 10 direct, 5 originals and 5 quoted negatives replace 10 positive/10 negative legacy rows, keeping 256 rows and 128/128 balance. Ten training-group counterparts stay review-only. All 15 validation candidates would augment the original 192, pending labels. Six additional website groups remain metadata-only reserves, with no constructed test examples.

All 45 human label fields are blank. The pending question asks who will provide independent human labels; no answer has been received. Do not invent human review or launch training from construction labels. Uncertain or disagreeing reviews require adjudication before exporting the replacement set. No new GPU allocation, model training, scoring or threshold selection occurred.

Runner `scripts/prepare_html_coverage_correction.py` provides prepare/check/self-check. Frozen protocol, source receipts, replacement IDs, exclusions, exact insertion spans and verification are under `results/html_coverage_correction_20260911/`. Checks exclude known duplicates against 8,376 prior inputs and keep new websites/categories apart, but historical family provenance is still unknown; max retained lexical similarity is 0.883. The small validation packet also couples category and placement, so do not claim isolated placement effects. The strict unseen-family benchmark remains unresolved.

The initial 18-group target shrank to 15 under unchanged eligibility checks. A recorded pre-training amendment selected 20 of 30 training candidates to avoid source-label imbalance. Earlier plans are preserved in `results/html_coverage_correction_20260911_preparation_v1/`; no raw pages or labels changed. Continue with human review and provenance decisions, then finalize one controlled data correction using the existing MPNet/base controls and all three seeds. Preserve the old diagnostic and holdouts as consumed evidence. Dirty work and staging are preserved; no commit, push, publication or deployment.

# September 11: external AI-reviewed diagnostic completed and verified

See `paper/EXTERNAL_DIAGNOSTIC.md`. All six models (three fine-tuned seeds and matched frozen controls) missed all three inserted instructions and flagged none of the three original pages. All 18 original/injected page-score pairs are exactly equal. Every insertion was fully scored in the final window; each winning window was elsewhere. Truncation does not explain these misses, and fine-tuning did not improve decisions on this sample. Representation, data coverage and source/threshold transfer remain hypotheses rather than isolated causes.

This is an exploratory six-page / three-dependent-pair AI-labeled diagnostic. It does not satisfy the strict human-reviewed unseen-family study. The human CSV is unchanged and the strict preflight remains blocked. These six pages are now consumed diagnostic data; do not use them to select a model or threshold and then claim fresh evaluation. Obtain human labels and source groups, build separate training/validation examples for long HTML with short inserted instructions, and reserve a new untouched final holdout.

Runner: `scripts/run_external_diagnostic.py`; evidence: `results/generalization_20260911/diagnostic/`; notebook: `notebooks/colab_external_diagnostic_output.ipynb`. All 56 frozen file hashes, seven artifact hashes, 2,170 independent window-token hashes, 36 saved head decisions and 3/3 notebook cells verified. Maximum local head replay error: 4.47e-8. GPU encoder inference was not independently repeated locally.

Before scoring, stale NumPy imports and incompatible optional TorchVision were resolved by fresh subprocesses and the prior text-only dependency setup. No checkpoints or thresholds changed. A100 session `mpnet-external-diagnostic-a100` was terminated after retrieval and verification; backend confirmed zero assignments at 20:36:20 UTC. The heartbeat is paused. Dirty work and original staging are preserved; no commit, push, publication or deployment.

# September 11: AI review completed; human review remains pending

At the user's request, Codex reviewed the six candidate pages and saved `results/generalization_20260911/AI_REVIEW.md` and `ai_review.csv`. Proposed labels: three injection attempts and three provisional non-injections. All raw hashes and exact pair differences were verified. The inserted instructions redirect the assistant into file-renaming code, Spanish output or reversed output. The original pages contain ordinary navigation, shopping/travel content and advertisements; negative labels remain provisional.

This is **AI annotation by the same assistant that constructed the packet**, not independent blind review or human sign-off. The human label sheet is unchanged, and all AI rows explicitly say `human_review_attested=no`. No inference, GPU use, model selection or threshold changes occurred. All three positives share an end-of-body `aside` insertion, so this small paired sample does not establish broad detection robustness. Human labels and the source-family scope decision remain unresolved; the previous pending preparation details below still apply.

# September 11: independent generalization preparation awaits human input

The latest request is to proceed with a new human-reviewed, unseen-website/template evaluation. See [GENERALIZATION_PREPARATION.md](GENERALIZATION_PREPARATION.md). Three fine-tuned weights and three matched frozen heads are frozen; thresholds unchanged. A source audit found **0/448 trusted website IDs, 0/448 template IDs, and no canonical or og:url metadata** in development inputs. Do not claim strict family independence from low text similarity or external dataset names.

A small, concrete **six-page review packet from three Mind2Web website groups** is in `results/generalization_20260911/REVIEW.md`; human labels are blank in `human_review.csv`. Original pages are paired with locally inserted published BIPIA instructions. This is a review-workflow sample, not a final benchmark. Parent-page overlap checks against 8,370 prior inputs passed, but they do not prove family novelty. Base encoder revision and upstream weight SHA are pinned; base weights need download/hash verification only when evaluation becomes authorized and ready.

**No new scoring, training or GPU allocation occurred.** Preflight hash checks pass and return expected exit 2 because human labels and family-provenance evidence are missing. Two user input panels ask who will review and whether the evaluation should retain strict unseen-family evidence or be narrowed to an external-dataset test with unknown overlap disclosed. Do not invent human review or treat elapsed time as an answer. Once resolved, freeze a sufficiently broad final sampling frame and labels before any A100 scoring. Use all three seeds and original thresholds; retain matched frozen controls. Prior consumed holdouts remain unavailable for tuning.

Run `HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_generalization_review.py check` to refresh the preflight. Self-check and diff check passed; original staged changes are preserved. No commit, push, publication or deployment. The existing completed-run heartbeat stays paused; there is no GPU job to monitor.

# September 11: supervised schedule experiment completed and verified

Read [SUPERVISED_SCHEDULE.md](SUPERVISED_SCHEDULE.md) for the current result. The A100 six-epoch follow-up and deterministic artifact recovery completed successfully. On the same 453-page holdout (217 attacks, 236 non-injections), revised seeds 17/42/73 achieve AUC **0.8341/0.8732/0.8660**, recall **69.12/66.36/67.74%**, and FPR **2.54/4.66/2.97%**. Prior fine-tuned recall on these same pages was **59.45/4.61/45.16%**. The declared directional criterion passes, but FPR increases for two seeds, nominal FPR upper bounds exceed 5%, and no standalone security-gate or original contrastive-over-defense claim is established.

Three complete portable weight packages are indexed in `results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay/INFERENCE.md`. Local verification passed at **10:01:24 UTC**: all **238 recorded hashes**, twelve model/threshold freezes, token coverage, saved predictions, validation selections, optimizer and window logs, and recorded A100 package replays. The nine-code-cell executed notebook is `notebooks/colab_supervised_schedule_replay_output.ipynb`; every cell completed without error. All six prior/new fine-tuned packages replayed all 192 validation pages with identical decisions (1,152 predictions, max error 5.96e-8). The vectors/heads are replayed locally; full GPU encoder computation is not independently repeated by the local verifier.

Preserve the two recovery disclosures. The original schedule stopped on gradient overflow after two epochs and before evaluation; the amended protocol uses bounded same-batch AMP scale-backoff retries. Every trial completed 384 finite updates with no batch omitted. The first amended runtime later disappeared during evaluation. Its recovered run exactly reproduced all six original checkpoint files, the two saved new checkpoint files (17/42), and all 36 captured validation summaries. Seed 73's interrupted checkpoint was not retrieved, so there is no byte comparison to that lost file. No scientific choices changed after test output. The 453-page test is now consumed; future choices require validation and a separate untouched final holdout.

All evidence and 14 stage logs are local. Run `HF_HUB_OFFLINE=1 .venv/bin/python scripts/verify_supervised_schedule.py results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay` for verification. Failed/interrupted artifacts remain in their own directories. The CLI scope-refresh repair retains the existing OAuth grant; receipt `results/supervised_schedule_20260911/colab_scope_repair.json` records it. Colab session `mpnet-schedule-replay-a100` was terminated after safe retrieval and verification. Backend check at **10:02:00 UTC** found **zero active assignments**, recorded in `runtime_closed.json`. No restart or GPU allocation is needed for this completed experiment.

The figures and affected PDF pages (1, 10–13) were visually inspected; the 14-page manuscript PDF was rebuilt with Tectonic without unresolved warnings. The `colab-experiment-updates` heartbeat is paused. All requested work for this experiment is complete. Dirty work and original staging are preserved; no commit, push, deployment or model publication was performed.

# September 11: supervised A100 pilot completed and verified

The user approved supervised fine-tuning, explicitly requested an A100, and supplied the official Google Colab CLI repository. See `paper/SUPERVISED_PILOT.md`. CLI 0.6.0 is installed with compatible `jupyter-kernel-client 0.15.0`. A100-SXM4-40GB was verified from inside session `mpnet-supervised-a100`, which has now been terminated after evidence verification. Execution session **75921 completed successfully**. Fine-tuned AUC for seeds 17/42/73 is **0.8353/0.4668/0.8584**, recall **61.88/10.31/51.12%**, FPR **2.23/8.48/0.45%**. Two seeds improve both recall and FPR; seed 42 degrades AUC and exceeds the exploratory holdout FPR budget. No stable improvement or security-gate claim.

The notebook is `notebooks/colab_supervised_pilot.ipynb`, launched with `colab exec -s mpnet-supervised-a100 --timeout 14400 -f notebooks/colab_supervised_pilot.ipynb`. It runs both arms for seeds 17/42/73, freezes all selections, then evaluates 447 new BrowseSafe holdout pages. Final frozen inputs are in `results/supervised_pilot_20260911/pilot/`. Do not use the preserved pre-review preparation. The earlier browser-created A100 was released.

All six checkpoints and all other evidence have been retrieved under `results/supervised_pilot_20260911/completed/mpnet_supervised_pilot/`. `scripts/verify_supervised_pilot.py` passed at 06:37:07 UTC: **81 hashes**, raw token coverage, linear-head and historical XGBoost replay, validation selections, paired training orders, GPU identity, counts, and nominal bootstrap intervals. GPU encoder inference/training was not independently repeated. The executed notebook `notebooks/colab_supervised_pilot_output.ipynb` has all seven expected completion outputs and no errors; CLI 0.6.0 leaves execution-count fields null, so preserve it as emitted. The verified figure and manuscript pages were visually inspected; `paper/paper.pdf` was rebuilt with Tectonic. The completed experiment heartbeat is paused. Original staging remains unchanged; no commit, push, deployment or model publication. The final holdout is now consumed; another model choice requires validation and a new untouched final holdout.

Disk filled during artifact retrieval. Three redundant prior download archives (`audit_checkpoints.tar`, `audit_evidence.tar.gz`, `window_followup_results.tar.gz`) were removed only after all archived files matched preserved extracted evidence. `results/supervised_pilot_20260911/duplicate_archive_cleanup.json` records the checks; no extracted evidence was removed. Individual file downloads resumed without retraining or rescoring.

# September 11: validation diagnosis and controlled follow-up completed

See `paper/VALIDATION_DIAGNOSIS.md` and `notebooks/windowed_followup_results.ipynb`. The Colab windowing experiment finished for classifier seeds 17/42/73, on 919 unused BrowseSafe rows screened before scoring. All 30 recorded checksums, window token coverage, saved classifier predictions and validation-only thresholds verified. Original base seed-42 validation scores match exactly. AUC gains of 0.007–0.010 have nominal intervals spanning zero; FPR rises to 36–60%. No useful improvement or security-gate readiness is claimed. This follow-up test is now consumed; reserve another independent final holdout before another correction. No staging, commit, push, deployment or model publication.

## September 11, 2026 — Colab experiment complete

All three variants completed. Evidence, checkpoints, and verification are in `results/colab_audit_20260911/`; see `paper/COLAB_RUN.md`. The active manuscript and PDF now include the measured results and limitations. Base/uniform/double accuracy: 69.03/62.55/63.67%; attack recall: 45.55/48.30/64.20%; FPR: 11.28/25.50/36.78%. This run does not support reduced over-defense. Original user staging remains unchanged.

> **Update — September 10, 2026:** The claim audit and editorial repairs are in [CLAIM_AUDIT.md](CLAIM_AUDIT.md). The active paper now treats saved scores as contaminated historical observations. The earlier checklist below is historical: merely reconciling numbers or running the old checkpoint will not produce valid held-out evidence. No retraining has been completed; use the audit's remaining experimental requirements before restoring performance claims.

# Handoff to Claude Code

This document captures what was done in the audit pass on `paper-fix-april-2026`
and what's still open. Pair with `CRITIQUE.md` (findings) and `FIX_PLAN.md`
(full remediation plan).

## What's already done on this branch

- **`paper/CRITIQUE.md`** — full reviewer-style critique with claim-vs-code
  audit, internal-consistency check, methodology misrepresentation flags,
  bibliography fact-check, and prioritised recommendations.
- **`paper/FIX_PLAN.md`** — runbook with decisions, P0/P1/P2 phases,
  per-task file paths, effort estimates, acceptance criteria, and a final
  verification checklist.
- **`paper/paper_annotated.tex`** — copy of the original `paper.tex` with
  inline `\critnote{}` margin comments referencing every finding.
- **`paper/references.bib`** — fully rebuilt with real arXiv IDs and
  authors. The 8 placeholder entries (e.g. `2501.00001`,
  `author = "Anonymous"`) are gone. Citation keys in `paper.tex` were
  updated to match.
- **`paper/paper.tex`** — partial rewrite executing the Detection-only
  re-scope (D1=b in FIX_PLAN.md):
  - New title: *"Injection-Aware Sentence Embeddings: Suppressing
    Over-Defense in Prompt Injection Detection"*.
  - Abstract rewritten: removed "comprehensive three-layer framework"
    language, removed TensorTrust, fixed 97.3% / 38.1↔43.3 ms / BrowseSafe
    97.8↔98.2 internal contradictions.
  - §1 Introduction restructured around the actual contribution
    (over-defense suppression via contrastive embedding training).
  - §3 Threat Model rewritten with proper out-of-scope definitions and
    explicit acknowledgement that adaptive attacks and multi-turn attacks
    are out of scope.
  - §4 collapsed: §4.2 (Coordination) and §4.3 (Response) merged into a
    single "future work" paragraph at the end of §4.
  - §5 Methodology rewritten to match the **actual** training corpus per
    `models/injection_aware_mpnet_metadata.json`
    (2,667 safe / 570 trigger-benign / 413 malicious — replacing the
    previously-claimed 2,800 / 2,800 / 1,400). Spearman 0.904 attribution
    corrected to "step 12,500, epoch 3.57; training stopped early".
  - Conclusion rewritten with corrected, internally-consistent numbers,
    code-release URL, and the dangling "Limited Effectiveness Over Long
    Workflows" paragraph removed (its underlying number had no source).
- **`results/legacy/`** — 9 stale results files moved here so a reviewer
  cloning the repo can't reproduce contradictory numbers from older
  models. README in that directory explains the provenance.
- **`evaluation/evaluate_mpnet_on_tensortrust.py`** — new script that runs
  the MPNet detector on the locally-cached TensorTrust dataset and writes
  `results/tensortrust_mpnet_evaluation.json`. **Not yet run** — the
  sandbox couldn't load the macOS-built venv.

## What's still open (P0 tail)

1. **Run the three new evaluations on your machine:**
   ```bash
   cd /Users/goodwiinz/development/prompt-injection-defense
   source .venv/bin/activate
   PYTHONPATH=. python evaluation/evaluate_mpnet_on_tensortrust.py
   ```
   Then update Table 2 (`paper/tables/per_dataset_metrics.tex`) with the
   real TensorTrust row, or remove it.

2. **Reconcile per-dataset numbers in `paper/tables/per_dataset_metrics.tex`**
   to match `results/latest_benchmark_results.txt` (the canonical run):
   - SaTML row recall: change 98.8 → 96.0
   - BrowseSafe row recall: change 96.7 → 94.1, FPR 0.5 → 0.0
   - Drop the `NotInject (HuggingFace)` row OR run the HF eval (loader
     exists at `benchmarks/benchmark_datasets.py:load_notinject_hf_dataset`)

3. **Read through §7 (Results)** for any leftover references to commented-
   out subsections. `\ref{sec:browsesafe}` and `\ref{sec:limitations}`
   labels referenced in §3 may need to be added in §7.6 and §8.

## What's still open (P1)

P1-1, P1-2, P1-3, P1-4, P1-5 from `FIX_PLAN.md` — all unchanged. None
were started in this pass because they all require running Python in the
proper venv:
- Add Wilson CIs to all reported metrics (`scripts/wilson_ci.py` helper
  is described in FIX_PLAN.md).
- Implement IDR computation script and re-fill the §7.4.5 table with
  real numbers (currently 2.31 / 0.67 with no script behind them).
- Run threshold sweep + figure for §5.1 to justify $\theta=0.764$ on the
  new MPNet model.
- Restore preliminary GCG/AutoDAN section (lines 481–493 of original
  paper.tex) with the n=36 caveat.
- NotInject HuggingFace evaluation (see #1 above).

## What's still open (P2 polish)

All 12 items from `FIX_PLAN.md` Phase 3 — none were touched.

## Branch / commit / push

The branch `paper-fix-april-2026` exists locally with all changes staged
but **not yet committed** because the sandbox couldn't remove
`.git/index.lock`. Run `paper/push.sh` (created next to this file) on
your Mac to commit and push.

## Decisions still pending

- **D1** is currently set to (b) Detection-only re-scope. If you actually
  want the original three-layer framing, revert this branch and execute
  D1=(a) instead — `git checkout main` and start over.
- **D2** TensorTrust: pending until you run the eval (#1 above).
- **D3** θ=0.764: kept as-is; needs the validation-curve figure to
  justify (P1-4).
- **D4** Author list: solo "Abdelghafour El Bikha" with a TODO comment
  in `\title{...}` block. Confirm whether Jennifer Marrero should be
  added.
- **D5** Code release: conclusion now points to
  `https://github.com/goodwiins/prompt-injection-defense`. Make sure that
  repo is public on submission.
