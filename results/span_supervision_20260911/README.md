# Reviewed span loss: T4 experiment complete

The existing page-max BCE is retained. A fixed auxiliary coefficient of 0.5 applies to full windows containing reviewed training insertions. Ten direct attacks use positive labels; five quoted examples, including their complete framing, use negative labels. Attack/quoted class weights are 0.75/1.5 so both classes contribute equal total weight. Multiple covering windows are averaged per example, then the auxiliary sum is divided by document batch size. Other windows receive no inferred span label.

This preserves the exact 256 training and 207 validation pages. The prior six-epoch settings and three seeds are retained in the prepared protocol. Input checks reject non-training rows, incomplete or uncertain reviews, changed hashes, cut framing and coverage mismatches. Both frozen and fine-tuned arms support the optional loss; inference and threshold rules are unchanged. With no configured auxiliary coefficient, the original objective is retained.

The implementation is in `training/supervised_pilot.py`; `scripts/run_html_coverage_correction.py prepare-spans` produces a fresh frozen input directory. Training revalidates those annotations against the raw pages and reviews. Per-step span diagnostics record the page and auxiliary losses, exact supervised window IDs, labels and weights. The verifier checks them against the frozen training order and combined optimizer loss.

The amended T4 run completed on 2026-09-24. All six models trained for six epochs, the three fine-tuned inference packages replayed all 207 validation pages with identical decisions, and local verification passed for 18 frozen input hashes, 151 artifact hashes, and 384 finite optimizer updates per trial. The final test was not evaluated. These are AI annotations, not human or blind independent review; this is a small exploratory sample.

| Seed | Frozen TP / FP | Fine-tuned TP / FP | Fine-tuned recall / FPR |
| --- | --- | --- | --- |
| 17 | 10 / 4 | 74 / 4 | 73.3% / 3.8% |
| 42 | 10 / 5 | 4 / 2 | 4.0% / 1.9% |
| 73 | 18 / 5 | 71 / 5 | 70.3% / 4.7% |

TP is out of 101 positive validation pages; FP is out of 106 negatives. All three fine-tuned models missed all five direct attacks in the new HTML cohort. The earlier A100 run without span loss detected 70, 71, and 70 positives across the same validation pages; the hardware change means that comparison does not isolate the auxiliary loss. The new span-loss variant is too variable to replace that prior MPNet model.

Research decision: retain the verified A100 no-span HTML-correction run as the comparison reference. Do not promote any T4 span-loss checkpoint or claim a deployment-ready detector. No application model pointer has been changed.

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python tests/test_supervised_pilot.py
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_html_coverage_correction.py verify --output results/span_supervision_20260911/inputs
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_html_coverage_correction.py verify --output results/span_supervision_20260911/completed/mpnet_span_t4
```

The original A100 inputs remain frozen. The separate `t4_inputs/` protocol changes only the GPU requirement and records the parent protocol hash. Its SHA-256 is `60b5d2202a191887ee626f4d6b79ae2602549c6caf2ca19022d87d9ce5324b14`. After training, `t4_packaging/` amended only the inference script's A100 labels and replay assertion for T4. The T4 training-script hash remained frozen.

The complete weights, selections, histories, replay outputs, manifest, and local verification are under `completed/mpnet_span_t4/`. `completion.json` records six models and three replays; `verification.json` records the per-cohort results and the local replay limit. The archive transferred from Colab matched SHA-256 `c7f9e55fe0c70ddb7083fd0b8d649241e4347e30293ffde42cd9e5fc2f9bbeec` before extraction. The remote artifact manifest SHA-256 is `2bf5be68609aa37a8d70c8368a12f9d417cbb98d7afc603a43685fc3e6c7cf79`.

## Matched no-span T4 control: blocked by GPU limit

`no_span_t4_inputs/` freezes the same 18 files, 256 training pages, 207 validation pages, model revision, six-epoch schedule, three seeds, two arms, and checkpoint/threshold rules. Its sole training change is span-loss weight `0` instead of `0.5`; the protocol SHA-256 is `0272be71356729ac6999ecc94ea7fae620db39ec560516e71ee5b989c62b9e2d`. Local input verification and the focused supervised-pilot tests passed. `review_packet_20260924.md` records a provisional AI assessment of all five missed direct HTML validation cases and all 15 training spans.

The T4 control began training, but Colab terminated the VM during fine-tuned seed 17 before any complete paired result could be transferred. T4 assignment retries failed again at 19:57 UTC, and the Colab notebook explicitly reported GPU usage limits. `no_span_t4_attempt_20260924.json` and its adjacent `.train.log` preserve the last confirmed stage. Partial epoch scores do not isolate the auxiliary loss. The final holdout remains sealed.

When T4 access returns, rerun the frozen control inputs and verify all six selected trials, three portable package replays, and the local artifact manifest before comparing per-seed recall/FPR. `no_span_t4_cache_provenance.json` records hashes for reusable base-encoder features from the completed span run; these features depend on the common base model and pages, not on the auxiliary loss.
