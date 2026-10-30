# Validation diagnosis and controlled follow-up — September 11, 2026

The diagnosis uses the completed audit's **2,286 training and 742 validation inputs**, saved validation scores, and actual MPNet tokenizer. No fresh holdout predictions were used to choose the intervention. The old test results were already known from the preceding audit; they are not a fresh final test.

## Confirmed findings

| Validation source | Inputs | Above 382 content tokens | Base ROC-AUC | Uniform ROC-AUC | Double ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BrowseSafe | 410 | 363 | 0.515 | 0.486 | 0.554 |
| deepset | 110 | 0 | 0.923 | 0.916 | 0.978 |
| SaTML | 154 | 5 | Undefined: attack-only | Undefined | Undefined |
| Synthetic education template | 68 | 0 | Undefined: non-injection-only | Undefined | Undefined |

The encoder's 384-token limit includes two special tokens, leaving 382 content tokens. BrowseSafe's median validation page contains 1,882 content tokens. The first window discards most of many pages before any classifier sees them.

**Truncation removes real instructions.** A fixed lexical locator flags 52 attack-labeled BrowseSafe validation inputs with instruction phrases after the first window and none inside it; all three variants miss 39 of those inputs. Manual inspection confirms explicit task-redirection or data-extraction instructions outside the window in `train/1038`, `train/1049`, and `train/1075`. This is a locator-assisted finding, not a complete attack-span annotation or proof that every flagged string is an attack. Two non-injection pages also contain such phrases in educational discussions.

**HTML extraction removes other payloads entirely.** In 21 attack-labeled validation pages, the locator matches raw HTML but not extracted text. Inspected examples include `train/74` (hidden input value), `train/85` and `train/135` (comments), and `train/164` (data attribute). The original extractor drops these locations. More windows cannot recover text already removed. The [source dataset card](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench) describes raw-HTML classification, hidden payload locations, and machine-generated labels; the old extracted-text experiment is a narrower, lossy task.

**The synthetic trigger data have very little structural variety.** All 136 training trigger examples come from three known template families (68 technical, 34 recipe, 34 manual). All 68 validation trigger examples come from one education family. Some substitutions are ungrammatical, such as using a noun where the template requires a verb. Validation false positives are 2/68 for base, 22/68 for uniform, and 15/68 for double. Known families were split apart; that does not make them representative of natural benign requests.

**Source and label are confounded.** SaTML and LLMail contribute only attack labels, while synthetic trigger templates contribute only non-injection labels. No LLMail group lands in validation. A diagnostic score using only each source's training attack prevalence obtains validation ROC-AUC **0.779**. This shows that pooled performance can reward source recognition; it does not prove any particular model learned that shortcut. BrowseSafe's within-source results are therefore particularly important.

**Exact isolation does not remove lexical near-duplicates.** Character 3–5-gram TF-IDF, fitted on training text, identifies 97 validation inputs with nearest-training cosine similarity at least 0.90. Seven nearest pairs have opposite labels. Those are review candidates, not automatic label errors: adding an attack to an otherwise similar page can legitimately change its label. The original split isolated exact text, truncated token inputs, and known groups, but not these lexical families.

## Label review and limits

The deterministic review sample contains 21 unique validation cases: up to three errors per source/label, plus three tail-only instruction candidates. Predictions, full extracted text, model-visible heads, token counts, and nearest training IDs are saved in `results/validation_diagnosis_20260911/review_sample.json`.

- deepset `train/199` is a harmful or prejudiced question labeled non-injection. This is not necessarily a mislabeled injection; **non-injection is not a general harmlessness label**.
- deepset `train/283` requests a new task; whether it overrides an authorized task depends on missing surrounding context. Its standalone label is ambiguous.
- SaTML `attack/28` is an ordinary-looking introductory phrase. Competition provenance alone does not establish an attack in the isolated first message. Context is missing; it was not relabeled.
- BrowseSafe `train/1072` quotes attack instructions in an educational transcript, and `train/1880` discusses prompt-injection risk. A keyword occurrence alone is not sufficient to change their non-injection labels.
- Several sampled BrowseSafe attack pages have no match for the English locator. That does not establish an incorrect label: attacks can be indirect, multilingual, in removed HTML, or outside the inspected excerpt.

No automatic relabeling occurred. This is a bounded analyst review, not an independent annotation study. It establishes specific input-loss defects and sampling limitations; it does not establish the sole cause of fine-tuning degradation or embedding collapse.

## One-factor Colab experiment

The correction changes **inference coverage only**: frozen base MPNet and the unchanged XGBoost training recipe score every overlapping 382-token window (96-token overlap), taking the maximum attack score per document. The control uses the first window. Training remains on the original first-window embeddings and labels; the experiment does not propagate document labels to every training window. The original HTML extractor remains unchanged to keep this a controlled test.

Seeds **17, 42, and 73** vary XGBoost subsampling; they do not vary encoder fine-tuning, because the encoder is frozen. Both methods use the same fitted classifier for a given seed. Each method/seed chooses its threshold solely by validation balanced accuracy, with the existing tie rule. No seed or method is selected after seeing results. Every threshold and classifier is frozen before final holdout inference.

The final holdout comes from fixed BrowseSafe official-test indices **600–1599**, beyond the old experiment's 0–599 prefix. Screening against all old raw inputs and within the new holdout excludes exact text, first-window token duplicates, and character TF-IDF cosine similarity at least 0.90. This leaves **919 inputs: 445 attacks and 474 non-injections**, with 81 exclusions. The protocol and manifest were frozen before validation or holdout scoring. The new holdout is unused in this experiment's fitting and selection; it is not a newly authored benchmark, a pretraining-contamination guarantee, or a guarantee of independent website families.

Primary endpoint: the paired change in holdout ROC-AUC for each seed. The predefined consistency criterion is a positive change for all three seeds. Recall, false positives, balanced accuracy, and average precision are reported for every seed even if the criterion fails. This criterion is descriptive; it does not establish statistical significance or production suitability. The holdout covers BrowseSafe only and cannot establish improved NotInject performance.

Protocol: `results/window_followup_20260911/protocol.json`. Holdout SHA256: `a19ed336bf3392ff19e9b9f9656e219dcabb8072b7d9b88ce4f93079699dd8b5`.

Source: `training/windowed_followup.py`, SHA256 `163d0f6ed0e56ae9ddab9873a68fa1374351466c8cfa7200d7ea676c03e288ed`.

Portable notebook: `notebooks/colab_windowed_followup.ipynb`; companion frozen inputs: `notebooks/window_followup_inputs.tar.gz`. The live run is appended to the [existing Colab notebook](https://colab.research.google.com/drive/1hnI9NUTIdHjk1TBAG6TbeOitbRQYfFxh), keeping earlier outputs intact.

## Completed Colab result

**Completed and independently verified.** Windowing alone did not produce a useful improvement. The predefined positive-AUC direction criterion holds for all three seeds, but every nominal paired 95% interval includes zero. False-positive rates rise substantially.

| Seed | Base AUC | Windowed AUC | Base recall | Windowed recall | Base FPR | Windowed FPR |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 0.5398 | 0.5471 | 17.08% | 51.24% | 12.24% | 45.99% |
| 42 | 0.5330 | 0.5430 | 11.24% | 43.15% | 6.96% | 36.08% |
| 73 | 0.5331 | 0.5412 | 30.11% | 65.39% | 22.78% | 60.34% |

The nominal paired, class-stratified bootstrap uses 1,000 resamples (seed 20260911). It assumes independent documents and does not account for unknown website/template groups. AUC delta intervals:
- Seed 17: +0.0073, nominal 95% interval [-0.0305, +0.0502].
- Seed 42: +0.0100, nominal 95% interval [-0.0305, +0.0536].
- Seed 73: +0.0080, nominal 95% interval [-0.0326, +0.0532].

Pooled validation AUC improved more (base 0.759–0.774, windowed 0.800–0.806) than fresh within-BrowseSafe AUC (base 0.533–0.540, windowed 0.541–0.547). The source mix differs, so these pooled and within-source values are not directly comparable; the distinction reinforces the need for source-specific reporting.

Evidence: `results/colab_window_followup_20260911/window_followup/`. All 30 recorded checksums matched; 5,978 validation and 11,618 holdout window inputs were checked against the tokenizer. Saved classifiers reproduced scores and thresholds. Seed 42 reproduced the original base validation scores exactly. Full GPU encoder inference was not independently repeated.

Archive SHA256: `f54ae7029b2ab11e0e5a49df1f16d94d622159c506f2862066299fb867579e97`.

Recheck with:

```sh
.venv/bin/python scripts/verify_windowed_followup.py results/colab_window_followup_20260911/window_followup
```

The executed report notebook is `notebooks/windowed_followup_results.ipynb`. Its displayed values and comparison figure were inspected. The separate portable Colab launcher is schema/syntax validated; the stages were executed in the existing live notebook, not rerun from a clean runtime.

**Next decision:** retain the base model as a research reference. Before any further encoder training, repair and audit HTML input preservation, then review context-dependent source labels. This completed holdout is now an evaluated test and must not be reused as an untouched final test for another correction. No model was deployed or published.
