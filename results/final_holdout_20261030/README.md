# Final holdout — frozen October 30, 2026

This is the test set for the next MPNet retraining. It was frozen **before** any new training, and nothing has been scored on it. Score it once, after the model, threshold and every other choice are final.

Produced by `scripts/freeze_final_holdout.py prepare` (a copy is saved here as `freeze_final_holdout.py`). Verify integrity at any time:

```sh
.venv/bin/python scripts/freeze_final_holdout.py check
```

## Contents

| Slice | Label | Rows | Label basis |
|---|---|---:|---|
| BrowseSafe test, rows 2,600–3,679 | 1 | 497 | Source page label |
| BrowseSafe test, rows 2,600–3,679 | 0 | 473 | Source page label |
| SaTML attack chats (11 reserved teams) | 1 | 503 | First user message of an attack chat |
| LLMail Phase 2 (12 reserved teams) | 1 | 188 | Unique submission whose attack made the tool call (`api_triggered`) |
| LLMail false-positive-test emails | 0 | 116 | Benign emails written by the challenge authors |
| **Total** | | **1,777** | |

LLMail rows use the challenge's own input format, `Subject of the email: {subject}.   Body: {body}`, for both attacks and benign emails. Earlier runs used the body only. Future training on LLMail should use the same format.

## How rows were selected

- **BrowseSafe:** every row in test rows 2,600–3,679. Rows 0–2,599 were used up by the September experiments. No label balancing and no selection on outcomes.
- **SaTML and LLMail:** a seeded random 25% of eligible teams was reserved (seed `20261030`). Teams seen in earlier training are not eligible. Up to 100 rows per team were taken, in source order. For LLMail, only teams with at least one verified tool-call attack were eligible.
- **Exclusions:** each row was screened against the 10,966 inputs that earlier experiments trained on, tuned on or scored (every `results/**/*.jsonl` text). A row was dropped if it matched a prior ID, normalized text or first-window token hash, or if its character TF-IDF cosine similarity was 0.90 or more with a prior input or with an already-accepted holdout row. Totals: 678 exact duplicates, 199 near-duplicates of prior inputs, and 2,125 near-duplicates within the holdout. Mostly these are SaTML attackers resubmitting near-identical prompts and the LLMail emails sharing templates. Every exclusion is listed in `exclusions.jsonl`.

## Rules for every later run

Every training, validation or threshold-selection row must pass `Guard.reason()` from `scripts/freeze_final_holdout.py`:

- Never use the BrowseSafe test split.
- Never use any row, from any phase or split, from a team listed in `reservation.json`.
- Drop rows matching a holdout ID, normalized text or first-window token hash.
- Drop rows with character TF-IDF cosine of 0.90 or more with a holdout row.
- SaTML and LLMail rows must carry their team. The guard raises an error rather than admit an unchecked row.

## Reporting

- **Concentration:** 470 of 503 SaTML rows come from 5 teams, and 152 of 188 LLMail rows from 3. Report recall per team and macro-averaged across teams, alongside pooled numbers, and use team-clustered intervals. Pooled numbers alone overweight a few attackers' styles.
- **No over-defense slice.** The holdout has no benign text containing trigger words. NotInject is used up, and no human-reviewed replacement exists yet. False positive rate here comes only from benign BrowseSafe pages and the 116 emails, so this holdout cannot test the over-defense claim.
- **Fresh means unused by this repository,** not unknown to model pretraining. Lexical filtering does not prove semantic or website independence.
- **Once scored, it is used up.** Any later change to the model needs a new untouched holdout.
