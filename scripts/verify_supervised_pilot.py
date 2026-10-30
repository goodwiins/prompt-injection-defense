#!/usr/bin/env python3
"""Verify saved evidence and replay linear heads; no encoder training/inference."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from retrain_audited import MODEL, MODEL_REVISION, digest, dump
from windowed_followup import load_rows, sha
from supervised_pilot import SEEDS, threshold_at_budget, token_windows


def read(path):
    return json.loads(path.read_text())


def verify(out):
    torch.set_num_threads(2)
    hashes = read(out / 'sha256.json')
    for name, expected in hashes.items():
        assert sha(out / name) == expected, name
    protocol = read(out / 'protocol.json')
    for name, expected in protocol['files'].items():
        assert sha(out / name) == expected, name
    selection = read(out / 'selection_frozen.json')
    started = read(out / 'evaluation_started.json')
    results = read(out / 'holdout_results.json')
    assert sha(out / 'protocol.json') == selection['protocol_sha256']
    assert sha(out / 'selection_frozen.json') == started['selection_sha256']
    assert datetime.fromisoformat(protocol['frozen_at']) < datetime.fromisoformat(selection['frozen_at']) < datetime.fromisoformat(started['started_at']) < datetime.fromisoformat(results['completed_at'])
    assert 'A100' in read(out / 'environment.json')['gpu']
    subsets = {s: load_rows(out / f'{s}.jsonl') for s in ('train', 'validation', 'holdout')}
    merged = sum(subsets.values(), [])
    for field in ('id', 'normalized_raw_sha256', 'token_sha256'):
        assert len({r[field] for r in merged}) == len(merged), field
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for split, rows in subsets.items():
        assert dict(Counter(str(r['label']) for r in rows)) == protocol['counts'][split]
        assert all(digest(r['text']) == r['raw_sha256'] for r in rows)
        expected_ledger = load_rows(out / f'{split}_windows.jsonl')
        offset = 0
        for row in rows:
            chunks, ledger = token_windows(tokenizer, [row])
            assert ledger == expected_ledger[offset:offset + len(ledger)]
            offset += len(ledger)
            del chunks, ledger
        assert offset == len(expected_ledger)
    start, stop = protocol['holdout_interval']
    assert all(start <= int(r['id'].split('/')[-1]) < stop for r in subsets['holdout'])
    exclusions = load_rows(out / 'exclusions.jsonl')
    assert {r['id'] for r in subsets['holdout']} | {r['id'] for r in exclusions if r['split'] == 'holdout'} == {f'perplexity-ai/browsesafe-bench/test/{i}' for i in range(start, stop)}
    scores_by_name = {}
    historical = XGBClassifier()
    historical.load_model(out / 'historical_classifier.json')
    reference = load_rows(out / 'historical_holdout_predictions.jsonl')
    assert [r['id'] for r in reference] == [r['id'] for r in subsets['holdout']]
    assert [r['label'] for r in reference] == [r['label'] for r in subsets['holdout']]
    np.testing.assert_allclose(historical.predict_proba(np.load(out / 'historical_holdout_features.npy'))[:, 1],
                               [r['score'] for r in reference], atol=1e-7, rtol=1e-7)
    reference_threshold = read(out / 'historical_threshold.json')['threshold']
    assert [r['prediction'] for r in reference] == [int(r['score'] >= reference_threshold) for r in reference]
    assert results['historical_reference']['tp'] == sum(r['prediction'] for r in reference if r['label'])
    assert results['historical_reference']['fp'] == sum(r['prediction'] for r in reference if not r['label'])
    for name, chosen in selection['selections'].items():
        target = out / name
        assert sha(target / 'checkpoint.pt') == chosen['checkpoint_sha256']
        state = torch.load(target / 'checkpoint.pt', weights_only=True, map_location='cpu', mmap=True)
        head = torch.nn.Linear(768, 1)
        head.load_state_dict(state['head'])
        for split in ('validation', 'holdout'):
            feature_file = target / f'{split}_features.pt'
            if split == 'holdout' and chosen['arm'] == 'frozen':
                feature_file = out / 'base_holdout_features.pt'
            vectors = torch.load(feature_file, weights_only=True, map_location='cpu', mmap=True)
            assert len(vectors) == len(subsets[split])
            with torch.no_grad():
                scores = np.array([torch.sigmoid(head(v).max()).item() for v in vectors])
            predictions = load_rows(target / f'{split}_predictions.jsonl')
            assert [r['id'] for r in predictions] == [r['id'] for r in subsets[split]]
            assert [r['label'] for r in predictions] == [r['label'] for r in subsets[split]]
            np.testing.assert_allclose(scores, [r['score'] for r in predictions], atol=2e-6, rtol=2e-6)
            assert [r['prediction'] for r in predictions] == [int(r['score'] >= chosen['threshold']) for r in predictions]
            if split == 'validation':
                assert threshold_at_budget([r['label'] for r in predictions], [r['score'] for r in predictions]) == chosen['threshold']
                assert sum(r['prediction'] for r in predictions if not r['label']) / sum(not r['label'] for r in predictions) <= .05
            else:
                y = np.array([r['label'] for r in predictions])
                p = np.array([r['prediction'] for r in predictions])
                reported = results['summaries'][name]
                assert reported['tp'] == int(((p == 1) & (y == 1)).sum())
                assert reported['fp'] == int(((p == 1) & (y == 0)).sum())
                assert abs(reported['roc_auc'] - roc_auc_score(y, [r['score'] for r in predictions])) < 1e-12
                scores_by_name[name] = predictions
        history = read(target / 'history.json')
        best = max(history, key=lambda e: (e['validation']['recall']['value'], e['validation']['roc_auc'], -e['epoch']))
        assert chosen['epoch'] == best['epoch']
        if chosen['arm'] == 'finetuned':
            assert chosen['encoder_probe_max_change'] > 0
        del state, vectors
    intervals = {}
    for seed in SEEDS:
        a, b = (scores_by_name[f'{arm}_{seed}'] for arm in ('frozen', 'finetuned'))
        assert selection['selections'][f'frozen_{seed}']['initial_head_sha256'] == selection['selections'][f'finetuned_{seed}']['initial_head_sha256']
        oa = load_rows(out / f'frozen_{seed}/training_order.jsonl')
        ob = load_rows(out / f'finetuned_{seed}/training_order.jsonl')
        assert oa[:min(len(oa), len(ob))] == ob[:min(len(oa), len(ob))]
        y = np.array([r['label'] for r in a])
        pa, pb = (np.array([r['prediction'] for r in rows]) for rows in (a, b))
        sa, sb = (np.array([r['score'] for r in rows]) for rows in (a, b))
        rng = np.random.default_rng(20260911)
        samples = []
        for _ in range(1000):
            pos, neg = (rng.choice(np.flatnonzero(y == k), size=int((y == k).sum()), replace=True) for k in (1, 0))
            both = np.r_[pos, neg]
            samples.append([float((pb[pos] - pa[pos]).mean()), float((pb[neg] - pa[neg]).mean()), roc_auc_score(y[both], sb[both]) - roc_auc_score(y[both], sa[both])])
        intervals[str(seed)] = dict(zip(('recall_delta', 'fpr_delta', 'auc_delta'), np.percentile(samples, [2.5, 97.5], axis=0).T.tolist()))
    report = dict(verified_at=datetime.now().astimezone().isoformat(), recorded_hashes_verified=len(hashes),
                  counts=protocol['counts'], windows=protocol['windows'],
                  paired_nominal_95_intervals=intervals,
                  scope='All recorded hashes, raw token coverage, linear and historical XGBoost prediction replay, threshold and epoch choice, training orders, GPU identity and counts; GPU encoder inference/training not independently repeated',
                  uncertainty='1000 paired class-stratified document bootstrap resamples; nominal intervals assume independent pages and ignore unknown website/template families')
    dump(out / 'independent_verification.json', report)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for ax, metric, title, scale in zip(axes, ('recall', 'fpr', 'roc_auc'),
                                       ('Attack recall difference (pp)', 'False-positive rate difference (pp)', 'ROC-AUC difference'), (100, 100, 1)):
        key = 'auc_delta' if metric == 'roc_auc' else f'{metric}_delta'
        for row, seed in enumerate(SEEDS):
            values = [results['summaries'][f'{arm}_{seed}'][metric] for arm in ('frozen', 'finetuned')]
            values = [v['value'] if isinstance(v, dict) else v for v in values]
            delta = (values[1] - values[0]) * scale
            lo, hi = np.array(intervals[str(seed)][key]) * scale
            ax.hlines(row, lo, hi, color='#2364aa', lw=2)
            ax.plot(delta, row, 'o', color='#2364aa')
        ax.axvline(0, color='#666666', ls='--', lw=1)
        ax.set(yticks=range(len(SEEDS)), yticklabels=[f'Seed {s}' for s in SEEDS], ylim=(2.5, -.5), xlabel=title)
        ax.grid(axis='x', alpha=.2)
    counts = protocol['counts']['holdout']
    fig.suptitle('Encoder fine-tuning minus frozen encoder: fresh BrowseSafe holdout\n'
                 f"{sum(counts.values())} pages: {counts['1']} attacks, {counts['0']} non-injections; validation-frozen thresholds", fontsize=12)
    fig.supxlabel('Whiskers: nominal paired 95% bootstrap intervals (1,000 resamples). Unknown website/template dependence is omitted.', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'verified_paired_effects.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
