#!/usr/bin/env python3
"""Verify the fixed-schedule comparison and compute its paired uncertainty."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from verify_supervised_pilot import verify, read
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from supervised_pilot import SEEDS, summarize
from windowed_followup import load_rows, sha
from retrain_audited import dump


def verify_schedule(out):
    verify(out)
    protocol = read(out / 'protocol.json')
    result = read(out / 'holdout_results.json')
    replay_note = None
    if (out / 'replay_consistency.json').exists():
        replay_note = read(out / 'replay_consistency.json')
        expected_replay = read(out / 'replay_expected.json')
        notebook = Path(__file__).resolve().parents[1] / 'notebooks/colab_supervised_schedule_recovery_output.ipynb'
        assert sha(notebook) == expected_replay['source_notebook_sha256'] == replay_note['source_notebook_sha256']
        captured = {}
        for cell in read(notebook)['cells']:
            for item in cell.get('outputs', []):
                for line in ''.join(item.get('text', [])).splitlines():
                    if line.startswith('VALIDATION '):
                        _, name, epoch, body = line.split(' ', 3)
                        captured.setdefault(name, {})[epoch] = json.loads(body)
        assert captured == expected_replay['validation']
        assert replay_note['matched_validation_summaries'] == 36
        for name, expected_hash in expected_replay['known_new_checkpoints'].items():
            assert sha(out / name / 'checkpoint.pt') == expected_hash
        for name, epochs in expected_replay['validation'].items():
            assert len(epochs) == 6
            for epoch in read(out / name / 'history.json'):
                assert epoch['validation'] == epochs[str(epoch['epoch'])]
    frozen = read(out / 'schedule_comparison_frozen.json')
    assert protocol['epochs'] == 6 and protocol['stop_on_non_improvement'] is False
    assert len(frozen['models']) == 12
    assert frozen['protocol_sha256'] == sha(out / 'protocol.json')
    assert frozen['prior_selections_sha256'] == sha(out / 'prior/selection_frozen.json') == protocol['prior_selections_sha256']
    assert frozen['new_selections_sha256'] == sha(out / 'selection_frozen.json')
    assert result['comparison_frozen_sha256'] == sha(out / 'schedule_comparison_frozen.json')
    assert datetime.fromisoformat(frozen['frozen_at']) < datetime.fromisoformat(read(out / 'evaluation_started.json')['started_at'])
    train = load_rows(out / 'train.jsonl')
    validation = load_rows(out / 'validation.jsonl')
    rows = load_rows(out / 'holdout.jsonl')
    source = load_rows(out / 'fresh_candidates.jsonl')
    assert {r['id'] for r in source} == {f'perplexity-ai/browsesafe-bench/test/{i}' for i in range(2100, 2600)}
    source_by_id = {r['id']: r for r in source}
    assert all(r['text'] == source_by_id[r['id']]['content'] and r['label'] == source_by_id[r['id']]['label'] for r in rows)
    diagnostics, prior_predictions = {}, {}
    for name, chosen in frozen['models'].items():
        folder = out / name
        assert sha(folder / 'checkpoint.pt') == chosen['checkpoint_sha256']
        if chosen['arm'] == 'finetuned':
            replay = read(folder / 'package_replay.json')
            config = read(folder / 'inference_config.json')
            assert replay['n'] == 192 and replay['identical_decisions'] and replay['max_absolute_score_error'] <= 2e-6
            assert replay['input_sha256'] == sha(out / 'validation.jsonl')
            assert replay['expected_sha256'] == sha(folder / 'validation_predictions.jsonl')
            assert replay['config_sha256'] == sha(folder / 'inference_config.json')
            assert 'A100' in replay['gpu']
            reference = load_rows(folder / 'validation_predictions.jsonl')
            actual = replay['predictions']
            assert [r['id'] for r in actual] == [r['id'] for r in reference] == [r['id'] for r in validation]
            error = max(abs(a['score'] - b['score']) for a, b in zip(actual, reference))
            assert error == replay['max_absolute_score_error']
            assert [r['prediction'] for r in actual] == [r['prediction'] for r in reference]
            assert all(r['prediction'] == int(r['score'] >= chosen['threshold']) and r['threshold'] == chosen['threshold'] for r in actual)
            assert all(a['windows'] == b['window_count'] and a['content_tokens'] == b['content_tokens'] for a, b in zip(actual, validation))
            assert config['threshold'] == chosen['threshold']
            assert (config['content_tokens'], config['overlap'], config['window_batch']) == (382, 96, 32)
            for path, expected in config['files'].items():
                assert sha(folder / path) == expected
        if name.startswith('prior/'):
            trial = name.removeprefix('prior/')
            predicted = load_rows(out / ('prior_' + trial) / 'holdout_predictions.jsonl')
            state = torch.load(folder / 'checkpoint.pt', weights_only=True, map_location='cpu', mmap=True)
            head = torch.nn.Linear(768, 1)
            head.load_state_dict(state['head'])
            if chosen['arm'] == 'frozen':
                vectors = torch.load(out / 'base_holdout_features.pt', weights_only=True, mmap=True)
                with torch.no_grad():
                    scores = [float(torch.sigmoid(head(v).max())) for v in vectors]
            else:
                vectors = torch.load(out / ('prior_' + trial) / 'winning_embeddings.pt', weights_only=True, mmap=True)
                assert vectors.shape == (len(rows), 768)
                with torch.no_grad():
                    scores = torch.sigmoid(head(vectors).flatten()).tolist()
                winners = load_rows(out / ('prior_' + trial) / 'winning_windows.jsonl')
                assert [r['id'] for r in winners] == [r['id'] for r in rows]
                assert all(w['windows'] == r['window_count'] and 0 <= w['winning_window'] < w['windows'] and w['content_tokens'] == r['content_tokens'] for w, r in zip(winners, rows))
            np.testing.assert_allclose(scores, [r['score'] for r in predicted], atol=2e-6, rtol=2e-6)
            expected, summary = summarize(rows, [r['score'] for r in predicted], chosen['threshold'])
            assert expected == predicted
            reported = result['prior_summaries'][trial]
            assert summary.keys() == reported.keys()
            for key, value in summary.items():
                if isinstance(value, dict):
                    assert value.keys() == reported[key].keys()
                    for metric, number in value.items():
                        np.testing.assert_allclose(number, reported[key][metric], atol=1e-12, rtol=0)
                else:
                    np.testing.assert_allclose(value, reported[key], atol=1e-12, rtol=0)
            prior_predictions[trial] = predicted
            del state, head, vectors
            continue
        history = read(folder / 'history.json')
        steps = load_rows(folder / 'optimizer_diagnostics.jsonl')
        assert len(history) == 6 and [r['epoch'] for r in history] == list(range(1, 7))
        assert len(steps) == 384 and [r['step'] for r in steps] == list(range(1, 385))
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm_before_clipping']) and not r['optimizer_step_skipped'] for r in steps)
        assert any(r['gradient_norm_before_clipping'] > 0 for r in steps)
        retries = [attempt for step in steps for attempt in step.get('overflow_retries', [])]
        assert all(0 < r['scale_after'] < r['scale_before'] and r['nonfinite_gradient_values'] > 0 for r in retries)
        assert all(len(r.get('overflow_retries', [])) <= 8 for r in steps)
        for epoch in (1, 2):
            a = load_rows(folder / f'validation_epoch_{epoch}.jsonl')
            b = load_rows(out / 'prior' / name / f'validation_epoch_{epoch}.jsonl')
            assert [r['id'] for r in a] == [r['id'] for r in b]
            np.testing.assert_allclose([r['score'] for r in a], [r['score'] for r in b], atol=2e-6, rtol=0)
            assert [r['prediction'] for r in a] == [r['prediction'] for r in b]
        order = load_rows(folder / 'training_order.jsonl')
        assert order[:2] == load_rows(out / 'prior' / name / 'training_order.jsonl')
        assert all(Counter(r['ids']) == Counter(x['id'] for x in train) for r in order)
        item = dict(optimizer_steps=len(steps), skipped_steps=0,
                    overflow_retried_batches=sum(bool(r.get('overflow_retries')) for r in steps),
                    overflow_skipped_attempts=len(retries),
                    fraction_clipped=float(np.mean([r['gradient_norm_before_clipping'] > 1 for r in steps])),
                    epoch_losses=[r['loss'] for r in history], selected_epoch=chosen['epoch'])
        if chosen['arm'] == 'finetuned':
            windows = load_rows(folder / 'training_windows.jsonl')
            assert len(windows) == 1536
            metadata = {r['id']: r for r in train}
            by_epoch = [{r['id']: r for r in windows if r['epoch'] == epoch} for epoch in range(1, 7)]
            assert all(set(x) == set(metadata) for x in by_epoch)
            assert all(0 <= r['winner'] < r['window_count'] == metadata[r['id']]['window_count'] for r in windows)
            item['median_training_logit_gaps'] = [float(np.median([r['top_logit_gap'] for r in epoch.values() if r['top_logit_gap'] is not None])) for epoch in by_epoch]
            item['winner_change_fraction'] = [sum(a[i]['winner'] != b[i]['winner'] for i in a) / len(a) for a, b in zip(by_epoch, by_epoch[1:])]
        diagnostics[name] = item
    intervals, changes = {}, {}
    for seed in SEEDS:
        name = f'finetuned_{seed}'
        a = prior_predictions[name]
        b = load_rows(out / name / 'holdout_predictions.jsonl')
        y = np.array([r['label'] for r in a])
        pa, pb = (np.array([r['prediction'] for r in data]) for data in (a, b))
        sa, sb = (np.array([r['score'] for r in data]) for data in (a, b))
        rng = np.random.default_rng(20260911)
        samples = []
        for _ in range(1000):
            pos, neg = (rng.choice(np.flatnonzero(y == k), size=int((y == k).sum()), replace=True) for k in (1, 0))
            both = np.r_[pos, neg]
            samples.append([float((pb[pos] - pa[pos]).mean()), float((pb[neg] - pa[neg]).mean()), roc_auc_score(y[both], sb[both]) - roc_auc_score(y[both], sa[both])])
        intervals[str(seed)] = dict(zip(('recall_delta', 'fpr_delta', 'auc_delta'), np.percentile(samples, [2.5, 97.5], axis=0).T.tolist()))
        changes[str(seed)] = dict(recall_delta=float((pb[y == 1] - pa[y == 1]).mean()),
            fpr_delta=float((pb[y == 0] - pa[y == 0]).mean()), auc_delta=float(roc_auc_score(y, sb) - roc_auc_score(y, sa)))
    passed = all(changes[str(seed)]['recall_delta'] > 0 and result['summaries'][f'finetuned_{seed}']['fpr']['value'] <= .05 for seed in SEEDS)
    report = dict(verified_at=datetime.now().astimezone().isoformat(),
        recorded_hashes_verified=len(read(out / 'sha256.json')), prior_and_new_models=12,
        gpu_package_replayed_predictions=6 * 192, same_first_two_epochs=True,
        diagnostics=diagnostics, schedule_changes=changes, paired_nominal_95_intervals=intervals,
        declared_directional_criterion_passed=passed,
        runtime_loss_replay=replay_note,
        scope='All recorded hashes, prior and new saved-head replay, full token coverage, all model/threshold freezes, six-epoch histories, optimizer/window logs and recorded A100 package replays; full GPU encoder training not independently repeated',
        limitations='Nominal paired document bootstrap omits website/template dependence, label uncertainty and broader seed variation. Prior fine-tuned head replay uses saved winning embeddings only.')
    dump(out / 'schedule_verification.json', report)
    plot(out, report)
    print(json.dumps(report, indent=2))


def plot(out, report):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for ax, metric, label, scale in zip(axes, ('recall_delta', 'fpr_delta', 'auc_delta'),
        ('Attack recall difference (pp)', 'False-positive rate difference (pp)', 'ROC-AUC difference'), (100, 100, 1)):
        for row, seed in enumerate(SEEDS):
            lo, hi = np.array(report['paired_nominal_95_intervals'][str(seed)][metric]) * scale
            ax.hlines(row, lo, hi, color='#2364aa', lw=2)
            ax.plot(report['schedule_changes'][str(seed)][metric] * scale, row, 'o', color='#2364aa')
        ax.axvline(0, color='#666666', ls='--', lw=1)
        ax.set(yticks=range(3), yticklabels=[f'Seed {s}' for s in SEEDS], ylim=(2.5, -.5), xlabel=label)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Six-epoch minus prior fine-tuned schedule: fresh BrowseSafe holdout\n453 pages: 217 attacks, 236 non-injections; validation-frozen thresholds', fontsize=12)
    fig.supxlabel('Whiskers: nominal paired 95% bootstrap intervals (1,000 resamples). Unknown website/template dependence is omitted.', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'verified_schedule_effects.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for seed, color, marker in zip(SEEDS, ('#2364aa', '#b24c16', '#3c7e56'), ('o', 's', '^')):
        history = read(out / f'finetuned_{seed}/history.json')
        axes[0].plot(range(1, 7), [r['loss'] for r in history], color=color, marker=marker, label=f'Seed {seed}')
        axes[1].plot(range(1, 7), [r['validation']['recall']['value'] for r in history], color=color, marker=marker)
    axes[0].set(ylabel='Mean page BCE loss', ylim=(0, None))
    axes[1].set(ylabel='Validation attack recall', ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set(xlabel='Epoch', xticks=range(1, 7))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Fine-tuning across six epochs\n256 training pages; 192 validation pages; recall selected at empirical FPR ≤5%', fontsize=12)
    fig.savefig(out / 'training_diagnostics.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    verify_schedule(Path(sys.argv[1]))
