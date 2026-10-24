#!/usr/bin/env python3
"""Freeze and verify the user-authorized AI-reviewed HTML correction.

Training and portable inference reuse the existing runner. No test is exported.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from retrain_audited import digest, dump, jsonl
from windowed_followup import load_rows, ranges, sha
from supervised_pilot import SEEDS, SPEC, reviewed_span_targets, summarize, threshold_at_budget

PACKET = ROOT / 'results/html_coverage_correction_20260911'
OLD = ROOT / 'results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay'


def prepare(out):
    assert not out.exists(), 'Use a fresh output directory; frozen inputs cannot be overwritten'
    parent = json.loads((PACKET / 'protocol.json').read_text())
    for name, expected in parent['files'].items():
        assert sha(ROOT / name) == expected, name
    authorization = json.loads((PACKET / 'ai_authorization.json').read_text())
    assert authorization['reviewer_type'] == 'AI' and not authorization['human_review_attested']
    assert authorization['parent_protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert authorization['human_csv_sha256'] == sha(PACKET / 'human_review.csv')
    with (PACKET / 'ai_review.csv').open(newline='') as stream:
        reviews = list(csv.DictReader(stream))
    candidates = {r['id']: r for r in load_rows(PACKET / 'candidates.jsonl')}
    assert len(reviews) == len(candidates) == 45
    labels = {}
    for review in reviews:
        rid = review['id']
        assert rid not in labels and rid in candidates
        assert review['label'] in ('injection', 'non-injection'), 'Quarantine uncertain labels before export'
        assert review['reviewer_type'] == 'AI'
        assert review['human_review_attested'] == review['independent_review_attested'] == 'no'
        assert review['evidence'] and review['notes']
        assert datetime.fromisoformat(review['reviewed_at']).tzinfo is not None
        assert review['source_sha256'] == candidates[rid]['sha256'] == sha(PACKET / candidates[rid]['path'])
        labels[rid] = int(review['label'] == 'injection')
    refs = {r['id']: r for r in load_rows(PACKET / 'construction_reference.jsonl')}
    assert all(labels[rid] == refs[rid]['proposed_label'] for rid in labels), 'Record adjudication before changing the frozen replacement plan'
    plan = json.loads((PACKET / 'replacement_plan.json').read_text())
    legacy = load_rows(OLD / 'train.jsonl')
    retained, removed = set(plan['retained_legacy_train_ids']), set(plan['removed_legacy_train_ids'])
    assert retained.isdisjoint(removed) and retained | removed == {r['id'] for r in legacy}
    assert len(retained) == 236 and len(removed) == 20
    selected = set(plan['added_candidate_train_ids'])
    assert len(selected) == 20 and selected.isdisjoint(plan['review_only_candidate_ids'])
    rows = {'train': [r for r in legacy if r['id'] in retained], 'validation': load_rows(OLD / 'validation.jsonl')}
    for rid, candidate in candidates.items():
        if candidate['split'] != 'validation' and rid not in selected:
            continue
        text = (PACKET / candidate['path']).read_text()
        row = dict(candidate, text=text, label=labels[rid], raw_sha256=digest(text),
                   window_count=len(ranges(candidate['content_tokens'])),
                   label_provenance='Codex AI review, not independent or human',
                   condition=refs[rid]['condition'], template_family=refs[rid]['template_family'])
        rows[candidate['split']].append(row)
    assert Counter(r['label'] for r in rows['train']) == {0: 128, 1: 128}
    assert Counter(r['label'] for r in rows['validation']) == {0: 106, 1: 101}
    assert {r['id'] for r in rows['train']}.isdisjoint(r['id'] for r in rows['validation'])
    reserved = set(json.loads((PACKET / 'holdout_reservation.json').read_text())['websites'])
    sites = {s: {r['website'] for r in candidates.values() if r['split'] == s} for s in rows}
    assert sites['train'].isdisjoint(sites['validation']) and reserved.isdisjoint(sites['train'] | sites['validation'])
    out.mkdir(parents=True)
    for split, data in rows.items():
        jsonl(out / f'{split}.jsonl', sorted(data, key=lambda r: r['id']))
    for name in ('retrain_audited.py', 'windowed_followup.py', 'supervised_pilot.py', 'mpnet_inference.py'):
        shutil.copy2(ROOT / 'training' / name, out / name)
    for name in ('ai_review.csv', 'ai_authorization.json', 'construction_reference.jsonl', 'replacement_plan.json', 'holdout_reservation.json'):
        shutil.copy2(PACKET / name, out / name)
    shutil.copytree(OLD / 'finetuned_17/inference_assets', out / 'inference_assets')
    old_protocol = json.loads((OLD / 'protocol.json').read_text())
    settings = {k: old_protocol[k] for k in ('model', 'revision', 'gpu', 'seeds', 'epochs', 'document_batch',
        'inference_window_batch', 'encoder_lr', 'head_lr', 'weight_decay', 'gradient_clip', 'max_length',
        'overlap', 'validation_fpr_budget', 'loss', 'comparison', 'dropout', 'precision', 'checkpoint_rule',
        'threshold_rule', 'stop_on_non_improvement', 'record_training_diagnostics', 'retry_amp_overflow', 'amp_overflow_policy')}
    for key in ('document_batch', 'inference_window_batch', 'encoder_lr', 'head_lr', 'weight_decay', 'gradient_clip', 'max_length', 'overlap', 'validation_fpr_budget'):
        assert settings[key] == SPEC[key], key
    protocol = dict(settings, frozen_at=datetime.now(timezone.utc).isoformat(),
        experiment='AI-reviewed HTML coverage correction; validation only',
        intervention='Replace 20 training rows with balanced reviewed HTML examples; add 15 validation examples',
        counts={s: dict(Counter(r['label'] for r in data)) for s, data in rows.items()},
        windows={s: sum(r['window_count'] for r in data) for s, data in rows.items()},
        expected_optimizer_updates_per_trial=384, final_test_included=False,
        primary_endpoint='Descriptive validation recall and FPR for every seed and arm, overall and by source/condition; no fresh-test success claim',
        limitations=authorization['limitations'], parent_protocol_sha256=sha(PACKET / 'protocol.json'),
        prior_schedule_protocol_sha256=sha(OLD / 'protocol.json'),
        legacy_train_sha256=sha(OLD / 'train.jsonl'), legacy_validation_sha256=sha(OLD / 'validation.jsonl'),
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    dump(out / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('counts', 'windows', 'frozen_at', 'final_test_included')}))


def prepare_spans(out):
    """Reuse the completed correction's data and add only reviewed span supervision."""
    from transformers import AutoTokenizer
    source = PACKET / 'completed/mpnet_html_coverage_ai'
    protocol = json.loads((source / 'protocol.json').read_text())
    assert not out.exists(), 'Use a fresh output directory'
    assert json.loads((source / 'verification.json').read_text())['protocol_sha256'] == sha(source / 'protocol.json')
    for name, expected in protocol['files'].items():
        assert sha(source / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(source / 'inference_assets', local_files_only=True)
    with (source / 'ai_review.csv').open(newline='') as stream:
        targets = reviewed_span_targets(tokenizer, load_rows(source / 'train.jsonl'),
            load_rows(source / 'construction_reference.jsonl'), list(csv.DictReader(stream)))
    assert Counter(r['label'] for r in targets) == {1: 10, 0: 5}
    out.mkdir(parents=True)
    for name in protocol['files']:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, path)
    shutil.copy2(ROOT / 'training/supervised_pilot.py', out / 'supervised_pilot.py')
    jsonl(out / 'span_targets.jsonl', targets)
    protocol.update(frozen_at=datetime.now(timezone.utc).isoformat(),
        experiment='Reviewed attack-span auxiliary loss; implementation prepared, untrained',
        intervention='Add a fixed auxiliary loss on complete reviewed training insertions; data and page loss unchanged',
        loss='Page-max BCE + 0.5 times class-balanced reviewed-span BCE, normalized per training page',
        span_supervision=dict(weight=.5, normalization='Average covering windows within each annotated page, apply N/(2*N_class), sum over annotated pages and divide by document batch size',
            labels='10 reviewed direct attacks and 5 reviewed quoted negatives; no inferred labels for other windows'),
        prior_correction_protocol_sha256=sha(source / 'protocol.json'),
        prior_correction_selections_sha256=sha(source / 'selection_frozen.json'),
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    dump(out / 'protocol.json', protocol)
    assert all(sha(out / f'{split}.jsonl') == sha(source / f'{split}.jsonl') for split in ('train', 'validation'))
    print('Prepared reviewed span supervision:', dict(Counter(r['label'] for r in targets)), 'weight=0.5; training not started')


def verify(out):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        assert sha(out / name) == expected, name
    assert protocol['final_test_included'] is False and not (out / 'holdout.jsonl').exists()
    tokenizer = AutoTokenizer.from_pretrained(out / 'inference_assets', local_files_only=True)
    rows = {s: load_rows(out / f'{s}.jsonl') for s in ('train', 'validation')}
    if protocol.get('span_supervision', {}).get('weight', 0):
        with (out / 'ai_review.csv').open(newline='') as stream:
            targets = reviewed_span_targets(tokenizer, rows['train'],
                load_rows(out / 'construction_reference.jsonl'), list(csv.DictReader(stream)))
        assert targets == load_rows(out / 'span_targets.jsonl')
    for split, data in rows.items():
        assert {str(k): v for k, v in Counter(r['label'] for r in data).items()} == protocol['counts'][split]
        expected_ledger = []
        for row in data:
            ids = tokenizer(row['text'], add_special_tokens=False, verbose=False)['input_ids']
            assert len(ids) == row['content_tokens'] and digest(json.dumps(ids)) == row['token_sha256']
            expected_ledger.extend(dict(id=row['id'], start=a, end=b, token_sha256=digest(json.dumps(ids[a:b]))) for a, b in ranges(len(ids)))
        assert len(expected_ledger) == protocol['windows'][split]
        if (out / f'{split}_windows.jsonl').exists():
            assert expected_ledger == load_rows(out / f'{split}_windows.jsonl')
    if not (out / 'selection_frozen.json').exists():
        print('Frozen inputs verified; training has not completed')
        return
    frozen = json.loads((out / 'selection_frozen.json').read_text())
    completion = json.loads((out / 'completion.json').read_text())
    assert completion['manifest_sha256'] == sha(out / 'artifact_manifest.json')
    artifacts = json.loads((out / 'artifact_manifest.json').read_text())
    for name, record in artifacts.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert (out / name).stat().st_size == record['bytes'] and sha(out / name) == record['sha256'], name
    environment = json.loads((out / 'environment.json').read_text())
    assert protocol['gpu'].split()[-1] in environment['gpu']
    assert frozen['protocol_sha256'] == sha(out / 'protocol.json')
    assert set(frozen['selections']) == {f'{arm}_{seed}' for arm in ('frozen', 'finetuned') for seed in SEEDS}
    summaries, max_error = {}, 0.
    refs = {r['id']: r for r in load_rows(out / 'construction_reference.jsonl')}
    window_diagnostics = []
    for name, selection in frozen['selections'].items():
        folder = out / name
        assert selection == json.loads((folder / 'selection.json').read_text())
        assert sha(folder / 'checkpoint.pt') == selection['checkpoint_sha256']
        history = json.loads((folder / 'history.json').read_text())
        assert len(history) == 6 and [h['optimizer_steps'] for h in history] == [64 * i for i in range(1, 7)]
        best = max(history, key=lambda h: (h['validation']['recall']['value'], h['validation']['roc_auc'], -h['epoch']))
        assert best['epoch'] == selection['epoch'] and best['validation'] == selection['validation']
        for h in history:
            preds = load_rows(folder / f"validation_epoch_{h['epoch']}.jsonl")
            assert [(r['id'], r['label']) for r in preds] == [(r['id'], r['label']) for r in rows['validation']]
            threshold = threshold_at_budget([r['label'] for r in preds], [r['score'] for r in preds])
            assert threshold == h['validation']['threshold']
            assert summarize(rows['validation'], [r['score'] for r in preds], threshold)[1] == h['validation']
        optimizer = load_rows(folder / 'optimizer_diagnostics.jsonl')
        assert [r['step'] for r in optimizer] == list(range(1, 385))
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm_before_clipping']) for r in optimizer)
        assert all(not r['optimizer_step_skipped'] and len(r.get('overflow_retries', [])) <= 8 for r in optimizer)
        if protocol.get('span_supervision', {}).get('weight', 0):
            annotations = {r['id']: r for r in targets}
            order = load_rows(folder / 'training_order.jsonl')
            batches = [(epoch['epoch'], epoch['ids'][i:i + protocol['document_batch']])
                       for epoch in order for i in range(0, len(epoch['ids']), protocol['document_batch'])]
            span_log = load_rows(folder / 'span_diagnostics.jsonl')
            assert len(span_log) == len(optimizer) == len(batches)
            for recorded, step, (epoch, ids) in zip(span_log, optimizer, batches):
                expected = [dict(id=rid, window=w, label=annotations[rid]['label'],
                                 weight=annotations[rid]['class_weight'] / len(annotations[rid]['windows']))
                            for rid in ids if rid in annotations for w in annotations[rid]['windows']]
                assert recorded['spans'] == expected and recorded['step'] == step['step']
                assert recorded['epoch'] == epoch and recorded['coefficient'] == protocol['span_supervision']['weight']
                assert np.isfinite(recorded['page_loss']) and np.isfinite(recorded['auxiliary_loss'])
                if not expected:
                    assert recorded['auxiliary_loss'] == 0
                np.testing.assert_allclose(step['loss'], recorded['page_loss'] +
                    recorded['coefficient'] * recorded['auxiliary_loss'], rtol=1e-6, atol=1e-7)
        vectors = torch.load(folder / 'validation_features.pt', map_location='cpu', weights_only=True)
        state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
        head = torch.nn.Linear(768, 1)
        head.load_state_dict(state['head'])
        with torch.no_grad():
            scores = [torch.sigmoid(head(v).max()).item() for v in vectors]
            for row, vector in zip(rows['validation'], vectors, strict=True):
                assert vector.shape == (row['window_count'], 768) and torch.isfinite(vector).all()
                if row['id'] in refs:
                    logits = head(vector).flatten()
                    covered = refs[row['id']]['full_insertion_windows']
                    window_diagnostics.append(dict(model=name, id=row['id'], website=row['website'],
                        condition=row['condition'], winning_window=int(logits.argmax()),
                        score=float(torch.sigmoid(logits.max())), full_insertion_windows=covered,
                        insertion_window_scores=[float(torch.sigmoid(logits[i])) for i in covered],
                        insertion_window_won=int(logits.argmax()) in covered))
        predictions = load_rows(folder / 'validation_predictions.jsonl')
        error = float(np.max(np.abs(np.array(scores) - [r['score'] for r in predictions])))
        max_error = max(max_error, error)
        assert error <= 2e-6
        # A threshold can equal a saved score; tolerate FP rounding for score replay,
        # but use the saved scores to verify exact decisions and selection.
        assert predictions == load_rows(folder / f"validation_epoch_{selection['epoch']}.jsonl")
        assert all(r['prediction'] == int(r['score'] >= selection['threshold']) for r in predictions)
        by_source = {}
        for cohort in ('all', 'legacy', 'new_html'):
            subset = [i for i, r in enumerate(rows['validation']) if cohort == 'all' or ('website' in r) == (cohort == 'new_html')]
            by_source[cohort] = summarize([rows['validation'][i] for i in subset], [predictions[i]['score'] for i in subset], selection['threshold'])[1]
        by_source['conditions'] = {condition: dict(n=len(indices), flagged=sum(predictions[i]['prediction'] for i in indices))
            for condition in ('original', 'direct', 'quoted')
            for indices in [[i for i, r in enumerate(rows['validation']) if r.get('condition') == condition]]}
        summaries[name] = by_source
        if selection['arm'] == 'finetuned':
            assert selection['encoder_probe_max_change'] > 0
            log = load_rows(folder / 'training_windows.jsonl')
            assert len(log) == 6 * len(rows['train']) and all(0 <= r['winner'] < r['window_count'] for r in log)
            replay = json.loads((folder / 'package_replay.json').read_text())
            assert replay['n'] == 207 and replay['identical_decisions'] and replay['max_absolute_score_error'] <= 2e-6
            assert replay['input_sha256'] == sha(out / 'validation.jsonl')
            assert replay['expected_sha256'] == sha(folder / 'validation_predictions.jsonl')
            assert replay['config_sha256'] == sha(folder / 'inference_config.json')
            config = json.loads((folder / 'inference_config.json').read_text())
            assert config['threshold'] == selection['threshold']
            for path, expected in config['files'].items():
                assert sha(folder / path) == expected
            actual = replay['predictions']
            assert [r['id'] for r in actual] == [r['id'] for r in predictions]
            assert [r['prediction'] for r in actual] == [r['prediction'] for r in predictions]
            assert max(abs(a['score'] - b['score']) for a, b in zip(actual, predictions)) == replay['max_absolute_score_error']
        del vectors, state, head
    for seed in SEEDS:
        a, b = out / f'frozen_{seed}', out / f'finetuned_{seed}'
        assert (a / 'training_order.jsonl').read_bytes() == (b / 'training_order.jsonl').read_bytes()
        assert frozen['selections'][a.name]['initial_head_sha256'] == frozen['selections'][b.name]['initial_head_sha256']
        for order in load_rows(a / 'training_order.jsonl'):
            assert len(order['ids']) == 256 and set(order['ids']) == {r['id'] for r in rows['train']}
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(out / 'protocol.json'),
        frozen_input_hashes=len(protocol['files']), artifact_hashes=len(artifacts), models=6, finite_updates_per_trial=384,
        max_local_head_score_error=max_error, validation=summaries, final_test_evaluated=False,
        limitation='Local verification replays saved features and heads; GPU encoder computation is not independently repeated locally.')
    dump(out / 'verification.json', result)
    jsonl(out / 'new_validation_window_diagnostics.jsonl', window_diagnostics)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'prepare-spans', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'prepare-spans': prepare_spans, 'verify': verify}[args.stage](args.output)
