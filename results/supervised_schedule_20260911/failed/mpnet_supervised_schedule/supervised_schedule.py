#!/usr/bin/env python3
"""Fixed six-epoch follow-up; original data and model-selection rule retained."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import supervised_pilot as pilot
from retrain_audited import digest, dump, jsonl, normalized
from windowed_followup import extract_html, load_rows, ranges, sha


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'darwin':
        subprocess.run(['/bin/cp', '-c', str(src), str(dst)], check=True)
    else:
        shutil.copy2(src, dst)


def prepare(old, out, parquet):
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoTokenizer
    from mpnet_inference import export
    if (out / 'protocol.json').exists():
        pilot.check(out)
        print('Preparation already frozen')
        return
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Prepare in an empty directory')
    original = json.loads((old / 'protocol.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, revision=pilot.MODEL_REVISION)
    prior_rows = load_rows(Path('results/colab_audit_20260911/audit_run/source_rows.jsonl'))
    raw_root = Path('results/supervised_pilot_20260911')
    raw = load_rows(raw_root / 'browsesafe_train_raw.jsonl') + load_rows(raw_root / 'browsesafe_test_raw.jsonl')
    assert len(raw) == 4100
    prior = {r['id']: r['text'] for r in prior_rows}
    for r in raw:
        if r['source_split'] == 'test':
            assert r['index'] < 2100
            prior[r['id']] = extract_html(r['content'])
    previous = list(prior.values())
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
    matrix = vectorizer.fit_transform(previous)
    text_hashes = {digest(normalized(t)) for t in previous}
    raw_hashes = {digest(normalized(r['content'])) for r in raw}
    token_hashes = {digest(json.dumps(tokenizer(t, truncation=True, max_length=384)['input_ids'])) for t in previous}
    print('Prior overlap filter fitted:', len(prior), 'inputs', flush=True)
    source = pq.read_table(parquet).slice(2100, 500).to_pylist()
    assert len(source) == 500
    candidates = []
    for index, r in enumerate(source, 2100):
        assert r['label'] in ('yes', 'no') and isinstance(r['content'], str)
        candidates.append(dict(id=f'perplexity-ai/browsesafe-bench/test/{index}', index=index,
                               source_split='test', label=int(r['label'] == 'yes'), content=r['content']))
    jsonl(out / 'fresh_candidates.jsonl', candidates)
    selected, vectors, exclusions = [], [], []
    for row in sorted(candidates, key=lambda r: digest('pilot-v1:' + r['id'])):
        assert row['id'] not in prior
        text = extract_html(row['content'])
        vector = vectorizer.transform([text])
        similarity = float((vector @ matrix.T).max())
        normalized_raw = digest(normalized(row['content']))
        head_hash = digest(json.dumps(tokenizer(text, truncation=True, max_length=384)['input_ids']))
        reason = None
        if not row['content'].strip():
            reason = 'empty raw HTML'
        elif normalized_raw in raw_hashes or digest(normalized(text)) in text_hashes or head_hash in token_hashes:
            reason = 'prior raw/extracted/first-window token duplicate'
        elif similarity >= .9:
            reason = 'extracted-text lexical cosine >=0.90 with prior inputs'
        elif any(float((vector @ v.T).toarray()[0, 0]) >= .9 for v in vectors):
            reason = 'within-holdout lexical cosine >=0.90'
        if reason:
            exclusions.append(dict(id=row['id'], split='holdout', reason=reason))
            continue
        encoded = tokenizer(row['content'], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids = encoded['input_ids']
        selected.append(dict(id=row['id'], split='holdout', label=row['label'], text=row['content'],
            extracted_text=text, raw_sha256=digest(row['content']), normalized_raw_sha256=normalized_raw,
            nearest_reserved_cosine=similarity, content_tokens=len(ids), window_count=len(ranges(len(ids))),
            token_sha256=digest(json.dumps(ids)), token_character_bounds=[encoded['offset_mapping'][0][0], encoded['offset_mapping'][-1][1]]))
        vectors.append(vector)
        raw_hashes.add(normalized_raw)
        text_hashes.add(digest(normalized(text)))
        token_hashes.add(head_hash)
    counts = dict(Counter(str(r['label']) for r in selected))
    if set(counts) != {'0', '1'} or min(counts.values()) < 100:
        raise ValueError(f'Insufficient holdout before any scoring: {counts}')
    jsonl(out / 'holdout.jsonl', sorted(selected, key=lambda r: r['id']))
    jsonl(out / 'exclusions.jsonl', exclusions)
    for name in ('train.jsonl', 'validation.jsonl', 'historical_classifier.json', 'historical_threshold.json'):
        copy(old / name, out / name)
        assert sha(out / name) == original['files'][name]
    for name in ('supervised_pilot.py', 'supervised_schedule.py', 'mpnet_inference.py', 'windowed_followup.py', 'retrain_audited.py'):
        copy(Path(__file__).with_name(name), out / name)
    copy(old / 'selection_frozen.json', out / 'prior/selection_frozen.json')
    for seed in pilot.SEEDS:
        for arm in ('frozen', 'finetuned'):
            name = f'{arm}_{seed}'
            for filename in ('checkpoint.pt', 'selection.json', 'history.json', 'training_order.jsonl',
                             'validation_epoch_1.jsonl', 'validation_epoch_2.jsonl', 'validation_predictions.jsonl'):
                copy(old / name / filename, out / 'prior' / name / filename)
            if arm == 'finetuned':
                export(out / 'prior' / name, old / name / 'inference_assets')
    protocol = dict(original, frozen_at=pilot.now(), epochs=6, stop_on_non_improvement=False,
        record_training_diagnostics=True, holdout_interval=[2100, 2600],
        experiment='fixed six epochs versus original at-most-two-epoch schedule; unchanged data and model selection',
        intervention='training schedule only: six fixed epochs, retaining the best validation checkpoint',
        checkpoint_rule='maximum validation recall at empirical FPR <=5%, then ROC-AUC; earliest epoch breaks ties; complete all six epochs',
        primary_endpoint='paired recall difference between six-epoch and prior fine-tuned schedules for every seed, accompanied by actual FPR',
        success_rule='Recall improves for all three fine-tuned seeds and each empirical holdout FPR is <=5%; report nominal uncertainty and all negative results',
        selection='Original train/validation bytes unchanged; all eligible official test rows 2100:2600 after prior and within-holdout overlap rejection; no test-score selection',
        prior_protocol_sha256=sha(old / 'protocol.json'), prior_selections_sha256=sha(old / 'selection_frozen.json'),
        source_parquet_sha256=sha(parquet),
        counts={**original['counts'], 'holdout': counts},
        windows={**original['windows'], 'holdout': sum(r['window_count'] for r in selected)},
        package_replay='All 192 validation pages for each prior and new fine-tuned checkpoint on A100; max absolute score error <=2e-6 and identical decisions',
        limitations=original['limitations'] + ['Validation reused across research iterations; only the new holdout remains unscored',
            'No website/template identifiers are supplied by this dataset; lexical filtering cannot establish family independence',
            'Prior fine-tuned holdout replay retains winning-window embeddings, not all window embeddings'])
    protocol['files'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    dump(out / 'protocol.json', protocol)
    print('FROZEN SCHEDULE INPUTS', json.dumps({k: protocol[k] for k in ('frozen_at', 'counts', 'windows')}), flush=True)


def prefix_check(out):
    report = {}
    for seed in pilot.SEEDS:
        for arm in ('frozen', 'finetuned'):
            name = f'{arm}_{seed}'
            history = json.loads((out / name / 'history.json').read_text())
            assert len(history) == 6
            diffs = []
            for epoch in (1, 2):
                a = load_rows(out / name / f'validation_epoch_{epoch}.jsonl')
                b = load_rows(out / 'prior' / name / f'validation_epoch_{epoch}.jsonl')
                assert [r['id'] for r in a] == [r['id'] for r in b]
                difference = float(np.max(np.abs(np.array([r['score'] for r in a]) - [r['score'] for r in b])))
                assert difference <= 2e-6, (name, epoch, difference)
                assert [r['prediction'] for r in a] == [r['prediction'] for r in b]
                diffs.append(difference)
            a = load_rows(out / name / 'training_order.jsonl')
            b = load_rows(out / 'prior' / name / 'training_order.jsonl')
            assert a[:2] == b
            report[name] = {'epoch_score_max_absolute_errors': diffs, 'same_decisions_and_page_orders': True}
    dump(out / 'prefix_reproduction.json', report)
    print('FIRST TWO EPOCHS REPRODUCED FOR ALL SIX TRIALS', flush=True)


def evaluate(out):
    import torch
    from mpnet_inference import Detector
    protocol = pilot.check(out)
    prefix_check(out)
    prior = json.loads((out / 'prior/selection_frozen.json').read_text())
    current = json.loads((out / 'selection_frozen.json').read_text())
    assert sha(out / 'prior/selection_frozen.json') == protocol['prior_selections_sha256']
    all_models = {}
    for prefix, selections in (('', current), ('prior/', prior)):
        for name, selected in selections['selections'].items():
            assert sha(out / prefix / name / 'checkpoint.pt') == selected['checkpoint_sha256']
            all_models[prefix + name] = selected
    for seed in pilot.SEEDS:
        for prefix in ('', 'prior/'):
            replay = json.loads((out / prefix / f'finetuned_{seed}/package_replay.json').read_text())
            assert replay['n'] == 192 and replay['identical_decisions'] and replay['max_absolute_score_error'] <= 2e-6
    if (out / 'holdout_results.json').exists():
        result = json.loads((out / 'holdout_results.json').read_text())
        if 'prior_summaries' in result:
            print('Complete comparison already evaluated; do not reuse for selection')
            return
    else:
        dump(out / 'schedule_comparison_frozen.json', dict(frozen_at=pilot.now(), protocol_sha256=sha(out / 'protocol.json'),
            new_selections_sha256=sha(out / 'selection_frozen.json'), prior_selections_sha256=sha(out / 'prior/selection_frozen.json'), models=all_models))
    pilot.evaluate(out)
    result = json.loads((out / 'holdout_results.json').read_text())
    rows = load_rows(out / 'holdout.jsonl')
    base = torch.load(out / 'base_holdout_features.pt', weights_only=True, mmap=True)
    summaries = {}
    for name, selected in prior['selections'].items():
        target = out / ('prior_' + name)
        target.mkdir(exist_ok=True)
        if selected['arm'] == 'frozen':
            state = torch.load(out / 'prior' / name / 'checkpoint.pt', weights_only=True)
            head = pilot.initialize(selected['seed'])
            head.load_state_dict(state['head'])
            scores = pilot.scores_for(head, base)
            del state, head
        else:
            model = Detector(out / 'prior' / name, 'cuda')
            outputs, vectors = [], []
            for i, row in enumerate(rows):
                prediction, vector = model.predict(row['text'], return_embedding=True)
                outputs.append(dict(id=row['id'], **prediction))
                vectors.append(vector)
                if i % 32 == 0:
                    print('PRIOR HOLDOUT', name, i + 1, '/', len(rows), flush=True)
            scores = [r['score'] for r in outputs]
            torch.save(torch.stack(vectors), target / 'winning_embeddings.pt')
            jsonl(target / 'winning_windows.jsonl', outputs)
            del model, vectors
        predictions, summaries[name] = pilot.summarize(rows, scores, selected['threshold'])
        jsonl(target / 'holdout_predictions.jsonl', predictions)
        print('PRIOR HOLDOUT RESULT', name, json.dumps(summaries[name]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    result.update(prior_summaries=summaries, completed_at=pilot.now(), comparison_frozen_sha256=sha(out / 'schedule_comparison_frozen.json'))
    dump(out / 'holdout_results.json', result)
    dump(out / 'sha256.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'sha256.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'prefix-check', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--parquet', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.old, args.output, args.parquet)
    elif args.stage == 'prefix-check':
        prefix_check(args.output)
    else:
        evaluate(args.output)
