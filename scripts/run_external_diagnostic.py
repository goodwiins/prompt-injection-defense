#!/usr/bin/env python3
"""Six AI-reviewed pages: frozen inference only, never human holdout approval."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from mpnet_inference import Detector, sha, windows
from retrain_audited import digest, dump, jsonl
from supervised_pilot import features, make_encoder, token_windows

DATA = ROOT / 'results/generalization_20260911'
OUT = DATA / 'diagnostic'
MODELS = ROOT / 'results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite frozen preparation'
    model_manifest = json.loads((DATA / 'models_frozen.json').read_text())
    paths = [ROOT / p for p in model_manifest['files']]
    paths += [DATA / n for n in ['models_frozen.json', 'candidates.jsonl', 'ai_review.csv',
                                'human_review.csv', 'construction_reference.jsonl']]
    paths += [DATA / r['path'] for r in read_rows(DATA / 'candidates.jsonl')]
    paths += [ROOT / 'training' / n for n in ['mpnet_inference.py', 'supervised_pilot.py',
                                            'retrain_audited.py', 'windowed_followup.py']]
    paths += [Path(__file__)]
    dump(OUT / 'protocol.json', dict(
        frozen_at=now(), scope='Exploratory diagnostic, six AI-reviewed pages in three dependent pairs',
        labels='Codex AI review; not independent blind annotation; negatives provisional',
        human_review_attested=False, unseen_family_verified=False,
        strict_generalization_preflight='Remains blocked; this diagnostic is not the strict study',
        selection='All six pages and all three seeds in both arms; no fitting or threshold changes',
        analysis='Descriptive decisions, paired score changes, full token coverage and payload-window scores',
        no_claims=['Population FPR or robustness', 'Independent family generalization', 'Downstream attack prevention'],
        files={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    check_inputs()
    print('Frozen exploratory protocol; strict human/family gate unchanged', flush=True)


def check_inputs():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['human_review_attested'] is False and protocol['unseen_family_verified'] is False
    for path, expected in protocol['files'].items():
        assert sha(ROOT / path) == expected, path
    models = json.loads((DATA / 'models_frozen.json').read_text())
    for path, expected in models['files'].items():
        assert sha(ROOT / path) == expected, path
    reviews = list(csv.DictReader((DATA / 'ai_review.csv').open()))
    rows = read_rows(DATA / 'candidates.jsonl')
    assert len(rows) == len(reviews) == 6
    assert {r['id'] for r in rows} == {r['id'] for r in reviews}
    assert len({r['pair_id'] for r in rows}) == 3
    labels = {r['id']: r for r in reviews}
    for row in rows:
        review = labels[row['id']]
        assert review['reviewer_type'] == 'AI' and review['human_review_attested'] == 'no'
        assert review['source_sha256'] == row['sha256'] == sha(DATA / row['path'])
        assert review['label'] in ('injection', 'non-injection')
        row.update(label=int(review['label'] == 'injection'), text=(DATA / row['path']).read_text())
    assert sum(r['label'] for r in rows) == 3
    return models, rows


def run():
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    assert not (OUT / 'completed.json').exists(), 'Completed diagnostic must not be rerun'
    models, rows = check_inputs()
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
    dump(OUT / 'environment.json', dict(started_at=now(), gpu=torch.cuda.get_device_name(),
        packages={n: importlib.metadata.version(n) for n in
                  ['torch', 'transformers', 'sentence-transformers', 'numpy', 'scikit-learn']},
        protocol_sha256=sha(OUT / 'protocol.json')))
    base = models['base_encoder']
    downloaded = Path(hf_hub_download(base['model'], base['filename'], revision=base['revision']))
    assert sha(downloaded) == base['sha256'] and downloaded.stat().st_size == base['bytes']
    dump(OUT / 'base_encoder_verified.json', dict(**base, verified_at=now()))
    encoder = make_encoder()
    spans = {r['id']: r for r in read_rows(DATA / 'construction_reference.jsonl')}
    coverage = []
    for row in rows:
        encoded = encoder.tokenizer(row['text'], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        assert len(ids) == row['content_tokens']
        row['token_sha256'] = digest(json.dumps(ids))
        bounds = list(windows(len(ids)))
        span = spans[row['id']]
        attack_tokens, attack_windows = [], []
        if row['label']:
            start, end = span['insertion_start'], span['insertion_end']
            attack_tokens = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            assert attack_tokens
            attack_windows = [i for i, (a, b) in enumerate(bounds)
                              if a <= attack_tokens[0] and b > attack_tokens[-1]]
            assert attack_windows, 'At least one inference window must contain the whole insertion'
        coverage.append(dict(id=row['id'], token_sha256=row['token_sha256'], content_tokens=len(ids),
            bounds=bounds, attack_token_bounds=[attack_tokens[0], attack_tokens[-1]+1] if attack_tokens else None,
            complete_attack_windows=attack_windows, first_window_contains_attack=bool(attack_windows and 0 in attack_windows)))
    jsonl(OUT / 'coverage.jsonl', coverage)
    documents, ledger = token_windows(encoder.tokenizer, rows)
    jsonl(OUT / 'window_ledger.jsonl', ledger)
    vectors = features(encoder, documents)
    del encoder, documents
    gc.collect()
    torch.cuda.empty_cache()
    predictions, winners = [], {}

    def record(name, row, trace, logits, vector):
        assert len(logits) == len(trace['bounds']) and np.isfinite(logits).all()
        winner = int(np.argmax(logits))
        score = float(torch.sigmoid(torch.tensor(logits[winner], dtype=torch.float32)))
        threshold = models['models'][name]['threshold']
        relevant = trace['complete_attack_windows']
        prediction = dict(model=name, id=row['id'], pair_id=row['pair_id'], website=row['website'],
            label=row['label'], score=score, threshold=threshold, prediction=int(score >= threshold),
            windows=len(logits), winning_window=winner, winning_token_bounds=trace['bounds'][winner],
            winner_contains_complete_attack=winner in relevant,
            payload_window_score=float(torch.sigmoid(torch.tensor(max(logits[i] for i in relevant)))) if relevant else None,
            window_logits=logits)
        predictions.append(prediction)
        winners[f"{name}/{row['id']}"] = vector.cpu()
        jsonl(OUT / 'predictions.jsonl', predictions)
        print(name, row['website'], row['label'], round(score, 6), prediction['prediction'], flush=True)

    for seed in (17, 42, 73):
        name = f'frozen_{seed}'
        head = torch.nn.Linear(768, 1).cuda().eval()
        head.load_state_dict(torch.load(MODELS / name / 'checkpoint.pt', weights_only=True, map_location='cpu')['head'])
        with torch.inference_mode():
            for row, trace, vector in zip(rows, coverage, vectors, strict=True):
                logits = head(vector.cuda()).flatten().cpu().tolist()
                record(name, row, trace, logits, vector[int(np.argmax(logits))])
        del head
    del vectors
    for seed in (17, 42, 73):
        name = f'finetuned_{seed}'
        detector = Detector(MODELS / name, 'cuda')
        assert detector.config['threshold'] == models['models'][name]['threshold']
        for row, trace in zip(rows, coverage, strict=True):
            batches = []
            hook = detector.head.register_forward_hook(lambda module, args, output: batches.append(output.detach().cpu().flatten()))
            try:
                result, vector = detector.predict(row['text'], return_embedding=True)
            finally:
                hook.remove()
            logits = torch.cat(batches).tolist()
            assert result['winning_window'] == int(np.argmax(logits))
            record(name, row, trace, logits, vector)
            assert abs(predictions[-1]['score'] - result['score']) <= 2e-7
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    torch.save(winners, OUT / 'winning_embeddings.pt')
    dump(OUT / 'completed.json', dict(completed_at=now(), predictions=len(predictions),
        files={p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'completed.json'}))
    verify()


def verify():
    import torch
    models, rows = check_inputs()
    completed = json.loads((OUT / 'completed.json').read_text())
    for path, expected in completed['files'].items():
        assert sha(OUT / path) == expected, path
    predictions = read_rows(OUT / 'predictions.jsonl')
    coverage = {r['id']: r for r in read_rows(OUT / 'coverage.jsonl')}
    assert len(predictions) == 36
    assert {(p['model'], p['id']) for p in predictions} == {(m, r['id']) for m in models['models'] for r in rows}
    vectors = torch.load(OUT / 'winning_embeddings.pt', map_location='cpu', weights_only=True)
    heads = {name: torch.load(MODELS / name / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)['head'] for name in models['models']}
    error = 0.
    for p in predictions:
        trace = coverage[p['id']]
        assert trace['bounds'] == [list(b) for b in windows(trace['content_tokens'])]
        assert len(p['window_logits']) == len(trace['bounds']) == p['windows']
        assert p['threshold'] == models['models'][p['model']]['threshold']
        assert p['prediction'] == int(p['score'] >= p['threshold'])
        assert p['winning_window'] == max(range(p['windows']), key=lambda i: p['window_logits'][i])
        head = heads[p['model']]
        replay = torch.sigmoid(torch.nn.functional.linear(vectors[f"{p['model']}/{p['id']}"], head['weight'], head['bias'])).item()
        error = max(error, abs(replay-p['score']))
        assert abs(replay-p['score']) <= 2e-6
        assert int(replay >= p['threshold']) == p['prediction']
    assert all(t['complete_attack_windows'] for r in rows if r['label'] for t in [coverage[r['id']]])
    print('VERIFIED: 36 predictions, frozen inputs/models, full window bounds, payload coverage, head replay;', error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'run', 'verify'])
    stage = parser.parse_args().stage
    {'prepare': prepare, 'check': check_inputs, 'run': run, 'verify': verify}[stage]()
