#!/usr/bin/env python3
"""Verify downloaded Colab evidence without retraining or choosing on test data."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata
from collections import Counter

import numpy as np
from xgboost import XGBClassifier


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def counts(predictions):
    c = Counter((r['label'], r['prediction']) for r in predictions)
    return dict(n=len(predictions), tp=c[1, 1], tn=c[0, 0], fp=c[0, 1], fn=c[1, 0])


def main(directory):
    root = directory / 'audit_run'
    hashes = read(root / 'sha256.json')
    for name, expected in hashes.items():
        assert (root / name).is_file(), f'Missing artifact: {name}'
        assert sha(root / name) == expected, f'Checksum mismatch: {name}'
    protocol = read(root / 'protocol.json')
    assert sha(directory / 'retrain_audited.py') == protocol['script_sha256']
    assert sha(root / 'manifest.jsonl') == read(root / 'manifest_sha256.json')
    rows = lines(root / 'manifest.jsonl')
    assert len({r['id'] for r in rows}) == len(rows)
    for field in ('text_hash', 'token_hash', 'group', 'component'):
        splits = {}
        for row in rows:
            assert splits.setdefault(row[field], row['split']) == row['split'], field
    for row in rows:
        normalized = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
        assert hashlib.sha256(normalized.encode()).hexdigest() == row['text_hash']
    parts = {s: [r for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    by_id = {r['id']: r for r in rows}
    # Re-tokenize actual model inputs with the downloaded encoder tokenizer.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / 'uniform/encoder', local_files_only=True)
    for row in rows:
        ids = tokenizer(row['text'], truncation=True, max_length=protocol['max_seq_length'])['input_ids']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == row['token_hash']
    for source, n in [('leolee99/NotInject', 339), ('local-synthetic-historical', 247)]:
        subset = [r for r in rows if r['source'] == source]
        assert len(subset) == n and all(r['split'] == 'test' for r in subset)
    reports, paired = {}, {}
    for variant in ('base', 'uniform', 'double'):
        folder = root / variant
        report = read(folder / 'metrics.json')
        frozen = read(folder / 'threshold.json')
        validation = lines(folder / 'validation_predictions.jsonl')
        predictions = lines(folder / 'test_predictions.jsonl')
        assert [r['id'] for r in validation] == [r['id'] for r in parts['validation']]
        assert frozen['validation_ids'] == [r['id'] for r in validation]
        assert [r['id'] for r in predictions] == [r['id'] for r in parts['test']]
        assert all(r['label'] == by_id[r['id']]['label'] for r in validation + predictions)
        assert all(math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in validation + predictions)
        y = np.array([r['label'] for r in validation])
        scores = np.array([r['score'] for r in validation])
        options = sorted({0., 1.0000001, *scores})
        best = max(options, key=lambda t: ((np.mean(scores[y == 1] >= t) + np.mean(scores[y == 0] < t))/2, t))
        assert best == frozen['threshold'] == report['threshold']
        assert all(r['prediction'] == int(r['score'] >= best) for r in predictions)
        clf = XGBClassifier()
        clf.load_model(folder / 'classifier.json')
        embeddings = np.load(folder / 'test_embeddings.npy')
        assert embeddings.shape == (len(predictions), 768) and np.isfinite(embeddings).all()
        recomputed_scores = clf.predict_proba(embeddings)[:, 1]
        np.testing.assert_allclose(recomputed_scores, [r['score'] for r in predictions], rtol=0, atol=1e-7)
        for subset, expected in [(predictions, report['pooled'])] + [([r for r in predictions if r['source'] == s], m) for s,m in report['per_source'].items()]:
            c = counts(subset)
            assert all(c[k] == expected[k] for k in c)
            for name, numerator, denominator in [('accuracy', c['tp']+c['tn'], c['n']), ('recall', c['tp'], c['tp']+c['fn']), ('fpr', c['fp'], c['fp']+c['tn'])]:
                if denominator:
                    assert math.isclose(expected[name]['value'], numerator/denominator, abs_tol=1e-12)
                else:
                    assert expected[name] is None
        if variant != 'base':
            pairs = lines(folder / 'training_pairs.jsonl')
            assert len(pairs) == protocol['pairs_per_epoch'] == 14000
            assert all(by_id[p[k]]['split'] == 'train' for p in pairs for k in ('left', 'right'))
            for pair in pairs:
                a,b = (by_id[pair[k]]['kind'] for k in ('left', 'right'))
                expected = 1. if a == b else .8 if {a,b} == {'safe','trigger'} else 0.
                assert pair['target'] == expected and pair['kind'] == f'{a}/{b}'
            mixture = Counter(p['kind'] for p in pairs)
            assert mixture['injection/trigger'] == (4000 if variant == 'double' else 2333)
        timings = lines(folder / 'single_request_latency.jsonl')
        assert len(timings) == 150 and len({r['id'] for r in timings}) == 150
        assert all(r['id'] in {p['id'] for p in predictions} and r['milliseconds'] > 0 for r in timings)
        for percentile in (50,95):
            assert math.isclose(float(np.percentile([r['milliseconds'] for r in timings], percentile)), report['latency'][f'p{percentile}_ms'], abs_tol=1e-10)
        reports[variant] = report
        paired[variant] = np.array([r['prediction'] == r['label'] for r in predictions])
    delta = dict(double_correct_uniform_wrong=int(np.sum(paired['double'] & ~paired['uniform'])), uniform_correct_double_wrong=int(np.sum(paired['uniform'] & ~paired['double'])))
    result = dict(status='passed', artifact_count=len(hashes), script_sha256=protocol['script_sha256'], split_counts={s:len(rs) for s,rs in parts.items()}, paired_correctness=delta, summary={v:dict(accuracy=r['pooled']['accuracy']['value'], recall=r['pooled']['recall']['value'], fpr=r['pooled']['fpr']['value'], official_notinject_fpr=r['per_source']['leolee99/NotInject']['fpr']['value']) for v,r in reports.items()}, limits=['Checksums establish artifact consistency, not independent GPU replay.', 'Saved classifiers were replayed on saved test embeddings; encoder inference and fitting were not repeated.', 'Single seed, capped data, and unmeasured semantic near-duplicates limit generalization.'])
    (directory / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
