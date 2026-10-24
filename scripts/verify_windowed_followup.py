#!/usr/bin/env python3
"""Independent checks of the completed follow-up and saved classifier predictions."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
from transformers import AutoTokenizer
from xgboost import XGBClassifier


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(out, old):
    protocol = read(out / 'protocol.json')
    hashes = read(out / 'sha256.json')
    for name, expected in hashes.items():
        assert sha(out / name) == expected, name
    assert sha(out / 'windowed_followup.py') == protocol['script_sha256']
    assert sha(out / 'development.jsonl') == protocol['development_sha256']
    assert sha(out / 'sealed_holdout.jsonl') == protocol['holdout_sha256']
    assert sha(old / 'manifest.jsonl') == protocol['old_manifest_sha256']
    assert sha(old / 'source_rows.jsonl') == protocol['old_raw_sha256']
    development = rows(out / 'development.jsonl')
    original = rows(old / 'manifest.jsonl')
    assert development == [r for r in original if r['split'] in ('train','validation')]
    val = [r for r in development if r['split'] == 'validation']
    holdout = rows(out / 'sealed_holdout.jsonl')
    exclusions = rows(out / 'holdout_exclusions.jsonl')
    assert len(holdout) == 919 and Counter(r['label'] for r in holdout) == {1:445,0:474}
    assert len(exclusions) == 81
    assert len({r['id'] for r in holdout+exclusions}) == 1000
    assert {int(r['id'].split('/')[-1]) for r in holdout+exclusions} == set(range(600,1600))
    raw = rows(old / 'source_rows.jsonl')
    for field in ('id','text_hash','token_hash'):
        assert len({r[field] for r in holdout}) == len(holdout)
        assert not {r[field] for r in holdout}.intersection(r[field] for r in raw)
    # Recompute lexical eligibility in batches, independently of the preparation loop.
    vectorizer = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=50000)
    previous = vectorizer.fit_transform(r['text'] for r in raw)
    fresh = vectorizer.transform(r['text'] for r in holdout)
    maximum = (fresh @ previous.T).max(axis=1).toarray().ravel()
    assert np.all(maximum < .9)
    np.testing.assert_allclose(maximum, [r['nearest_prior_cosine'] for r in holdout], atol=1e-12)
    self_similarity = (fresh @ fresh.T).toarray()
    np.fill_diagonal(self_similarity,0)
    assert self_similarity.max() < .9
    frozen = read(out / 'validation_frozen.json')
    started = read(out / 'evaluation_started.json')
    result = read(out / 'holdout_results.json')
    assert started['validation_freeze_sha256'] == sha(out / 'validation_frozen.json')
    assert datetime.fromisoformat(protocol['frozen_at']) < datetime.fromisoformat(frozen['frozen_at']) < datetime.fromisoformat(started['started_at']) < datetime.fromisoformat(result['completed_at'])
    for name, expected in frozen['files'].items():
        assert sha(out / name) == expected
    tok = AutoTokenizer.from_pretrained(old / 'uniform/encoder',local_files_only=True)
    score_sets = {}
    window_counts = {}
    for split, subset in [('validation',val),('holdout',holdout)]:
        windows = rows(out / f'{split}_windows.jsonl')
        vectors = np.load(out / f'{split}_embeddings.npy')
        assert vectors.shape == (len(windows),768) and np.isfinite(vectors).all()
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
        window_counts[split] = len(windows)
        by_row = {i:[] for i in range(len(subset))}
        for w in windows:
            assert subset[w['row_index']]['id'] == w['id']
            by_row[w['row_index']].append(w)
        for i,row in enumerate(subset):
            tokens = tok(row['text'].strip(),add_special_tokens=False,truncation=False,verbose=False)['input_ids']
            seen = by_row[i]
            assert seen[0]['start'] == 0 and seen[-1]['end'] == len(tokens)
            for j,w in enumerate(seen):
                assert w['content_tokens'] == len(tokens)
                assert w['start'] == j*286 and w['end'] == min(j*286+382,len(tokens))
                ids = tok.build_inputs_with_special_tokens(tokens[w['start']:w['end']])
                assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == w['input_sha256']
        owners = np.array([r['row_index'] for r in windows])
        head_indexes = [i for i,r in enumerate(windows) if r['start'] == 0]
        for seed in (17,42,73):
            clf = XGBClassifier()
            clf.load_model(out / f'classifier_{seed}.json')
            window_scores = clf.predict_proba(vectors)[:,1]
            maxima = np.full(len(subset),-np.inf)
            np.maximum.at(maxima, owners, window_scores)
            for method, predicted in [('base',window_scores[head_indexes]),('windowed',maxima)]:
                key = f'{method}_{seed}'
                saved = rows(out / f'{split}_predictions_{key}.jsonl')
                assert [r['id'] for r in saved] == [r['id'] for r in subset]
                assert [r['label'] for r in saved] == [r['label'] for r in subset]
                np.testing.assert_allclose(predicted,[r['score'] for r in saved],atol=1e-7,rtol=0)
                y = np.array([r['label'] for r in saved])
                scores = np.array([r['score'] for r in saved])
                threshold = frozen['summaries'][key]['threshold']
                if split == 'validation':
                    choices = {0.,1.0000001,*scores}
                    expected = max(choices,key=lambda t:((np.mean(scores[y==1]>=t)+np.mean(scores[y==0]<t))/2,t))
                    assert threshold == expected
                decisions = (scores>=threshold).astype(int)
                assert [r['prediction'] for r in saved] == decisions.tolist()
                expected = (frozen if split=='validation' else result)['summaries'][key]
                assert expected['threshold'] == threshold
                c = Counter(zip(y,decisions))
                for label,count in dict(n=len(y),tp=c[1,1],tn=c[0,0],fp=c[0,1],fn=c[1,0]).items():
                    assert expected[label] == count
                for label,value in dict(roc_auc=roc_auc_score(y,scores),balanced_accuracy=balanced_accuracy_score(y,decisions),average_precision=average_precision_score(y,scores)).items():
                    assert abs(expected[label]-value)<1e-12
                for label,value in dict(accuracy=np.mean(y==decisions),recall=np.mean(decisions[y==1]),fpr=np.mean(decisions[y==0])).items():
                    assert abs(expected[label]['value']-value)<1e-12
                score_sets[split,key] = scores
    historical = rows(old / 'base/validation_predictions.jsonl')
    old_diff = float(np.max(np.abs(score_sets['validation','base_42']-[r['score'] for r in historical])))
    assert old_diff <= 1e-7, 'Frozen base seed 42 failed to reproduce original validation scores'
    # Nominal paired, class-stratified bootstrap; unknown website groups limit inference.
    rng = np.random.default_rng(20260911)
    y = np.array([r['label'] for r in holdout])
    indices = [np.flatnonzero(y==label) for label in (0,1)]
    draws = [np.concatenate([rng.choice(i,len(i),replace=True) for i in indices]) for _ in range(1000)]
    intervals = {}
    for seed in (17,42,73):
        a,b = (score_sets['holdout',f'{method}_{seed}'] for method in ('base','windowed'))
        delta = roc_auc_score(y,b)-roc_auc_score(y,a)
        assert abs(delta-result['paired_auc_deltas'][str(seed)])<1e-12
        samples = [roc_auc_score(y[i],b[i])-roc_auc_score(y[i],a[i]) for i in draws]
        intervals[str(seed)] = dict(delta=float(delta),nominal_paired_stratified_bootstrap_95=np.percentile(samples,[2.5,97.5]).tolist())
    verification = dict(status='passed',checksums=len(hashes),holdout_n=len(holdout),window_counts=window_counts,
        maximum_holdout_prior_lexical_cosine=float(maximum.max()),
        historical_base_seed42_max_validation_score_difference=old_diff,paired_auc_intervals=intervals,
        limits=['Saved classifier replay and token window verification; no independent GPU encoder replay.',
                'Bootstrap treats documents as independent; unknown website/template dependence is not modeled.',
                'Encoder is frozen; three seeds cover classifier randomness only.'])
    (out / 'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1]),Path('results/colab_audit_20260911/audit_run'))
