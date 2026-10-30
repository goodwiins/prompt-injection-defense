#!/usr/bin/env python3
"""One controlled correction: frozen MPNet, max score over overlapping token windows.

Classifier fitting still uses original training heads. Seeds vary XGBoost only.
Preparation reserves unused BrowseSafe rows without inspecting their predictions.
Validation freezes every threshold before final holdout inference is permitted.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_audited import MODEL, MODEL_REVISION, SOURCES, choose_threshold, digest, dump, jsonl, normalized, metrics

SEEDS = (17, 42, 73)
SOURCE = 'perplexity-ai/browsesafe-bench'
PROTOCOL = dict(
    seeds=list(SEEDS), seed_scope='XGBoost subsampling only; encoder frozen for every run',
    intervention='inference token coverage only: first window versus maximum score over all windows',
    max_seq_length=384, content_tokens=382, overlap_tokens=96, stride=286,
    classifier_training='original train split, first 384 tokens, unchanged labels and XGBoost parameters',
    threshold_rule='maximum validation balanced accuracy; highest threshold breaks ties',
    fresh_holdout=dict(source=SOURCE, revision=SOURCES[SOURCE], split='test', start=600, stop=1600,
        selection='all eligible rows in fixed index interval; no label balancing or outcome selection',
        exclusions='prior source IDs, normalized full text, first-window token hash, or character TF-IDF cosine >=0.90 against every prior raw input; same checks within fresh holdout'),
    primary_endpoint='paired ROC-AUC delta on fresh BrowseSafe holdout for each seed',
    secondary_endpoints=['recall', 'false-positive rate', 'balanced accuracy', 'average precision'],
    success_rule='positive holdout ROC-AUC delta for all three seeds; report each FPR and recall tradeoff without selecting a winning seed',
    selection_policy='no search or seed selection; validation thresholds only; evaluate both frozen methods once even if validation fails',
    limitations=['HTML extraction remains unchanged and can remove hidden payloads',
        'document labels are not verified span labels; no automatic relabeling',
        'training source imbalance and synthetic template quality remain unchanged',
        'fresh means unused by this experiment, not unknown to model pretraining or globally unpublished',
        'character similarity filtering does not guarantee semantic or website independence',
        'fresh holdout covers BrowseSafe only, not NotInject or downstream attack success',
        'three classifier seeds do not measure encoder-training seed robustness',
        'research detector; not a standalone security gate'])


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ranges(length):
    if length < 1:
        raise ValueError('Empty token input')
    spans = [(0, min(length, 382))]
    while spans[-1][1] < length:
        start = spans[-1][0] + 286
        spans.append((start, min(start + 382, length)))
    return spans


def extract_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = ' '.join(soup.get_text(separator=' ').split())
    if len(text) < 50 and len(html) > 100:
        text = html[:500] + ' [TEXT]: ' + text
    return text


def prepare(old, out):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    if (out / 'protocol.json').exists():
        check(out)
        print('Existing frozen preparation verified.', flush=True)
        return
    if any(out.iterdir()):
        raise ValueError('Use an empty directory for a new protocol')
    if sha(Path(__file__).with_name('retrain_audited.py')) != '06c908c571486b62313af603406e2c3544184cc28f57e304e97d4491a06c9a07':
        raise ValueError('Original helper source changed')
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    assert tok.num_special_tokens_to_add(pair=False) == 2
    raw = load_rows(old / 'source_rows.jsonl')
    manifest = load_rows(old / 'manifest.jsonl')
    development = [r for r in manifest if r['split'] in ('train', 'validation')]
    # The old test is used only to reject overlap. It never enters fitting/thresholding.
    hashes = {r['text_hash'] for r in raw}
    token_hashes = {r['token_hash'] for r in raw}
    old_ids = {r['id'] for r in raw}
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
    previous = vectorizer.fit_transform(r['text'] for r in raw)
    dataset = load_dataset(SOURCE, split='test', revision=SOURCES[SOURCE], streaming=True)
    accepted, rejected, accepted_vectors = [], [], []
    for index, row in enumerate(islice(dataset, 1600)):
        if index < 600:
            continue
        rid = f'{SOURCE}/test/{index}'
        if row['label'] not in ('yes', 'no') or not isinstance(row['content'], str):
            raise ValueError('Unexpected source schema')
        text = extract_html(row['content'])
        th = digest(normalized(text))
        tokens = tok(text, truncation=True, max_length=384)['input_ids']
        kh = digest(json.dumps(tokens))
        reason, nearest = None, None
        if not text.strip():
            reason = 'empty extracted input'
        elif rid in old_ids or th in hashes or kh in token_hashes:
            reason = 'prior or within-holdout ID/text/first-window duplicate'
        else:
            vec = vectorizer.transform([text])
            nearest = float((vec @ previous.T).max())
            if nearest >= .9:
                reason = 'lexical cosine >=0.90 with prior raw input'
            elif accepted_vectors and max(float((vec @ v.T).toarray()[0, 0]) for v in accepted_vectors) >= .9:
                reason = 'lexical cosine >=0.90 within fresh holdout'
        if reason:
            rejected.append(dict(id=rid, reason=reason, nearest_prior_cosine=nearest))
            continue
        hashes.add(th)
        token_hashes.add(kh)
        accepted_vectors.append(vec)
        accepted.append(dict(id=rid, source=SOURCE, source_split='test', text=text,
            label=int(row['label'] == 'yes'), text_hash=th, token_hash=kh,
            raw_html_sha256=digest(row['content']), nearest_prior_cosine=nearest))
    if len(accepted) < 100 or min(Counter(r['label'] for r in accepted).values(), default=0) < 30 or {r['label'] for r in accepted} != {0, 1}:
        raise ValueError('Insufficient fresh eligible holdout; do not change selection after scoring')
    jsonl(out / 'development.jsonl', development)
    jsonl(out / 'sealed_holdout.jsonl', accepted)
    jsonl(out / 'holdout_exclusions.jsonl', rejected)
    shutil.copyfile(__file__, out / 'windowed_followup.py')
    shutil.copyfile(Path(__file__).with_name('retrain_audited.py'), out / 'retrain_audited.py')
    protocol = dict(PROTOCOL, frozen_at=datetime.now(timezone.utc).isoformat(),
        model=MODEL, model_revision=MODEL_REVISION, old_manifest_sha256=sha(old / 'manifest.jsonl'),
        old_raw_sha256=sha(old / 'source_rows.jsonl'), script_sha256=sha(Path(__file__)),
        development_sha256=sha(out / 'development.jsonl'), holdout_sha256=sha(out / 'sealed_holdout.jsonl'),
        exclusion_sha256=sha(out / 'holdout_exclusions.jsonl'),
        counts=dict(train=sum(r['split'] == 'train' for r in development),
                    validation=sum(r['split'] == 'validation' for r in development),
                    fresh_holdout=len(accepted), holdout_labels=dict(Counter(r['label'] for r in accepted)),
                    excluded=len(rejected)))
    dump(out / 'protocol.json', protocol)
    print('FROZEN', json.dumps(protocol['counts']), 'holdout', protocol['holdout_sha256'], flush=True)


def check(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    assert protocol['script_sha256'] == sha(Path(__file__)), 'Source changed after preparation'
    assert sha(out / 'retrain_audited.py') == '06c908c571486b62313af603406e2c3544184cc28f57e304e97d4491a06c9a07'
    for key, name in [('development_sha256', 'development.jsonl'), ('holdout_sha256', 'sealed_holdout.jsonl'), ('exclusion_sha256', 'holdout_exclusions.jsonl')]:
        assert protocol[key] == sha(out / name), name
    return protocol


def encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError('This follow-up must run on the Colab GPU')
    result = SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cuda')
    result.max_seq_length = 384
    result.eval()
    return result


def encode_windows(model, rows, out, stem):
    import torch
    tok = model.tokenizer
    heads = model.encode([r['text'] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    records, features, head_positions = [], [], []
    for i, row in enumerate(rows):
        tokens = tok(row['text'].strip(), add_special_tokens=False, truncation=False, verbose=False)['input_ids']
        head_positions.append(len(records))
        for start, end in ranges(len(tokens)):
            ids = tok.build_inputs_with_special_tokens(tokens[start:end])
            records.append(dict(id=row['id'], row_index=i, start=start, end=end, content_tokens=len(tokens), input_sha256=digest(json.dumps(ids))))
            features.append(dict(input_ids=ids, attention_mask=[1] * len(ids)))
    vectors = np.empty((len(records), heads.shape[1]), dtype=np.float32)
    vectors[head_positions] = heads
    head_set = set(head_positions)
    extra = [i for i in range(len(records)) if i not in head_set]
    began = time.monotonic()
    for offset in range(0, len(extra), 32):
        indexes = extra[offset:offset+32]
        batch = tok.pad([features[i] for i in indexes], padding=True, return_tensors='pt')
        with torch.inference_mode():
            result = model({k:v.to('cuda') for k,v in batch.items()})['sentence_embedding']
            vectors[indexes] = torch.nn.functional.normalize(result, p=2, dim=1).cpu().numpy()
        if offset % 1024 == 0:
            print(f'{stem}: additional windows {min(offset+32,len(extra))}/{len(extra)}, {time.monotonic()-began:.0f}s', flush=True)
    assert np.isfinite(vectors).all()
    np.save(out / f'{stem}_embeddings.npy', vectors)
    jsonl(out / f'{stem}_windows.jsonl', records)
    return vectors, records


def document_scores(clf, vectors, windows, n):
    scores = clf.predict_proba(vectors)[:, 1]
    heads = np.empty(n, dtype=np.float32)
    maximum = np.full(n, -np.inf, dtype=np.float32)
    for score, row in zip(scores, windows):
        i = row['row_index']
        if row['start'] == 0:
            heads[i] = score
        maximum[i] = max(maximum[i], score)
    assert np.isfinite(heads).all() and np.isfinite(maximum).all() and np.all(maximum >= heads)
    return dict(base=heads, windowed=maximum)


def report(rows, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score
    predictions = [dict(id=r['id'], label=r['label'], source=r['source'], score=float(p), prediction=int(p >= threshold)) for r,p in zip(rows, scores)]
    y = [r['label'] for r in rows]
    result = metrics(predictions)
    result.update(roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                  balanced_accuracy=float(balanced_accuracy_score(y, [p['prediction'] for p in predictions])))
    return predictions, result


def validate(out):
    import importlib.metadata
    import platform
    import torch
    from xgboost import XGBClassifier
    check(out)
    if (out / 'validation_frozen.json').exists():
        print('Validation already frozen; not fitting again.', flush=True)
        return
    rows = load_rows(out / 'development.jsonl')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'validation']
    model = encoder()
    dump(out / 'environment.json', dict(python=platform.python_version(), gpu=torch.cuda.get_device_name(), packages={p:importlib.metadata.version(p) for p in ('torch', 'sentence-transformers', 'transformers', 'xgboost', 'numpy', 'scikit-learn', 'datasets')}))
    training = model.encode([r['text'] for r in train], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    np.save(out / 'train_embeddings.npy', training)
    vectors, windows = encode_windows(model, val, out, 'validation')
    summaries = {}
    frozen_files = {}
    for seed in SEEDS:
        clf = XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=6, min_child_weight=1,
            subsample=.8, colsample_bytree=.8, gamma=.1, reg_alpha=.01, reg_lambda=1,
            eval_metric='auc', tree_method='hist', n_jobs=2, random_state=seed)
        clf.fit(training, [r['label'] for r in train])
        clf.save_model(out / f'classifier_{seed}.json')
        frozen_files[f'classifier_{seed}.json'] = sha(out / f'classifier_{seed}.json')
        for method, scores in document_scores(clf, vectors, windows, len(val)).items():
            threshold = choose_threshold([r['label'] for r in val], scores)
            predictions, summary = report(val, scores, threshold)
            key = f'{method}_{seed}'
            summary['threshold'] = threshold
            summary['per_source'] = {s:metrics([r for r in predictions if r['source'] == s]) for s in sorted({r['source'] for r in val})}
            jsonl(out / f'validation_predictions_{key}.jsonl', predictions)
            frozen_files[f'validation_predictions_{key}.jsonl'] = sha(out / f'validation_predictions_{key}.jsonl')
            summaries[key] = summary
            print('VALIDATION', key, json.dumps({k:summary[k] for k in ('roc_auc','balanced_accuracy','recall','fpr')}), flush=True)
    dump(out / 'validation_frozen.json', dict(frozen_at=datetime.now(timezone.utc).isoformat(), summaries=summaries, files=frozen_files))


def evaluate(out):
    from xgboost import XGBClassifier
    check(out)
    if (out / 'holdout_results.json').exists():
        print('Final holdout already evaluated; not evaluating again.', flush=True)
        return
    frozen = json.loads((out / 'validation_frozen.json').read_text())
    for name, expected in frozen['files'].items():
        assert sha(out / name) == expected, 'Changed frozen classifier or validation predictions'
    dump(out / 'evaluation_started.json', dict(started_at=datetime.now(timezone.utc).isoformat(), validation_freeze_sha256=sha(out / 'validation_frozen.json')))
    rows = load_rows(out / 'sealed_holdout.jsonl')
    model = encoder()
    vectors, windows = encode_windows(model, rows, out, 'holdout')
    summaries = {}
    for seed in SEEDS:
        clf = XGBClassifier()
        clf.load_model(out / f'classifier_{seed}.json')
        for method, scores in document_scores(clf, vectors, windows, len(rows)).items():
            key = f'{method}_{seed}'
            threshold = frozen['summaries'][key]['threshold']
            predictions, summary = report(rows, scores, threshold)
            summary['threshold'] = threshold
            jsonl(out / f'holdout_predictions_{key}.jsonl', predictions)
            summaries[key] = summary
    deltas = {str(seed):summaries[f'windowed_{seed}']['roc_auc']-summaries[f'base_{seed}']['roc_auc'] for seed in SEEDS}
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), summaries=summaries, paired_auc_deltas=deltas,
                  positive_auc_delta_all_seeds=all(d>0 for d in deltas.values()), limitations=PROTOCOL['limitations'])
    dump(out / 'holdout_results.json', result)
    dump(out / 'sha256.json', {p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != 'sha256.json'})
    print('HOLDOUT COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','validate','evaluate'))
    parser.add_argument('--old', type=Path, default=Path('/content/audit_run'))
    parser.add_argument('--output', type=Path, default=Path('/content/window_followup'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare(args.old, args.output)
    elif args.stage == 'validate':
        validate(args.output)
    else:
        evaluate(args.output)
