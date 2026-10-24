#!/usr/bin/env python3
"""Inspect only existing train/validation text and predictions; never read test scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer


MARKER = re.compile(r"ignore.{0,35}(?:previous|prior|above|instructions)|"
                    r"(?:system|developer)\s+prompt|override.{0,30}(?:instructions|safety)|"
                    r"(?:send|forward|transmit|exfiltrate).{0,60}(?:password|secret|prompt)|"
                    r"you are now.{0,30}(?:mode|assistant)", re.I)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(root, output):
    output.mkdir(parents=True, exist_ok=True)
    # The manifest is inspected for split membership only before text filtering.
    rows = [r for r in read_rows(root / 'manifest.jsonl') if r['split'] in ('train', 'validation')]
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    tok = AutoTokenizer.from_pretrained(root / 'uniform/encoder', local_files_only=True)
    content_budget = 384 - tok.num_special_tokens_to_add(pair=False)
    for row in rows:
        encoded = tok(row['text'], add_special_tokens=False, truncation=False,
                      return_offsets_mapping=True, verbose=False)
        row['content_tokens'] = len(encoded['input_ids'])
        cut = encoded['offset_mapping'][content_budget - 1][1] if row['content_tokens'] > content_budget else len(row['text'])
        row['cut_character'] = cut
        row['visible_head'] = row['text'][:cut]
        row['marker_in_head'] = bool(MARKER.search(row['visible_head']))
        row['marker_after_head'] = bool(MARKER.search(row['text'][cut:]))
    predictions = {}
    for variant in ('base', 'uniform', 'double'):
        saved = read_rows(root / variant / 'validation_predictions.jsonl')
        assert [r['id'] for r in saved] == [r['id'] for r in validation]
        threshold = json.loads((root / variant / 'threshold.json').read_text())['threshold']
        predictions[variant] = {r['id']: dict(score=r['score'], prediction=int(r['score'] >= threshold)) for r in saved}
    slices = []
    for split in ('train', 'validation'):
        for source in sorted({r['source'] for r in rows}):
            subset = [r for r in rows if r['source'] == source and r['split'] == split]
            entry = dict(split=split, source=source, n=len(subset), attacks=sum(r['label'] for r in subset),
                         truncated=sum(r['content_tokens'] > content_budget for r in subset),
                         median_content_tokens=float(np.median([r['content_tokens'] for r in subset])) if subset else None,
                         marker_only_after_head=sum(r['marker_after_head'] and not r['marker_in_head'] for r in subset))
            if split == 'validation' and subset:
                entry['variants'] = {}
                for variant, scores in predictions.items():
                    c = Counter((r['label'], scores[r['id']]['prediction']) for r in subset)
                    entry['variants'][variant] = dict(tp=c[1, 1], tn=c[0, 0], fp=c[0, 1], fn=c[1, 0],
                        roc_auc=float(roc_auc_score([r['label'] for r in subset], [scores[r['id']]['score'] for r in subset])) if len({r['label'] for r in subset}) == 2 else None)
            slices.append(entry)
    # Lexical similarity flags candidates; it does not establish semantic equivalence.
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
    xtrain = tfidf.fit_transform(r['text'] for r in train)
    xval = tfidf.transform(r['text'] for r in validation)
    sims = (xval @ xtrain.T).toarray()
    nearest = sims.argmax(axis=1)
    details = []
    for i, row in enumerate(validation):
        neighbor = train[int(nearest[i])]
        details.append(dict(row, predictions={v:p[row['id']] for v,p in predictions.items()},
                            nearest_train_id=neighbor['id'], nearest_train_label=neighbor['label'],
                            lexical_cosine=float(sims[i, nearest[i]])))
    (output / 'validation_details.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in details))
    rates = {s: np.mean([r['label'] for r in train if r['source'] == s]) for s in {r['source'] for r in train}}
    summary = dict(content_token_budget=content_budget, slices=slices,
        training_template_groups=dict(Counter(r['group'] for r in train if r['source'] == 'synthetic-additional-templates')),
        validation_template_groups=dict(Counter(r['group'] for r in validation if r['source'] == 'synthetic-additional-templates')),
        lexical_near_duplicates_at_090=sum(r['lexical_cosine'] >= .9 for r in details),
        lexical_near_duplicates_opposite_labels_at_090=sum(r['lexical_cosine'] >= .9 and r['label'] != r['nearest_train_label'] for r in details),
        source_only_validation_auc=float(roc_auc_score([r['label'] for r in validation], [rates[r['source']] for r in validation])),
        marker_caveat='Lexical flags locate candidate instructions, not attack ground truth. No automatic relabeling.',
        manifest_sha256=hashlib.sha256((root / 'manifest.jsonl').read_bytes()).hexdigest())
    write(output / 'summary.json', summary)
    raw_path = output / 'validation_browsesafe_raw.json'
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        html_counts, removed = Counter(), []
        for row in details:
            if row['source'] != 'perplexity-ai/browsesafe-bench':
                continue
            source_row = raw[row['id'].split('/')[-1]]
            assert int(source_row['label'] == 'yes') == row['label']
            html = source_row['content']
            match = MARKER.search(html)
            extracted = bool(MARKER.search(row['text']))
            html_counts[str((row['label'], bool(match), extracted, row['marker_in_head']))] += 1
            if row['label'] and match and not extracted:
                removed.append(dict(id=row['id'], snippet=html[max(0,match.start()-120):match.end()+450]))
        assert sum(html_counts.values()) == 410
        write(output / 'html_marker_diagnostics.json', dict(counts=dict(html_counts), removed_examples=removed))
    review = []
    for source in sorted({r['source'] for r in details}):
        for label in (0, 1):
            errors = [r for r in details if r['source'] == source and r['label'] == label and any(p['prediction'] != label for p in r['predictions'].values())]
            review.extend(sorted(errors, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:3])
    tails = [r for r in details if r['label'] == 1 and r['marker_after_head'] and not r['marker_in_head']]
    review.extend(sorted(tails, key=lambda r:r['id'])[:3])
    unique = {r['id']:r for r in review}
    write(output / 'review_sample.json', list(unique.values()))
    print(json.dumps(summary, indent=2))
    print('Review sample:', len(unique))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/colab_audit_20260911/audit_run'))
    parser.add_argument('--output', type=Path, default=Path('results/validation_diagnosis_20260911'))
    args = parser.parse_args()
    main(args.root, args.output)
