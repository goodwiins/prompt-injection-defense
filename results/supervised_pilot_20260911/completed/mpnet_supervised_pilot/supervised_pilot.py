#!/usr/bin/env python3
"""Small GPU-only MPNet classifier ablation with page-level supervision.

Raw HTML is preserved. A page's score is its maximum window logit, and BCE is
applied once per page. Window labels are never fabricated from page labels.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrain_audited import MODEL, MODEL_REVISION, digest, dump, jsonl, metrics, normalized
from windowed_followup import extract_html, load_rows, ranges, sha

SEEDS = (17, 42, 73)
SPEC = dict(
    model=MODEL, revision=MODEL_REVISION, gpu='NVIDIA A100', seeds=list(SEEDS), epochs=2,
    train_per_class=128, validation_per_class=96, document_batch=4,
    inference_window_batch=32, encoder_lr=1e-5, head_lr=1e-3, weight_decay=.01,
    gradient_clip=1., max_length=384, overlap=96, validation_fpr_budget=.05,
    inputs='unchanged raw HTML, every overlapping token window; no content truncation',
    loss='BCEWithLogitsLoss on maximum window logit per page; no chunk labels',
    comparison='same linear head initialization, data, document order, loss and optimizer settings; encoder frozen versus updated',
    dropout='disabled in both arms; gradients still update the fine-tuned encoder',
    precision='CUDA float16 autocast and GradScaler; parameters remain float32',
    checkpoint_rule='maximum validation recall at empirical FPR <=5%, then ROC-AUC; earliest epoch breaks ties; stop after one non-improving epoch',
    threshold_rule='maximum validation recall at empirical FPR <=5%; highest threshold breaks ties',
    primary_endpoint='paired holdout recall difference at validation-frozen thresholds, accompanied by actual holdout FPR',
    secondary_endpoints=['ROC-AUC', 'average precision', 'balanced accuracy'],
    holdout_interval=[1600, 2100],
    review_exclusions={'perplexity-ai/browsesafe-bench/train/1567':
        'Full raw HTML contains ordinary account notices and official-domain links; attack intent/payload is not established from this isolated page. Quarantine, not a confirmed label error.'},
    limitations=[
        'Small BrowseSafe-only pilot; no general prompt-injection or downstream security claim',
        'Source labels are machine-generated, not independently verified attack-span annotations',
        'A max-pooled page classifier cannot resolve dependencies across distant windows',
        'Five percent is an exploratory empirical validation budget, not a production safety target or guarantee',
        'Lexical filtering does not prove independence of unknown websites, templates or pretraining data',
        'All seeds are reported; three training runs do not establish universal robustness',
        'Prior test rows and scores never enter training or model selection; prior text is used to reject overlap',
    ],
)


def now():
    return datetime.now(timezone.utc).isoformat()


def threshold_at_budget(labels, scores, budget=.05):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(y.tolist()) != {0, 1} or not np.isfinite(p).all():
        raise ValueError('Finite predictions and both validation classes are required')
    candidates = np.unique(np.r_[p, np.nextafter(p.max(), np.inf)])
    allowed = math.floor(budget * int((y == 0).sum()) + 1e-12)
    feasible = [float(t) for t in candidates if int(((p >= t) & (y == 0)).sum()) <= allowed]
    return max(feasible, key=lambda t: (int(((p >= t) & (y == 1)).sum()), t))


def summarize(rows, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    predictions = [dict(id=r['id'], label=r['label'], score=float(p), prediction=int(p >= threshold))
                   for r, p in zip(rows, scores)]
    result = metrics(predictions)
    result.update(threshold=threshold,
                  roc_auc=float(roc_auc_score([r['label'] for r in rows], scores)),
                  average_precision=float(average_precision_score([r['label'] for r in rows], scores)))
    result['balanced_accuracy'] = (result['recall']['value'] + 1 - result['fpr']['value']) / 2
    return predictions, result


def prepare(raw_dir, old, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if (out / 'protocol.json').exists():
        check(out)
        print('Frozen preparation already exists', flush=True)
        return
    if any(out.iterdir()):
        raise ValueError('Preparation requires an empty output directory')
    manifest = load_rows(old / 'manifest.jsonl')
    historical = load_rows(old / 'source_rows.jsonl')
    raw = load_rows(raw_dir / 'browsesafe_train_raw.jsonl') + load_rows(raw_dir / 'browsesafe_test_raw.jsonl')
    assert len(raw) == 4100 and len({r['id'] for r in raw}) == 4100
    membership = {r['id']: r for r in manifest}
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    prior = {r['id']: dict(id=r['id'], text=r['text'], split=membership.get(r['id'], {}).get('split', 'excluded')) for r in historical}
    for row in raw:
        if row['source_split'] == 'test' and row['index'] < 1600:
            prior[row['id']] = dict(id=row['id'], text=extract_html(row['content']), split='test')
    prior = list(prior.values())
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
    previous = vectorizer.fit_transform(r['text'] for r in prior)
    previous_hashes = {digest(normalized(r['text'])) for r in prior}
    previous_tokens = {digest(json.dumps(token(r['text'], truncation=True, max_length=384)['input_ids'])) for r in prior}
    candidates, exclusions = [], []
    # Membership is inherited before sampling. Old held-out rows cannot move to train.
    for row in raw:
        fresh = row['source_split'] == 'test' and 1600 <= row['index'] < 2100
        assigned = 'holdout' if fresh else membership.get(row['id'], {}).get('split')
        if assigned not in ('train', 'validation', 'holdout'):
            continue
        if row['id'] in SPEC['review_exclusions']:
            exclusions.append(dict(id=row['id'], split=assigned, reason=SPEC['review_exclusions'][row['id']]))
            continue
        text = extract_html(row['content'])
        if assigned != 'holdout':
            assert text == membership[row['id']]['text']
            assert row['label'] == membership[row['id']]['label']
        kh = digest(json.dumps(token(text, truncation=True, max_length=384)['input_ids']))
        vec = vectorizer.transform([text])
        scores = (vec @ previous.T).toarray()[0]
        blocked = np.array([r['split'] != 'train' if assigned == 'train' else r['split'] not in ('train', 'validation') if assigned == 'validation' else True for r in prior])
        nearest = float(scores[blocked].max()) if blocked.any() else 0.
        reason = None
        if nearest >= .9:
            reason = 'extracted-text lexical cosine >=0.90 with reserved earlier partition'
        elif fresh and (digest(normalized(text)) in previous_hashes or kh in previous_tokens):
            reason = 'prior extracted-text or first-window token duplicate'
        elif not row['content'].strip():
            reason = 'empty raw HTML'
        if reason:
            exclusions.append(dict(id=row['id'], split=assigned, reason=reason))
            continue
        candidates.append(dict(id=row['id'], label=row['label'], split=assigned, text=row['content'],
                               raw_sha256=digest(row['content']), normalized_raw_sha256=digest(normalized(row['content'])),
                               extracted_text=text, nearest_reserved_cosine=nearest))
    # Within-partition lexical families are thinned before the fixed hash sample.
    selected, selected_vectors, hashes = [], [], set()
    for split in ('validation', 'train', 'holdout'):
        counts = Counter()
        subset = sorted((r for r in candidates if r['split'] == split), key=lambda r: digest('pilot-v1:' + r['id']))
        for row in subset:
            cap = SPEC['train_per_class'] if split == 'train' else SPEC['validation_per_class'] if split == 'validation' else None
            if cap is not None and counts[row['label']] >= cap:
                exclusions.append(dict(id=row['id'], split=split, reason='fixed per-class pilot cap'))
                continue
            vec = vectorizer.transform([row['extracted_text']])
            similar = any(float((vec @ v.T).toarray()[0, 0]) >= .9 for v in selected_vectors)
            if row['normalized_raw_sha256'] in hashes or similar:
                exclusions.append(dict(id=row['id'], split=split, reason='within-pilot raw duplicate or lexical cosine >=0.90'))
                continue
            encoded = token(row['text'], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
            tokens = encoded['input_ids']
            if not tokens:
                raise ValueError('Empty token input')
            row['content_tokens'] = len(tokens)
            row['window_count'] = len(ranges(len(tokens)))
            row['token_sha256'] = digest(json.dumps(tokens))
            row['token_character_bounds'] = [encoded['offset_mapping'][0][0], encoded['offset_mapping'][-1][1]]
            hashes.add(row['normalized_raw_sha256'])
            selected_vectors.append(vec)
            selected.append(row)
            counts[row['label']] += 1
        expected = SPEC['train_per_class'] if split == 'train' else SPEC['validation_per_class'] if split == 'validation' else 100
        if set(counts) != {0, 1} or min(counts.values()) < expected:
            raise ValueError(f'Insufficient {split} rows: {counts}; revise before any scoring')
    for split in ('train', 'validation', 'holdout'):
        jsonl(out / f'{split}.jsonl', sorted((r for r in selected if r['split'] == split), key=lambda r: r['id']))
    jsonl(out / 'exclusions.jsonl', exclusions)
    for name in ('retrain_audited.py', 'windowed_followup.py', 'supervised_pilot.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    shutil.copyfile(old / 'base/classifier.json', out / 'historical_classifier.json')
    shutil.copyfile(old / 'base/threshold.json', out / 'historical_threshold.json')
    protocol = dict(SPEC, frozen_at=now(), source_revision='b506fb5bc7fd4472c8738055a67a0ef6406afdc9',
                    old_manifest_sha256=sha(old / 'manifest.jsonl'),
                    selection='inherit original train/validation; reject lexical overlap with reserved partitions and within pilot; fixed hash sample 128/class train and 96/class validation; all eligible test indices 1600:2100',
                    counts={s: dict(Counter(r['label'] for r in selected if r['split'] == s)) for s in ('train', 'validation', 'holdout')},
                    windows={s: sum(r['window_count'] for r in selected if r['split'] == s) for s in ('train', 'validation', 'holdout')},
                    files={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                    raw_inputs_sha256={p.name: sha(p) for p in raw_dir.glob('browsesafe_*_raw.jsonl')})
    dump(out / 'protocol.json', protocol)
    print('FROZEN', json.dumps({k: protocol[k] for k in ('counts', 'windows', 'frozen_at')}), flush=True)


def check(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        if sha(out / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    if sha(Path(__file__)) != protocol['files']['supervised_pilot.py']:
        raise ValueError('Runner differs from frozen source')
    return protocol


def make_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; CPU training is disabled')
    if 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('This experiment requires the user-requested A100 GPU')
    encoder = SentenceTransformer(MODEL, revision=MODEL_REVISION, device='cuda')
    encoder.max_seq_length = 384
    encoder.eval()
    return encoder


def token_windows(tokenizer, rows):
    documents, ledger = [], []
    for row in rows:
        ids = tokenizer(row['text'], add_special_tokens=False, verbose=False)['input_ids']
        assert len(ids) == row['content_tokens'] and digest(json.dumps(ids)) == row['token_sha256']
        chunks = []
        for start, end in ranges(len(ids)):
            chunks.append(dict(input_ids=tokenizer.build_inputs_with_special_tokens(ids[start:end])))
            ledger.append(dict(id=row['id'], start=start, end=end, token_sha256=digest(json.dumps(ids[start:end]))))
        documents.append(chunks)
    return documents, ledger


def embed(encoder, chunks, grad=False):
    import torch
    inputs = encoder.tokenizer.pad(chunks, padding=True, return_tensors='pt').to('cuda')
    with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=torch.float16):
        return encoder(inputs)['sentence_embedding'].float()


def features(encoder, documents, target=None):
    import torch
    outputs = []
    start = time.monotonic()
    for i, chunks in enumerate(documents):
        values = [embed(encoder, chunks[j:j + SPEC['inference_window_batch']]).cpu() for j in range(0, len(chunks), SPEC['inference_window_batch'])]
        outputs.append(torch.cat(values))
        if i % 32 == 0:
            print(f'Encoding pages {i + 1}/{len(documents)} ({time.monotonic() - start:.0f}s)', flush=True)
    if target:
        torch.save(outputs, target)
    return outputs


def scores_for(head, vectors):
    import torch
    with torch.no_grad():
        return np.array([torch.sigmoid(head(v.to('cuda')).max()).item() for v in vectors])


def initialize(seed):
    import torch
    from transformers import set_seed
    set_seed(seed)
    head = torch.nn.Linear(768, 1).to('cuda')
    return head


def head_digest(head):
    return digest(''.join(v.detach().cpu().numpy().tobytes().hex() for v in head.state_dict().values()))


def train(out):
    import torch
    check(out)
    if (out / 'selection_frozen.json').exists():
        print('All six models already frozen; training skipped', flush=True)
        return
    rows = {s: load_rows(out / f'{s}.jsonl') for s in ('train', 'validation')}
    encoder = make_encoder()
    environment = dict(recorded_at=now(), gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                       packages={p: importlib.metadata.version(p) for p in ('torch', 'sentence-transformers', 'transformers', 'numpy', 'scikit-learn')})
    dump(out / 'environment.json', environment)
    print('CUDA VERIFIED', json.dumps(environment), flush=True)
    documents = {}
    for s in rows:
        documents[s], ledger = token_windows(encoder.tokenizer, rows[s])
        jsonl(out / f'{s}_windows.jsonl', ledger)
    cached = {}
    for s in rows:
        path = out / f'base_{s}_features.pt'
        cached[s] = torch.load(path, weights_only=True) if path.exists() else features(encoder, documents[s], path)
    del encoder
    torch.cuda.empty_cache()
    selections = {}
    for seed in SEEDS:
        for arm in ('frozen', 'finetuned'):
            name = f'{arm}_{seed}'
            target = out / name
            if (target / 'selection.json').exists():
                selections[name] = json.loads((target / 'selection.json').read_text())
                continue
            if target.exists():
                raise RuntimeError(f'Incomplete trial {name}; inspect it before an explicit restart')
            target.mkdir()
            head = initialize(seed)
            initial_head = head_digest(head)
            encoder = make_encoder() if arm == 'finetuned' else None
            if encoder is not None:
                # Fixed dropout behavior makes two-pass max selection an exact subgradient
                # of this max-pooled loss, apart from finite-precision batching differences.
                encoder.eval()
            groups = [dict(params=head.parameters(), lr=SPEC['head_lr'])]
            if encoder is not None:
                groups.append(dict(params=encoder.parameters(), lr=SPEC['encoder_lr']))
            optimizer = torch.optim.AdamW(groups, weight_decay=SPEC['weight_decay'])
            scaler = torch.amp.GradScaler('cuda')
            best, history, order_log, steps = None, [], [], 0
            initial_probe = next(encoder.parameters()).detach().cpu().clone() if encoder is not None else None
            for epoch in range(1, SPEC['epochs'] + 1):
                order = list(range(len(rows['train'])))
                random.Random(seed * 100 + epoch).shuffle(order)
                order_log.append(dict(epoch=epoch, ids=[rows['train'][i]['id'] for i in order]))
                total_loss, started = 0., time.monotonic()
                for offset in range(0, len(order), SPEC['document_batch']):
                    batch = order[offset:offset + SPEC['document_batch']]
                    optimizer.zero_grad(set_to_none=True)
                    if encoder is None:
                        logits = torch.stack([head(cached['train'][i].to('cuda')).max() for i in batch])
                    else:
                        winners = []
                        for i in batch:
                            chunks = documents['train'][i]
                            with torch.no_grad():
                                values = torch.cat([head(embed(encoder, chunks[j:j + SPEC['inference_window_batch']])).flatten() for j in range(0, len(chunks), SPEC['inference_window_batch'])])
                            winners.append(chunks[int(values.argmax())])
                        logits = head(embed(encoder, winners, grad=True)).flatten()
                    labels = torch.tensor([rows['train'][i]['label'] for i in batch], dtype=torch.float32, device='cuda')
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], SPEC['gradient_clip'], error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    steps += 1
                    total_loss += loss.item() * len(batch)
                    if offset % 32 == 0:
                        print(f'TRAIN {name} epoch {epoch} pages {offset + len(batch)}/{len(order)} loss {loss.item():.4f} elapsed {time.monotonic() - started:.0f}s', flush=True)
                val_vectors = cached['validation'] if encoder is None else features(encoder, documents['validation'])
                scores = scores_for(head, val_vectors)
                threshold = threshold_at_budget([r['label'] for r in rows['validation']], scores)
                predictions, summary = summarize(rows['validation'], scores, threshold)
                key = (summary['recall']['value'], summary['roc_auc'])
                history.append(dict(epoch=epoch, loss=total_loss / len(order), validation=summary, seconds=time.monotonic() - started, optimizer_steps=steps))
                print('VALIDATION', name, epoch, json.dumps(summary), flush=True)
                jsonl(target / f'validation_epoch_{epoch}.jsonl', predictions)
                dump(target / 'history.json', history)
                jsonl(target / 'training_order.jsonl', order_log)
                if best is not None and key <= best:
                    break
                best = key
                state = dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()})
                if encoder is not None:
                    state['encoder'] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
                torch.save(state, target / 'checkpoint.pt')
                torch.save(val_vectors, target / 'validation_features.pt')
                jsonl(target / 'validation_predictions.jsonl', predictions)
                chosen = dict(name=name, seed=seed, arm=arm, epoch=epoch, threshold=threshold, validation=summary,
                              initial_head_sha256=initial_head, final_head_sha256=head_digest(head),
                              encoder_probe_max_change=float((next(encoder.parameters()).detach().cpu() - initial_probe).abs().max()) if encoder is not None else 0.,
                              checkpoint_sha256=sha(target / 'checkpoint.pt'))
            chosen['frozen_at'] = now()
            if encoder is not None and chosen['encoder_probe_max_change'] == 0:
                raise RuntimeError('Encoder did not update')
            dump(target / 'selection.json', chosen)
            selections[name] = chosen
            del encoder, head, optimizer, state, val_vectors
            gc.collect()
            torch.cuda.empty_cache()
    for seed in SEEDS:
        assert selections[f'frozen_{seed}']['initial_head_sha256'] == selections[f'finetuned_{seed}']['initial_head_sha256']
    dump(out / 'selection_frozen.json', dict(frozen_at=now(), protocol_sha256=sha(out / 'protocol.json'), selections=selections))
    print('ALL SIX MODELS FROZEN; holdout has not been scored', flush=True)


def evaluate(out):
    import torch
    from xgboost import XGBClassifier
    check(out)
    frozen = json.loads((out / 'selection_frozen.json').read_text())
    assert frozen['protocol_sha256'] == sha(out / 'protocol.json')
    assert set(frozen['selections']) == {f'{a}_{s}' for a in ('frozen', 'finetuned') for s in SEEDS}
    for name, selected in frozen['selections'].items():
        assert sha(out / name / 'checkpoint.pt') == selected['checkpoint_sha256']
    if (out / 'holdout_results.json').exists():
        print('Holdout already evaluated; do not reuse for model selection', flush=True)
        return
    dump(out / 'evaluation_started.json', dict(started_at=now(), selection_sha256=sha(out / 'selection_frozen.json')))
    rows = load_rows(out / 'holdout.jsonl')
    encoder = make_encoder()
    documents, ledger = token_windows(encoder.tokenizer, rows)
    jsonl(out / 'holdout_windows.jsonl', ledger)
    base_vectors = features(encoder, documents, out / 'base_holdout_features.pt')
    historical = XGBClassifier()
    historical.load_model(out / 'historical_classifier.json')
    heads = encoder.encode([r['extracted_text'] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    np.save(out / 'historical_holdout_features.npy', heads)
    original_threshold = json.loads((out / 'historical_threshold.json').read_text())['threshold']
    predictions, reference = summarize(rows, historical.predict_proba(heads)[:, 1], original_threshold)
    jsonl(out / 'historical_holdout_predictions.jsonl', predictions)
    summaries = {}
    for name, selection in frozen['selections'].items():
        state = torch.load(out / name / 'checkpoint.pt', weights_only=True)
        head = initialize(selection['seed'])
        head.load_state_dict(state['head'])
        if selection['arm'] == 'finetuned':
            encoder.load_state_dict(state['encoder'])
            encoder.eval()
            vectors = features(encoder, documents, out / name / 'holdout_features.pt')
        else:
            vectors = base_vectors
        predictions, summaries[name] = summarize(rows, scores_for(head, vectors), selection['threshold'])
        jsonl(out / name / 'holdout_predictions.jsonl', predictions)
        print('HOLDOUT', name, json.dumps(summaries[name]), flush=True)
        del state, head, vectors
        gc.collect()
        torch.cuda.empty_cache()
    dump(out / 'holdout_results.json', dict(completed_at=now(), summaries=summaries, historical_reference=reference,
                                          limitations=SPEC['limitations']))
    dump(out / 'sha256.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'sha256.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--output', type=Path, default=Path('results/supervised_pilot_20260911/pilot'))
    parser.add_argument('--raw', type=Path, default=Path('results/supervised_pilot_20260911'))
    parser.add_argument('--old', type=Path, default=Path('results/colab_audit_20260911/audit_run'))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.stage == 'prepare':
        prepare(args.raw, args.old, args.output)
    elif args.stage == 'train':
        train(args.output)
    else:
        evaluate(args.output)
