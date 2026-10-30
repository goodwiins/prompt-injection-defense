#!/usr/bin/env python3
"""Inspect training histories and validation winners; never read holdout scores."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from retrain_audited import dump, jsonl
from windowed_followup import load_rows, ranges, sha


def diagnose(source, out):
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(source / 'validation.jsonl')
    report = {'source_validation_sha256': sha(source / 'validation.jsonl'), 'seeds': {},
              'scope': 'Training histories and validation only; winning windows are model evidence, not annotated attack spans'}
    winners_by_seed = {}
    for seed in (17, 42, 73):
        folder = source / f'finetuned_{seed}'
        state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
        head = torch.nn.Linear(768, 1)
        head.load_state_dict(state['head'])
        vectors = torch.load(folder / 'validation_features.pt', weights_only=True, mmap=True)
        predictions = load_rows(folder / 'validation_predictions.jsonl')
        winners = []
        for row, vector, prediction in zip(rows, vectors, predictions, strict=True):
            with torch.no_grad():
                logits = head(vector).flatten()
                index = int(logits.argmax())
                score = float(torch.sigmoid(logits[index]))
            assert row['id'] == prediction['id']
            assert abs(score - prediction['score']) < 2e-6
            start, end = ranges(row['content_tokens'])[index]
            top = torch.topk(logits, min(2, len(logits))).values.tolist()
            winners.append(dict(id=row['id'], label=row['label'], index=index, windows=len(logits),
                                start=start, end=end, score=score, prediction=prediction['prediction'],
                                top_logit_gap=top[0] - top[1] if len(top) > 1 else None))
        jsonl(out / f'validation_winners_{seed}.jsonl', winners)
        history = json.loads((folder / 'history.json').read_text())
        chosen = json.loads((folder / 'selection.json').read_text())
        report['seeds'][str(seed)] = dict(
            epoch_losses=[h['loss'] for h in history],
            validation_auc=[h['validation']['roc_auc'] for h in history],
            validation_recall=[h['validation']['recall']['value'] for h in history],
            selected_epoch=chosen['epoch'], encoder_probe_max_change=chosen['encoder_probe_max_change'],
            head_weight_norm=float(head.weight.detach().norm()), head_bias=float(head.bias.detach()[0]),
            median_top_logit_gap=float(np.median([r['top_logit_gap'] for r in winners if r['top_logit_gap'] is not None])),
            first_window_winners=sum(r['index'] == 0 for r in winners),
            validation_false_positives=sum(r['prediction'] and not r['label'] for r in winners),
            validation_false_negatives=sum(not r['prediction'] and r['label'] for r in winners))
        winners_by_seed[seed] = winners
    report['validation_winner_agreement'] = {
        f'{a}_vs_{b}': sum(x['index'] == y['index'] for x, y in zip(winners_by_seed[a], winners_by_seed[b])) / len(rows)
        for a, b in ((17, 42), (17, 73), (42, 73))}
    dump(out / 'diagnosis.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    diagnose(Path(sys.argv[1]), Path(sys.argv[2]))
