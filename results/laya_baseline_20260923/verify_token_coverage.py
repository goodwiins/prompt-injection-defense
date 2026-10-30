"""Independently retokenize the frozen validation inputs locally; no model weights."""
import json
import os
from pathlib import Path
import sys

os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from mpnet_inference import sha, windows
from retrain_audited import digest, dump, jsonl
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

protocol = json.loads((OUT / 'protocol.json').read_text())
env = json.loads((OUT / 'environment.json').read_text())
source = ROOT / 'results/span_supervision_20260911/inputs/validation.jsonl'
assert sha(source) == protocol['data_sha256']
snapshot = Path(snapshot_download(protocol['model'], revision=protocol['revision'],
    allow_patterns=['tokenizer/*'], local_files_only=True))
for name in ('tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json'):
    assert sha(snapshot / name) == env['loaded_model_files'][name], name
tokenizer = AutoTokenizer.from_pretrained(snapshot / 'tokenizer', local_files_only=True)
expected_path = OUT / 'token_coverage_expected.jsonl'
if sys.argv[1] == 'prepare':
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    refs = {r['id']: r for r in map(json.loads, (source.parent / 'construction_reference.jsonl').read_text().splitlines())}
    expected = []
    for row in rows:
        text = row['text'].replace(tokenizer.mask_token, ' ')
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids = encoded['input_ids']
        bounds = list(windows(len(ids), env['content_window_tokens'], protocol['overlap']))
        covering = None
        if row.get('condition') in ('direct', 'quoted'):
            ref = refs[row['id']]
            start = len(row['text'][:ref['insertion_start']].replace(tokenizer.mask_token, ' '))
            end = len(row['text'][:ref['insertion_end']].replace(tokenizer.mask_token, ' '))
            relevant = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and a < end]
            covering = [i for i, (a, b) in enumerate(bounds) if a <= relevant[0] and b > relevant[-1]]
            assert covering, row['id']
        expected.append(dict(id=row['id'], token_sha256=digest(json.dumps(ids)),
            content_tokens=len(ids), bounds=[list(b) for b in bounds],
            window_token_hashes=[digest(json.dumps(ids[a:b])) for a, b in bounds],
            complete_insertion_windows=covering))
    jsonl(expected_path, expected)
    print('LOCAL TOKENIZATION COMPLETE', len(expected), 'pages;', sum(len(r['bounds']) for r in expected), 'windows')
else:
    assert sys.argv[1] == 'verify'
    expected = [json.loads(line) for line in expected_path.read_text().splitlines()]
    actual = [json.loads(line) for line in (OUT / 'predictions.jsonl').read_text().splitlines()]
    for e, a in zip(expected, actual, strict=True):
        for key in ('id', 'token_sha256', 'content_tokens', 'bounds', 'window_token_hashes'):
            assert e[key] == a[key], (e['id'], key)
    dump(OUT / 'token_coverage_verification.json', dict(pages=len(actual),
        windows=sum(a['windows'] for a in actual),
        all_ten_insertions_have_complete_windows=sum(e['complete_insertion_windows'] is not None for e in expected) == 10,
        files={p.name: sha(p) for p in [Path(__file__), source, expected_path, OUT / 'predictions.jsonl']}))
    print('LOCAL COVERAGE VERIFIED: every token and window hash, all ten direct/quoted insertions')
