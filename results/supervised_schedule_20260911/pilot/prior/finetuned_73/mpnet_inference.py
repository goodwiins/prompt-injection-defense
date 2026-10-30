#!/usr/bin/env python3
"""Portable, offline inference for this MPNet raw-HTML research classifier."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def windows(size, width=382, overlap=96):
    if size <= 0 or width <= overlap or overlap < 0:
        raise ValueError('Nonempty token input and valid window sizes required')
    start = 0
    while True:
        end = min(start + width, size)
        yield start, end
        if end == size:
            break
        start += width - overlap


def export(folder, assets):
    selection = json.loads((folder / 'selection.json').read_text())
    if selection['arm'] != 'finetuned':
        raise ValueError('Export requires the full fine-tuned encoder checkpoint')
    assert sha(folder / 'checkpoint.pt') == selection['checkpoint_sha256']
    target = folder / 'inference_assets'
    target.mkdir(exist_ok=True)
    names = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.txt', '1_Pooling/config.json']
    for name in names:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets / name, dest)
    script = folder / 'mpnet_inference.py'
    if script.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, script)
    config = dict(seed=selection['seed'], epoch=selection['epoch'], threshold=selection['threshold'],
                  model='sentence-transformers/all-mpnet-base-v2', revision='e8c3b32edf5434bc2275fc9bab85f82640a19130',
                  content_tokens=382, overlap=96, window_batch=32,
                  input='unchanged UTF-8 text or raw HTML; no extraction or truncation',
                  pooling='masked mean, L2 normalization, linear head, maximum window logit, sigmoid',
                  reference_precision='A100 CUDA float16 encoder autocast; float32 head; CPU uses float32',
                  limitations='Research classifier. A negative prediction does not authorize tool access or establish safety.',
                  files={str(p.relative_to(folder)): sha(p) for p in [folder / 'checkpoint.pt', script, *target.rglob('*')] if p.is_file()})
    (folder / 'inference_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (folder / 'INFERENCE.md').write_text(
        '# MPNet research inference package\n\n'
        'Keep checkpoint.pt, inference_config.json, mpnet_inference.py and inference_assets/ together. '
        'The encoder, tokenizer, pooling, head and threshold load offline. No pretrained weights are downloaded.\n\n'
        'Reference environment: torch==2.9.1, transformers==4.57.3, sentence-transformers==5.1.2, numpy==2.3.5.\n\n'
        'Run `python mpnet_inference.py predict --package . --input page.html --device cuda` on an A100. '
        'Use `--device cpu` for float32 inference; small numerical differences from the CUDA reference are possible. '
        'Long inputs are processed completely and may take longer. Empty input raises an error.\n\n'
        'The output is a detector score and decision, not a security authorization.\n')
    return config


class Detector:
    def __init__(self, folder, device='cpu'):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from sentence_transformers.models import Pooling, Normalize
        self.folder, self.device = Path(folder), device
        self.config = json.loads((self.folder / 'inference_config.json').read_text())
        for name, expected in self.config['files'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha(self.folder / path) != expected:
                raise ValueError(f'Package integrity check failed: {name}')
        if not math.isfinite(self.config['threshold']):
            raise ValueError('Finite threshold required')
        assets = self.folder / 'inference_assets'
        config = AutoConfig.from_pretrained(assets, local_files_only=True)
        if config.model_type != 'mpnet' or config.hidden_size != 768:
            raise ValueError('This package loader supports the audited MPNet architecture only')
        self.tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
        self.encoder = AutoModel.from_config(config)
        self.pool = Pooling(**json.loads((assets / '1_Pooling/config.json').read_text()))
        self.normalize = Normalize()
        self.head = torch.nn.Linear(768, 1)
        state = torch.load(self.folder / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
        prefix = '0.auto_model.'
        if not all(k.startswith(prefix) for k in state['encoder']):
            raise ValueError('Unexpected encoder parameter structure')
        self.encoder.load_state_dict({k[len(prefix):]: v for k, v in state['encoder'].items()}, strict=True)
        self.head.load_state_dict(state['head'], strict=True)
        self.encoder.to(device).eval()
        self.head.to(device).eval()

    def predict(self, text, return_embedding=False):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Nonempty text required')
        ids = self.tokenizer(text, add_special_tokens=False, verbose=False)['input_ids']
        bounds = list(windows(len(ids), self.config['content_tokens'], self.config['overlap']))
        values, vectors = [], []
        with torch.inference_mode():
            for offset in range(0, len(bounds), self.config['window_batch']):
                chunks = [dict(input_ids=self.tokenizer.build_inputs_with_special_tokens(ids[a:b]))
                          for a, b in bounds[offset:offset + self.config['window_batch']]]
                inputs = self.tokenizer.pad(chunks, padding=True, return_tensors='pt').to(self.device)
                autocast = torch.autocast('cuda', dtype=torch.float16) if self.device == 'cuda' else nullcontext()
                with autocast:
                    tokens = self.encoder(**inputs).last_hidden_state
                    features = self.normalize(self.pool(dict(token_embeddings=tokens, attention_mask=inputs['attention_mask'])))
                    embeddings = features['sentence_embedding'].float()
                values.append(self.head(embeddings).flatten())
                if return_embedding:
                    vectors.append(embeddings.cpu())
            logits = torch.cat(values)
            winner = int(logits.argmax())
            score = float(torch.sigmoid(logits[winner]))
        result = dict(score=score, threshold=self.config['threshold'], prediction=int(score >= self.config['threshold']),
                    windows=len(bounds), winning_window=winner, winning_token_bounds=bounds[winner], content_tokens=len(ids))
        return (result, torch.cat(vectors)[winner]) if return_embedding else result


def replay(folder, data, expected, destination):
    import numpy as np
    import torch
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()
    model = Detector(folder, 'cuda')
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    references = [json.loads(line) for line in expected.read_text().splitlines()]
    outputs = []
    for i, (row, reference) in enumerate(zip(rows, references, strict=True)):
        assert row['id'] == reference['id']
        outputs.append(dict(id=row['id'], **model.predict(row['text'])))
        if i % 32 == 0:
            print(f'Package replay {folder.name}: {i + 1}/{len(rows)}', flush=True)
    error = float(np.max(np.abs(np.array([r['score'] for r in outputs]) - [r['score'] for r in references])))
    assert error <= 2e-6, error
    assert [r['prediction'] for r in outputs] == [r['prediction'] for r in references]
    destination.write_text(json.dumps(dict(n=len(outputs), max_absolute_score_error=error,
        identical_decisions=True, gpu=torch.cuda.get_device_name(), input_sha256=sha(data),
        expected_sha256=sha(expected), config_sha256=sha(folder / 'inference_config.json'), predictions=outputs), indent=2) + '\n')
    print('PACKAGE REPLAY PASSED', folder.name, len(outputs), error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'predict', 'replay'])
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.stage == 'export':
        export(args.package, args.assets)
        print('Exported portable inference metadata and assets:', args.package)
    elif args.stage == 'predict':
        print(json.dumps(Detector(args.package, args.device).predict(args.input.read_text())))
    else:
        replay(args.package, args.input, args.expected, args.output)
