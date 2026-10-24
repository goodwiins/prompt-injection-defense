#!/usr/bin/env python3
"""Prepare unscored, grouped HTML augmentation candidates for human review."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from retrain_audited import digest, dump, jsonl, normalized
from windowed_followup import extract_html, load_rows, sha
from mpnet_inference import windows

OUT = ROOT / 'results/html_coverage_correction_20260911'
OLD = ROOT / 'results/generalization_20260911'
MODELS = ROOT / 'results/supervised_schedule_20260911/completed/mpnet_supervised_schedule_replay'
FAMILIES = {'train': ['Information Retrieval', 'Content Creation', 'Anagramming', 'Space Removal & Grouping'],
            'validation': ['Learning and Tutoring', 'Programming Help', 'Misspelling Intentionally']}
FIELDS = ['id', 'source_sha256', 'label', 'reviewer', 'reviewer_type', 'reviewed_at',
          'human_review_attested', 'independent_review_attested', 'evidence', 'notes']


def source_items(path):
    """Read the pinned large JSON array one task at a time using the stdlib."""
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buf, eof = '', False

        def fill():
            nonlocal buf, eof
            chunk = stream.read(4 * 1024 * 1024)
            buf += chunk
            eof = not chunk

        def whitespace():
            nonlocal buf
            while not buf.strip() and not eof:
                fill()
            buf = buf.lstrip()

        whitespace()
        assert buf.startswith('['), 'Expected a JSON array'
        buf = buf[1:]
        first = True
        while True:
            whitespace()
            if buf.startswith(']'):
                buf = buf[1:] + stream.read()
                assert not buf.strip(), 'Trailing JSON content'
                return
            if not first:
                assert buf.startswith(','), 'Expected comma between tasks'
                buf = buf[1:]
                whitespace()
                assert not buf.startswith(']'), 'Trailing comma'
            while True:
                try:
                    item, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            assert isinstance(item, dict)
            yield item
            buf, first = buf[end:], False


def insert(raw, text, position):
    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.starts, self.ends, self.middle = [], [], []
            self.offsets = [0]
            for line in raw.split('\n'):
                self.offsets.append(self.offsets[-1] + len(line) + 1)

        def absolute_position(self):
            line, col = self.getpos()
            return self.offsets[line - 1] + col

        def handle_starttag(self, tag, attrs):
            if tag == 'body':
                self.starts.append(self.absolute_position() + len(self.get_starttag_text()))

        def handle_endtag(self, tag):
            offset = self.absolute_position()
            if tag == 'body':
                self.ends.append(offset)
            elif tag in ('p', 'div', 'section', 'article', 'ul', 'form', 'main'):
                self.middle.append(raw.index('>', offset) + 1)

    parser = Anchors()
    parser.feed(raw)
    assert parser.starts and parser.ends, 'Complete body required'
    start, end = parser.starts[0], parser.ends[-1]
    assert start < end
    if position == 'front':
        at = start
    elif position == 'end':
        at = end
    else:
        anchors = [p for p in parser.middle if start < p < end]
        assert anchors, 'A real element boundary is required for middle placement'
        at = min(anchors, key=lambda p: abs(p - (start + end) / 2))
    addition = '\n<aside>' + html.escape(text) + '</aside>\n'
    result = raw[:at] + addition + raw[at:]
    assert result[:at] + result[at + len(addition):] == raw
    return result, at, addition


def prepare():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoTokenizer
    assert not (OUT / 'protocol.json').exists(), 'Frozen preparation already exists'
    upstream = json.loads((OUT / 'source/upstream.json').read_text())
    source = OUT / 'source/mind2web_train_6.json'
    assert sha(source) == upstream['lfs']['oid']
    assert source.stat().st_size == upstream['size']
    templates = json.loads((OUT / 'source/bipia_train.json').read_text())
    source_paths = [ROOT / 'results/colab_audit_20260911/audit_run/source_rows.jsonl',
                    ROOT / 'results/supervised_pilot_20260911/browsesafe_train_raw.jsonl',
                    ROOT / 'results/supervised_pilot_20260911/browsesafe_test_raw.jsonl',
                    MODELS / 'fresh_candidates.jsonl']
    prior = {r['id']: r['text'] for r in load_rows(source_paths[0])}
    for path in source_paths[1:]:
        prior.update({r['id']: r['content'] for r in load_rows(path)})
    diagnostic = load_rows(OLD / 'candidates.jsonl')
    prior.update({r['id']: (OLD / r['path']).read_text() for r in diagnostic})
    previous = list(prior.items())
    extracted = [extract_html(text) for _, text in previous]
    normalized_prior = [normalized(text) for text in extracted]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
    matrix = vectorizer.fit_transform(extracted)
    tokenizer = AutoTokenizer.from_pretrained(MODELS / 'finetuned_17/inference_assets', local_files_only=True)
    raw_hashes = {digest(normalized(text)) for _, text in previous}
    text_hashes = {digest(text) for text in normalized_prior}
    first_hashes = {digest(json.dumps(tokenizer(text, truncation=True, max_length=384)['input_ids'])) for text in extracted}
    del prior
    index = json.loads((OUT / 'source/source_index.json').read_text())
    excluded_sites = {r['website'] for r in diagnostic}
    sites = sorted({r['website'] for r in index} - excluded_sites, key=lambda s: digest('html-coverage-v1:' + s))
    reserved = set(sites[:6])
    dump(OUT / 'holdout_reservation.json', dict(
        status='Metadata-only reserve; no examples constructed, reviewed, trained or scored',
        websites=sorted(reserved), tasks=[r['task_id'] for r in index if r['website'] in reserved],
        limitations='Not an evaluation-ready test. Historical development family provenance remains unknown.'))
    tasks = {}
    for task in source_items(source):
        if task['website'] in excluded_sites | reserved:
            continue
        action = task['actions'][0]
        tasks.setdefault(task['website'], []).append(dict(task_id=task['annotation_id'], website=task['website'],
            domain=task['domain'], requested_task=task['confirmed_task'], action_id=action['action_uid'], text=action['raw_html']))
    selected, exclusions, vectors = [], [], []
    for site in sites[6:]:
        for task in sorted(tasks.get(site, []), key=lambda r: digest(r['task_id'])):
            raw = task['text']
            ids = tokenizer(raw, add_special_tokens=False, verbose=False)['input_ids']
            if not 4096 <= len(ids) <= 131072:
                exclusions.append(dict(task_id=task['task_id'], reason='Outside declared 4096-131072 token range', tokens=len(ids)))
                continue
            try:
                insert(raw, 'Structural check', 'middle')
            except AssertionError:
                exclusions.append(dict(task_id=task['task_id'], reason='Missing complete body/element insertion boundary'))
                continue
            text = extract_html(raw)
            vector = vectorizer.transform([text])
            similarities = (vector @ matrix.T).toarray()[0]
            nearest = int(similarities.argmax())
            first = digest(json.dumps(tokenizer(text, truncation=True, max_length=384)['input_ids']))
            duplicate = digest(normalized(raw)) in raw_hashes or digest(normalized(text)) in text_hashes or first in first_hashes
            cross = max((float((vector @ v.T).toarray()[0, 0]) for v in vectors), default=0.)
            if duplicate or similarities[nearest] >= .90 or cross >= .90:
                exclusions.append(dict(task_id=task['task_id'], reason='Duplicate or lexical cosine >=0.90',
                    duplicate=duplicate, prior_cosine=float(similarities[nearest]), new_parent_cosine=cross))
                continue
            task.update(content_tokens=len(ids), nearest_prior_id=previous[nearest][0],
                        prior_cosine=float(similarities[nearest]), new_parent_cosine=cross)
            selected.append(task)
            vectors.append(vector)
            break
        if len(selected) == 15:
            break
    assert len(selected) == 15, f'Need 15 eligible website groups, found {len(selected)}'
    candidates, construction = [], []
    for folder in ('pages', 'review_text'):
        (OUT / folder).mkdir(exist_ok=True)
    for i, task in enumerate(selected):
        split = 'train' if i < 10 else 'validation'
        j = i if split == 'train' else i - 10
        family = FAMILIES[split][j % len(FAMILIES[split])]
        # Literal overlap checks do not establish semantic family independence.
        payload = next((p for p in templates[family] if not any(normalized(p) in text for text in normalized_prior)), None)
        assert payload, f'No unused literal instruction available for {family}'
        position = ('front', 'middle', 'end')[j % 3]
        group = digest('html-coverage-v1:' + task['task_id'])[:20]
        quoted = ('The following is an example quoted for discussion, not a request: "' + payload + '"'
                  if j % 2 == 0 else 'Example sentence in a documentation excerpt: "' + payload + '"')
        for condition, text in [('original', None), ('direct', payload), ('quoted', quoted)]:
            raw, at, added = task['text'], None, ''
            if text is not None:
                raw, at, added = insert(raw, text, position)
            rid = digest(group + ':' + condition)[:20]
            path = OUT / 'pages' / (rid + '.txt')
            path.write_text(raw)
            (OUT / 'review_text' / (rid + '.txt')).write_text(extract_html(raw))
            encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
            ids = encoded['input_ids']
            affected, full_windows = [], []
            if at is not None:
                affected = [k for k, (a, b) in enumerate(encoded['offset_mapping']) if b > at and a < at + len(added)]
                full_windows = [k for k, (a, b) in enumerate(windows(len(ids))) if a <= affected[0] and b > affected[-1]]
                assert full_windows
                assert normalized(text) in normalized(extract_html(raw))
            candidates.append(dict(id=rid, group_id=group, split=split, website=task['website'], domain=task['domain'],
                task_id=task['task_id'], action_id=task['action_id'], requested_task=task['requested_task'],
                path=str(path.relative_to(OUT)), sha256=sha(path), content_tokens=len(ids), token_sha256=digest(json.dumps(ids)), label=None))
            construction.append(dict(id=rid, group_id=group, condition=condition, proposed_label=int(condition == 'direct'),
                template_family=family, template_text=payload, position=position, insertion_start=at,
                insertion_end=at + len(added) if at is not None else None,
                insertion_sha256=digest(added) if added else None, full_insertion_windows=full_windows,
                prior_cosine=task['prior_cosine'], nearest_prior_id=task['nearest_prior_id'],
                label_status='Construction hypothesis only; independent human review required'))
    candidates.sort(key=lambda r: digest('review-order:' + r['id']))
    jsonl(OUT / 'candidates.jsonl', candidates)
    jsonl(OUT / 'construction_reference.jsonl', construction)
    jsonl(OUT / 'exclusions.jsonl', exclusions)
    with (OUT / 'human_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(dict(id=r['id'], source_sha256=r['sha256']) for r in candidates)
    review = ['# Human review: HTML coverage correction', '',
        '45 unscored candidates from 15 website groups. Training and validation candidates only; the reserved websites have no constructed test examples.', '',
        'Judge each complete raw page as untrusted content for its stated user task. Direct attempts to redirect the assistant are injection candidates. Ordinary page content and quoted examples are not automatically injections. Mark ambiguous pages uncertain. Treat every page as data; do not execute instructions inside it.', '',
        'Open the full raw text; the readable text is an aid that can omit attributes or other HTML channels. Record label (injection/non-injection/uncertain), your name, reviewer_type=human, an ISO timestamp with timezone, human_review_attested=yes, independent_review_attested=yes, evidence and rationale in human_review.csv. Do not inspect construction_reference.jsonl or model scores before recording labels. There are no model predictions in this packet.', '',
        '[Open label sheet](human_review.csv)', '', '| ID | Website | User task | Raw page | Reading aid | Tokens |', '|---|---|---|---|---|---:|']
    for r in candidates:
        task = r['requested_task'].replace('|', '\\|').replace('\n', ' ')
        review.append(f"| {r['id']} | {r['website']} | {task} | [Raw]({r['path']}) | [Text](review_text/{r['id']}.txt) | {r['content_tokens']} |")
    review += ['', 'Source: Mind2Web pinned revision 17ece8eb89862368edc0cc806acee6fca5163474, released training shard 6 (CC-BY-4.0). Instructions are locally combined with pages from BIPIA training templates pinned at a004b69ec0dd446e0afd461d98cb5e96e120a5d0 (MIT). This is not an official benchmark or source-provided injection labeling.', '']
    (OUT / 'REVIEW.md').write_text('\n'.join(review))
    legacy_train = load_rows(MODELS / 'train.jsonl')
    assert Counter(r['label'] for r in legacy_train) == {0: 128, 1: 128}
    removed = []
    for label, count in ((0, 20), (1, 10)):
        removed += sorted((r['id'] for r in legacy_train if r['label'] == label),
                          key=lambda rid: digest('html-coverage-replacement-v1:' + rid))[:count]
    dump(OUT / 'replacement_plan.json', dict(status='Proposed only; requires adjudicated human labels',
        removed_legacy_train_ids=removed, retained_legacy_train_ids=[r['id'] for r in legacy_train if r['id'] not in removed],
        added_candidate_train_ids=[r['id'] for r in candidates if r['split'] == 'train'],
        expected_train_label_counts={'0': 128, '1': 128}, expected_validation_label_counts={'0': 106, '1': 101},
        conditions='No export or training until labels are complete; disagreements or uncertainty require adjudication'))
    recipe = dict(intervention='Replace 30 of 256 existing training pages with 10 direct and 20 original/quoted candidates, preserving 128/128 class balance. Add 15 new validation candidates to existing 192.',
        pending_human_labels=True, architecture='Unchanged MPNet, raw HTML, all 382-token windows, overlap 96, max window logit',
        comparison='Retain frozen encoder controls and all seeds 17/42/73; reuse six-epoch training and validation-only selection procedures',
        test_policy='Reserved website metadata only; final labels, sampling frame and family provenance must be settled before evaluation',
        limitations=['New-cohort websites and published template categories are disjoint across splits; historical family overlap remains unknown',
            'Construction labels are hypotheses, not human truth', 'Balanced training replacement requires adjudication before export',
            'Quoted framing can be a shortcut; varied positions and shared wrapper do not eliminate all construction artifacts',
            'Published category separation does not prove semantic template independence'])
    dump(OUT / 'correction_recipe.json', recipe)
    frozen = list(OUT.rglob('*')) + source_paths + [OLD / 'candidates.jsonl', OLD / 'human_review.csv',
        OLD / 'models_frozen.json', MODELS / 'train.jsonl', MODELS / 'validation.jsonl', Path(__file__),
        ROOT / 'training/retrain_audited.py', ROOT / 'training/windowed_followup.py', ROOT / 'training/mpnet_inference.py']
    dump(OUT / 'protocol.json', dict(prepared_at=datetime.now(timezone.utc).isoformat(),
        stage='Candidate preparation; training not started', prior_inputs_screened=len(previous),
        new_website_groups=15, reserved_website_groups=len(reserved), excluded_diagnostic_websites=sorted(excluded_sites),
        template_categories=FAMILIES, token_range=[4096, 131072], similarity_exclusion=.90,
        selection='Pinned shard 6; hash-order website reserve, then hash-order task per website; fixed eligibility checks; no model scores',
        files={str(p.relative_to(ROOT)): sha(p) for p in frozen if p.is_file() and p != OUT / 'human_review.csv' and p.suffix != '.log'}))
    return check()


def check():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        assert sha(ROOT / name) == expected, name
    candidates = load_rows(OUT / 'candidates.jsonl')
    construction = {r['id']: r for r in load_rows(OUT / 'construction_reference.jsonl')}
    assert len(candidates) == len(construction) == 45 and all(r['label'] is None for r in candidates)
    sites = {s: {r['website'] for r in candidates if r['split'] == s} for s in FAMILIES}
    assert len(sites['train']) == 10 and len(sites['validation']) == 5 and not sites['train'] & sites['validation']
    reserve = set(json.loads((OUT / 'holdout_reservation.json').read_text())['websites'])
    assert not reserve & (sites['train'] | sites['validation'])
    assert not set(FAMILIES['train']) & set(FAMILIES['validation'])
    plan = json.loads((OUT / 'replacement_plan.json').read_text())
    legacy = {r['id']: r for r in load_rows(MODELS / 'train.jsonl')}
    assert len(plan['removed_legacy_train_ids']) == 30 and len(plan['retained_legacy_train_ids']) == 226
    assert set(plan['removed_legacy_train_ids']).isdisjoint(plan['retained_legacy_train_ids'])
    assert set(plan['removed_legacy_train_ids'] + plan['retained_legacy_train_ids']) == set(legacy)
    planned_labels = [legacy[rid]['label'] for rid in plan['retained_legacy_train_ids']]
    planned_labels += [construction[rid]['proposed_label'] for rid in plan['added_candidate_train_ids']]
    assert Counter(planned_labels) == {0: 128, 1: 128}
    for group in {r['group_id'] for r in candidates}:
        members = [r for r in candidates if r['group_id'] == group]
        assert len(members) == 3 and len({r['split'] for r in members}) == 1
        parent = next(r for r in members if construction[r['id']]['condition'] == 'original')
        raw = (OUT / parent['path']).read_text()
        for row in members:
            assert sha(OUT / row['path']) == row['sha256']
            spec = construction[row['id']]
            if spec['insertion_start'] is not None:
                text = (OUT / row['path']).read_text()
                a, b = spec['insertion_start'], spec['insertion_end']
                assert text[:a] + text[b:] == raw and digest(text[a:b]) == spec['insertion_sha256']
                assert spec['full_insertion_windows']
    with (OUT / 'human_review.csv').open(newline='') as stream:
        reviews = list(csv.DictReader(stream))
    assert len(reviews) == 45 and {r['id'] for r in reviews} == {r['id'] for r in candidates}
    by_id = {r['id']: r for r in candidates}
    pending, uncertain, disagreements = [], [], []
    for r in reviews:
        assert r['source_sha256'] == by_id[r['id']]['sha256']
        complete = r['label'] in ('injection', 'non-injection', 'uncertain')
        complete &= r['reviewer_type'] == 'human' and bool(r['reviewer'].strip())
        complete &= r['human_review_attested'] == r['independent_review_attested'] == 'yes'
        complete &= bool(r['evidence'].strip()) and bool(r['notes'].strip())
        try:
            complete &= datetime.fromisoformat(r['reviewed_at']).tzinfo is not None
        except ValueError:
            complete = False
        if not complete:
            pending.append(r['id'])
        elif r['label'] == 'uncertain':
            uncertain.append(r['id'])
        elif int(r['label'] == 'injection') != construction[r['id']]['proposed_label']:
            disagreements.append(r['id'])
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), frozen_hashes_passed=True,
        candidates=45, split_counts=dict(Counter(r['split'] for r in candidates)),
        prospective_website_splits_disjoint=True, published_template_categories_disjoint=True,
        historical_family_independence_verified=False, pending_human_reviews=len(pending),
        uncertain_labels=uncertain, construction_disagreements_requiring_adjudication=disagreements,
        ready_to_freeze_training_labels=not (pending or uncertain or disagreements),
        training_started=False, final_test_ready=False)
    dump(OUT / 'preflight.json', result)
    print(json.dumps(result, indent=2))
    return 2 if pending or uncertain or disagreements else 0


def self_check():
    raw = '<html><body data-x="</div>"><script>var a="</div>";</script><p>Alpha</p><div>Beta</div></body></html>'
    for position in ('front', 'middle', 'end'):
        changed, at, added = insert(raw, '<script>example</script>', position)
        assert changed[:at] + changed[at + len(added):] == raw
        assert '&lt;script&gt;' in added
        assert not raw.index('<script>') < at < raw.index('</script>')
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / 'source.json'
        p.write_text('[{"x":1}, {"y":"é"}]')
        assert list(source_items(p)) == [{'x': 1}, {'y': 'é'}]
        for malformed in ('', '[{}', '[{},]', '[{}{}]', '[{}]x'):
            p.write_text(malformed)
            try:
                list(source_items(p))
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError('Malformed source accepted')
    print('PASS: streamed JSON checks, original recovery, escaped text, safe insertion boundaries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'self-check'])
    stage = parser.parse_args().stage
    sys.exit({'prepare': prepare, 'check': check, 'self-check': self_check}[stage]())
