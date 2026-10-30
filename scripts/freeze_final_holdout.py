#!/usr/bin/env python3
"""Freeze the October 2026 final holdout before any further MPNet training.

prepare  Select and write the holdout once. Nothing is scored here.
check    Re-verify every recorded hash.
Guard    Every later training or threshold-selection run must drop each row Guard.reason() names.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from retrain_audited import MODEL, MODEL_REVISION, SOURCES, digest, dump, jsonl, normalized
from windowed_followup import extract_html, load_rows, sha

OUT = ROOT / 'results/final_holdout_20261030'
BROWSESAFE = 'perplexity-ai/browsesafe-bench'
SATML = 'ethz-spylab/ctf-satml24'
LLMAIL = 'microsoft/llmail-inject-challenge'
FP_EMAILS = 'data/emails_for_fp_tests.json'
P2_LABELS = 'data/labelled_unique_submissions_phase2.json'
SPEC = dict(
    seed=20261030, base_model=MODEL, base_revision=MODEL_REVISION,
    source_revisions={s: SOURCES[s] for s in (BROWSESAFE, SATML, LLMAIL)},
    browsesafe_test_interval=[2600, 3680],
    satml_prior_accepted=1500, llmail_prior_phase1_rows=500,
    team_fraction=0.25, per_team_cap=100, per_source_target=1000, cosine_threshold=0.90,
    llmail_text_format='Subject of the email: {subject}.   Body: {body}',
    labels={
        BROWSESAFE: 'source page label (yes/no)',
        SATML: '1: first user message of an attack chat',
        LLMAIL + '/Phase2': "1: unique submission with attack_attempt True and reason 'api_triggered' "
                            '(the attack made the tool call); judge-only, Unclear and False rows are not tested',
        LLMAIL + '/emails_for_fp_tests': '0: challenge-authored benign email',
    },
    exclusion_rules=[
        'prior row ID, normalized text (raw or extracted) or first-window token hash',
        'character TF-IDF cosine >= 0.90 with any prior input',
        'character TF-IDF cosine >= 0.90 with an already-accepted holdout row',
    ],
    training_rules=[
        'never train, validate, or select thresholds on the BrowseSafe test split',
        'never use any row, from any phase or split, of a reserved SaTML or LLMail team',
        'drop any row matching a holdout ID, normalized text or first-window token hash',
        'drop any row with character TF-IDF cosine >= 0.90 to a holdout row',
        'SaTML and LLMail training rows must carry their team; rows without one are rejected',
    ],
    limitations=[
        'no benign trigger-word (over-defense) rows: NotInject is consumed and no human-reviewed replacement exists',
        'SaTML and LLMail are injection-only; their FPR comes from the 203 LLMail emails and BrowseSafe pages',
        'fresh means unused by this repository, not unknown to model pretraining',
        'lexical filtering does not prove semantic or website independence',
    ])


def first_window(tokenizer, text):
    return digest(json.dumps(tokenizer(text, truncation=True, max_length=384)['input_ids']))


def as_text(text):
    return extract_html(text) if '<' in text and '>' in text else ' '.join(text.split())


def prior_inputs():
    """Every text an earlier experiment trained, tuned or scored. Over-inclusion only shrinks the holdout."""
    prior = {}
    for path in sorted((ROOT / 'results').rglob('*.jsonl')):
        if OUT in path.parents or any(s in path.name for s in ('window', 'prediction', 'pairs')):
            continue
        for row in load_rows(path):
            if not isinstance(row, dict):
                continue
            text = next((row[k] for k in ('text', 'content') if isinstance(row.get(k), str)), None)
            if text is None and isinstance(row.get('path'), str) and (path.parent / row['path']).is_file():
                text = (path.parent / row['path']).read_text(errors='replace')
            if text and text.strip():
                prior[digest(text)] = (str(row.get('id', '')), text)
    return list(prior.values())


class Screen:
    def __init__(self, prior, tokenizer):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tokenizer = tokenizer
        extracted = [as_text(text) for _, text in prior]
        self.ids = {rid for rid, _ in prior if rid}
        self.hashes = {digest(normalized(t)) for _, t in prior} | {digest(normalized(t)) for t in extracted}
        self.tokens = {first_window(tokenizer, t) for t in extracted}
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
        self.prior = self.vectorizer.fit_transform(extracted)
        self.accepted = None

    def admit(self, rid, text):
        """Return (row fields, None) for an eligible row or (None, exclusion)."""
        from scipy.sparse import vstack
        th, kh = digest(normalized(text)), first_window(self.tokenizer, text)
        if not text.strip():
            return None, dict(id=rid, reason='empty input')
        if rid in self.ids or th in self.hashes or kh in self.tokens:
            return None, dict(id=rid, reason='prior or within-holdout ID/text/first-window duplicate')
        vec = self.vectorizer.transform([text])
        nearest = float((vec @ self.prior.T).max())
        if nearest >= SPEC['cosine_threshold']:
            return None, dict(id=rid, reason='lexical cosine >=0.90 with prior input', nearest_prior_cosine=nearest)
        if self.accepted is not None and float((vec @ self.accepted.T).max()) >= SPEC['cosine_threshold']:
            return None, dict(id=rid, reason='lexical cosine >=0.90 within holdout', nearest_prior_cosine=nearest)
        self.hashes.add(th)
        self.tokens.add(kh)
        self.accepted = vec if self.accepted is None else vstack([self.accepted, vec]).tocsr()
        return dict(text_hash=th, token_hash=kh, nearest_prior_cosine=nearest), None


def select_teams(candidates, prior_groups, screen, source, rows, exclusions):
    """Reserve a seeded fraction of eligible teams; test at most per_team_cap rows from each."""
    by_team = defaultdict(list)
    for c in candidates:
        by_team[c['group']].append(c)
    eligible = sorted(g for g in by_team if g not in prior_groups)
    random.Random(f"{SPEC['seed']}:{source}").shuffle(eligible)
    reserved = eligible[:max(1, round(len(eligible) * SPEC['team_fraction']))]
    taken = 0
    for team in reserved:
        kept = 0
        for c in by_team[team]:
            if kept == SPEC['per_team_cap'] or taken == SPEC['per_source_target']:
                break
            fields, excluded = screen.admit(c['id'], c['text'])
            if excluded:
                exclusions.append(excluded)
                continue
            rows.append({**c, **fields})
            kept += 1
            taken += 1
    return reserved, len(eligible)


def prepare():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f'{OUT} already exists; a frozen holdout is never regenerated')
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    prior = prior_inputs()
    print(f'Screening against {len(prior)} prior inputs', flush=True)
    screen = Screen(prior, tokenizer)
    prior_text = {rid: text for rid, text in prior}
    rows, exclusions, reservation = [], [], dict(groups=[], eligible_teams={})

    start, stop = SPEC['browsesafe_test_interval']
    pages = load_dataset(BROWSESAFE, split='test', revision=SOURCES[BROWSESAFE])
    assert len(pages) == stop, 'BrowseSafe test size changed'
    for i in range(start, stop):
        page = pages[i]
        if page['label'] not in ('yes', 'no'):
            raise ValueError('Unexpected BrowseSafe label')
        rid = f'{BROWSESAFE}/test/{i}'
        fields, excluded = screen.admit(rid, extract_html(page['content']))
        if excluded:
            exclusions.append(excluded)
            continue
        rows.append(dict(id=rid, source=BROWSESAFE, source_split='test', index=i, group=f'browsesafe-test:{i}',
                         text=extract_html(page['content']), label=int(page['label'] == 'yes'),
                         label_basis=SPEC['labels'][BROWSESAFE], raw_sha256=digest(page['content']), **fields))

    chats = load_dataset(SATML, 'interaction_chats', split='attack', revision=SOURCES[SATML])
    candidates, prior_groups, accepted, checked = [], set(), 0, 0
    for i, chat in enumerate(chats):
        history = chat['history']
        if not (history and history[0]['role'] == 'user' and history[0]['content'].strip()):
            continue
        rid, team = f'{SATML}/attack/{i}', f"satml-team:{chat['user']['team']}"
        if accepted < SPEC['satml_prior_accepted']:
            prior_groups.add(team)
            accepted += 1
            if rid in prior_text:
                assert prior_text[rid] == history[0]['content'], f'Row order differs from the audit run: {rid}'
                checked += 1
            continue
        candidates.append(dict(id=rid, source=SATML, source_split='attack', index=i, group=team,
                               text=history[0]['content'], label=1, label_basis=SPEC['labels'][SATML]))
    assert checked, 'No SaTML audit rows matched; row order cannot be confirmed'
    reserved, eligible = select_teams(candidates, prior_groups, screen, SATML, rows, exclusions)
    reservation['groups'] += reserved
    reservation['eligible_teams'][SATML] = eligible

    phase1 = load_dataset(LLMAIL, split='Phase1', revision=SOURCES[LLMAIL], streaming=True)
    prior_groups = set()
    for i, sub in zip(range(SPEC['llmail_prior_phase1_rows']), phase1):
        prior_groups.add(f"llmail-team:{sub['team_id']}")
        rid = f'{LLMAIL}/Phase1/{i}'
        assert rid not in prior_text or prior_text[rid] == sub['body'], f'Row order differs from the audit run: {rid}'
    labels_path = hf_hub_download(LLMAIL, P2_LABELS, repo_type='dataset', revision=SOURCES[LLMAIL])
    labels = {normalized(k): v for k, v in json.loads(Path(labels_path).read_text()).items()}
    phase2 = load_dataset(LLMAIL, split='Phase2', revision=SOURCES[LLMAIL])
    candidates, joined = [], 0
    for i, sub in enumerate(phase2):
        text = SPEC['llmail_text_format'].format(subject=sub['subject'], body=sub['body'])
        label = labels.get(normalized(text))
        joined += label is not None
        if label and str(label['attack_attempt']) == 'True' and label['reason'] == 'api_triggered':
            candidates.append(dict(id=f'{LLMAIL}/Phase2/{i}', source=LLMAIL, source_split='Phase2', index=i,
                                   group=f"llmail-team:{sub['team_id']}", text=text, label=1,
                                   label_basis=SPEC['labels'][LLMAIL + '/Phase2']))
    if joined < 0.9 * len(phase2):
        raise ValueError(f'Only {joined}/{len(phase2)} Phase2 rows matched the labelled submissions')
    reserved, eligible = select_teams(candidates, prior_groups, screen, LLMAIL, rows, exclusions)
    reservation['groups'] += reserved
    reservation['eligible_teams'][LLMAIL] = eligible

    emails_path = hf_hub_download(LLMAIL, FP_EMAILS, repo_type='dataset', revision=SOURCES[LLMAIL])
    for i, email in enumerate(json.loads(Path(emails_path).read_text())):
        rid = f'{LLMAIL}/emails_for_fp_tests/{i}'
        fields, excluded = screen.admit(rid, email)
        if excluded:
            exclusions.append(excluded)
            continue
        rows.append(dict(id=rid, source=LLMAIL, source_split='emails_for_fp_tests', index=i,
                         group='llmail-fp-emails', text=email, label=0,
                         label_basis=SPEC['labels'][LLMAIL + '/emails_for_fp_tests'], **fields))

    counts = Counter(f"{r['source']}/{r['source_split']}:{r['label']}" for r in rows)
    if min(counts.get(k, 0) for k in (f'{BROWSESAFE}/test:0', f'{BROWSESAFE}/test:1', f'{SATML}/attack:1',
                                       f'{LLMAIL}/Phase2:1', f'{LLMAIL}/emails_for_fp_tests:0')) < 30:
        raise ValueError(f'A holdout slice has fewer than 30 rows: {dict(counts)}')
    jsonl(OUT / 'holdout.jsonl', rows)
    jsonl(OUT / 'exclusions.jsonl', exclusions)
    dump(OUT / 'reservation.json', reservation)
    shutil.copyfile(__file__, OUT / 'freeze_final_holdout.py')
    files = ('holdout.jsonl', 'exclusions.jsonl', 'reservation.json', 'freeze_final_holdout.py')
    dump(OUT / 'protocol.json', dict(
        SPEC, created=datetime.now(timezone.utc).isoformat(), prior_inputs_screened=len(prior),
        counts=dict(sorted(counts.items())), exclusion_reasons=dict(Counter(e['reason'] for e in exclusions)),
        reserved_groups=len(reservation['groups']), llmail_phase2_label_join=f'{joined}/{len(phase2)}',
        source_files={FP_EMAILS: sha(Path(emails_path)), P2_LABELS: sha(Path(labels_path))},
        files={name: sha(OUT / name) for name in files}))
    check()
    print(json.dumps(dict(sorted(counts.items())), indent=2))


def check(out=OUT):
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, expected in protocol['files'].items():
        assert sha(out / name) == expected, f'{name} changed after freezing'
    rows = load_rows(out / 'holdout.jsonl')
    assert all(r['text_hash'] == digest(normalized(r['text'])) for r in rows), 'Holdout text changed'
    assert len({r['id'] for r in rows}) == len(rows), 'Duplicate holdout IDs'
    counts = Counter(f"{r['source']}/{r['source_split']}:{r['label']}" for r in rows)
    assert dict(sorted(counts.items())) == protocol['counts'], 'Holdout counts changed'
    print(f'Verified {len(rows)} frozen holdout rows and {len(protocol["files"])} file hashes', flush=True)
    return protocol


class Guard:
    """Training-side quarantine for the frozen holdout. Fails closed when a team cannot be checked."""

    def __init__(self, out=OUT):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from transformers import AutoTokenizer
        check(out)
        rows = load_rows(out / 'holdout.jsonl')
        self.ids = {r['id'] for r in rows}
        self.hashes = {r['text_hash'] for r in rows}
        self.tokens = {r['token_hash'] for r in rows}
        self.groups = set(json.loads((out / 'reservation.json').read_text())['groups'])
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000)
        self.matrix = self.vectorizer.fit_transform(r['text'] for r in rows)

    def reason(self, rid, source, split, text, group=None):
        """Return why a candidate training row must be dropped, or None if it may be used."""
        if source == BROWSESAFE and split == 'test':
            return 'BrowseSafe test split is reserved'
        if source in (SATML, LLMAIL) and split != 'emails_for_fp_tests':
            if not group:
                raise ValueError(f'{rid}: SaTML/LLMail training rows must carry their team group')
            if group in self.groups:
                return 'reserved team'
        if rid in self.ids or digest(normalized(text)) in self.hashes:
            return 'holdout ID or text'
        if first_window(self.tokenizer, text) in self.tokens:
            return 'holdout first-window tokens'
        if float((self.vectorizer.transform([text]) @ self.matrix.T).max()) >= SPEC['cosine_threshold']:
            return 'lexical cosine >=0.90 with holdout'
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check'))
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else check()


if __name__ == '__main__':
    main()
