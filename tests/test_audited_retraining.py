"""Offline regression check for split isolation and validation-only decisions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.retrain_audited import freeze, make_pairs, choose_threshold, metrics, assert_isolated


def main():
    rows = []
    for split in ('train', 'validation', 'test'):
        for kind in ('injection', 'safe', 'trigger'):
            for i in range(3):
                key = f'{split}:{kind}:{i}'
                rows.append(dict(id=key, group=key, text_hash=key, token_hash=key,
                                 kind=kind, label=int(kind == 'injection'), fixed_split=split))
    # Different full texts that truncate to the same input must never span splits.
    train = next(r for r in rows if r['id'] == 'train:injection:0')
    external = dict(train, id='external', text_hash='different suffix', fixed_split='test')
    rows.append(external)
    kept, excluded = freeze(rows)
    assert len(excluded) == 1
    assert next(r for r in kept if r['id'] == 'external')['split'] == 'test'
    assert not any(r['id'] == train['id'] for r in kept)
    # Team/template families isolate distinct texts, too.
    extra = dict(rows[1], id='same-team', text_hash='newtext', token_hash='newtokens', fixed_split='test')
    kept, _ = freeze(rows + [extra])
    assert next(r for r in kept if r['id'] == rows[1]['id'])['split'] == 'test'
    assert_isolated(kept)
    pairs = make_pairs([r for r in freeze(rows)[0] if r['split'] == 'train'], 'double')
    assert len(pairs) == 14000
    assert sum(p['kind'] == 'injection/trigger' for p in pairs) == 4000
    assert all(not p['left'].startswith('test:') for p in pairs)
    assert choose_threshold([0, 0, 1, 1], [.1, .3, .4, .9]) == .4
    safe_only = metrics([dict(label=0, prediction=0)])
    assert safe_only['recall'] is None and safe_only['fpr']['value'] == 0
    conflicting = dict(rows[0], id='conflict', label=0)
    conflict_kept, conflict_excluded = freeze(rows + [conflicting])
    assert not any(r['token_hash'] == rows[0]['token_hash'] for r in conflict_kept)
    assert any('conflicting' in r['reason'] for r in conflict_excluded)
    print('Audit split, pairs, threshold, and missing-class checks passed.')


if __name__ == '__main__':
    main()
