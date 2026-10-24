"""Offline check: python3 tests/test_paper_metrics.py."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.generate_per_dataset_metrics_table import SOURCE, metrics, proportion, render


def main():
    data = json.loads(SOURCE.read_text())
    assert proportion(0, 0) == 'N/A'
    assert proportion(0, 247) == r'\shortstack{0.0\% \\ {[0.0, 1.5]}}'
    assert proportion(247, 247) == r'\shortstack{100.0\% \\ {[98.5, 100.0]}}'
    attack = {'samples': 10, 'confusion_matrix': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 10},
              'accuracy': 0, 'recall': 0, 'fpr': 0}
    result = metrics(attack)
    assert result[-1] == 'N/A' and result[-2].startswith(r'\shortstack{0.0\%')
    pooled = metrics(data['datasets']['Overall'])
    assert pooled[0] == '4793' and pooled[1].startswith(r'\shortstack{97.5\%')
    assert 'not held-out performance' in render(data)
    assert 'Local synthetic triggers' in render(data)
    for field, value in [('samples', 11), ('accuracy', 1.0), ('recall', float('nan'))]:
        bad = copy.deepcopy(attack)
        bad[field] = value
        try:
            metrics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid {field}')
    for count in [-1, 1.5, True]:
        bad = copy.deepcopy(attack)
        bad['confusion_matrix']['tp'] = count
        try:
            metrics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid count {count}')
    print('PASS: Wilson bounds, absent classes, zero recall, counts, and invalid evidence')


if __name__ == '__main__':
    main()
