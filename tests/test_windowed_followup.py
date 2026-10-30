"""Runnable checks for full coverage and document-level score aggregation."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from windowed_followup import ranges, document_scores, extract_html


def main():
    for length in (1, 381, 382, 383, 668, 669, 5000):
        spans = ranges(length)
        assert spans[0][0] == 0 and spans[-1][1] == length
        assert all(0 < end-start <= 382 for start,end in spans)
        assert set().union(*(set(range(a,b)) for a,b in spans)) == set(range(length))
        assert all(a < b[1] and b[0] < a for (a,_),b in zip(spans[1:],spans))
    try:
        ranges(0)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty inputs must fail')

    class Classifier:
        def predict_proba(self, vectors):
            return np.column_stack((1-vectors[:, 0], vectors[:, 0]))

    windows = [dict(row_index=0,start=0), dict(row_index=0,start=286), dict(row_index=1,start=0)]
    scores = document_scores(Classifier(), np.array([[.2],[.9],[.4]],dtype=np.float32), windows, 2)
    np.testing.assert_allclose(scores['base'], [.2,.4])
    np.testing.assert_allclose(scores['windowed'], [.9,.4])
    # Explicitly retain the known extraction limitation in this one-factor experiment.
    assert extract_html('<div>visible</div><!-- hidden instruction --><script>code</script>') == 'visible'
    print('Window coverage, aggregation, short-input invariance and unchanged extraction passed.')


if __name__ == '__main__':
    main()
