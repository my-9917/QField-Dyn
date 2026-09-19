"""Check inference artifacts and exact coordinates after packaging or relocation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from trajectory_delivery import read_xtc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    review = json.loads((a.prediction/'review.json').read_text())
    assert review['completed'] and review['rows']
    rows = []
    for row in review['rows']:
        name = row['id']
        artifact = torch.load(a.prediction/(name+'.pt'), map_location='cpu', weights_only=False)
        assert row['passed'] and artifact['exact_replay_passed']
        assert artifact['resolution_checks'][-1]['passed']
        actual = read_xtc(a.prediction/(name+'_pred.xtc'))
        comparison = None
        if a.reference:
            expected = read_xtc(a.reference/(name+'_pred.xtc'))
            np.testing.assert_array_equal(actual['coordinates_angstrom'], expected['coordinates_angstrom'])
            np.testing.assert_array_equal(actual['times_ps'], expected['times_ps'])
            comparison = 'elementwise_equal'
        rows.append(dict(id=name, frames=len(actual['coordinates_angstrom']),
            finite=bool(np.isfinite(actual['coordinates_angstrom']).all()),
            exact_replay_passed=True, numerical_resolution_passed=True, reference_comparison=comparison))
    result = dict(passed=True, rows=rows)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
