"""Bind the published validation members to an authorized local native cache."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--release-manifest', type=Path, default=Path('results/truth90/manifest.json'))
    p.add_argument('--models', type=Path, default=Path('models'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cache = json.loads((a.cache/'manifest.json').read_text())
    assert cache['completed']
    members = {row['id']: row for row in cache['rows']}
    spec = json.loads(a.release_manifest.read_text())
    for task in spec['tasks']:
        row = members[task['id']]
        assert row['partition'] == 'validation'
        matches = [Path(f) for f in row['records'] if Path(f).stem == task['key']]
        assert len(matches) == 1 and matches[0].is_file(), task['key']
        task['record'] = str(matches[0].resolve())
    spec['original_model'] = dict(spec['model'])
    spec['model'].update(
        base_sha256=hashlib.sha256((a.models/'E3.pt').read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256((a.models/'epoch_02.pt').read_bytes()).hexdigest(),
        status='goai-finals-2026')
    spec['completed'] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(spec, indent=2))
    print(json.dumps(dict(cases=len(spec['tasks']), output=str(a.output))))


if __name__ == '__main__':
    main()
