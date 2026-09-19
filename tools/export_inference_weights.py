"""Export inference tensors/specs while retaining the original weight provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def public_metadata(value):
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(public_metadata(item) for item in value)
    if isinstance(value, str) and value.startswith(('/home/', '/mnt/')):
        return Path(value).name
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base = torch.load(a.base, map_location='cpu', weights_only=False)
    adapter = torch.load(a.adapter, map_location='cpu', weights_only=False)
    original = dict(base_sha256=digest(a.base), adapter_sha256=digest(a.adapter))
    assert adapter['config']['base_checkpoint_sha256'] == original['base_sha256']
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save(dict(spec=public_metadata(base['spec']), model=base['model'],
                    provenance=original), a.output/'E3.pt')
    config = public_metadata(adapter['config'])
    config['base_checkpoint_sha256'] = digest(a.output/'E3.pt')
    torch.save(dict(spec=public_metadata(adapter['spec']), model=adapter['model'],
                    config=config, provenance=original), a.output/'epoch_02.pt')
    for name, state in [('E3.pt', base), ('epoch_02.pt', adapter)]:
        exported = torch.load(a.output/name, map_location='cpu', weights_only=False)
        assert state['model'].keys() == exported['model'].keys()
        assert all(torch.equal(state['model'][key], exported['model'][key]) for key in state['model'])
    report = dict(original=original, exported=dict(base_sha256=digest(a.output/'E3.pt'),
        adapter_sha256=digest(a.output/'epoch_02.pt')), tensors_bitwise_equal=True,
        format='inference tensors and constructor specs; training snapshots retained separately')
    (a.output/'provenance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
