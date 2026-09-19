"""Assemble the existing flat execution layout from functionally grouped sources."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = sorted(path for folder in ('src', 'scripts', 'tests')
                   for path in (root/folder).rglob('*.py'))
    assert len({path.name for path in files}) == len(files), 'runtime filenames must be unique'
    args.output.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.copy2(path, args.output/path.name)
    shutil.copytree(root/'configs', args.output/'configs')
    (args.output/'source_layout.json').write_text(json.dumps(
        {path.name: str(path.relative_to(root)) for path in files}, indent=2))
    print(json.dumps(dict(files=len(files), output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
