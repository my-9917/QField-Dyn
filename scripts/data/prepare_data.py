"""Build all three native prefixes directly from qualified coordinates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import time
import h5py
import torch
from native_coordinates import read_training_topology
from semiflexible_inputs import training_record, VERSION
from ligand_geometry import build_geometry

TIERS = {'T1': (10, 10), 'T2': (80, 20), 'T3': (20, 80)}


def prepare_system(job):
    row, config, output = job
    torch.set_num_threads(1)
    start = time.perf_counter()
    topology = read_training_topology(row['topology'])
    calibration = json.loads(Path(config['calibration']).read_text())
    records = []
    with h5py.File(config['md'], 'r') as file:
        for tier, (k, h) in TIERS.items():
            spec = dict(tier=tier, n_obs=k, n_pred=h, dt_ps=80., partition=row['partition'])
            record = training_record(file[row['id']], topology, spec, row['id'])
            record['geometry'] = build_geometry(record['inputs'], calibration)
            assert torch.isfinite(record['X_future']).all()
            assert all(torch.isfinite(v).all() for v in record['encoder_inputs'].values())
            destination = Path(output)/row['partition']/f"{row['id']}_{tier}.pt"
            torch.save(record, destination)
            records.append(str(destination))
    return dict(id=row['id'], partition=row['partition'], records=records, seconds=time.perf_counter()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    eligibility = json.loads(Path(config['eligibility']).read_text())
    assert eligibility['passed'] and len(eligibility['failures']) == 0
    rows = eligibility['eligible']
    train = {row['id'] for row in rows if row['partition'] == 'train'}
    val = {row['id'] for row in rows if row['partition'] == 'validation'}
    assert train.isdisjoint(val)
    members = []
    for partition in ('train', 'validation'):
        candidates = sorted((row for row in rows if row['partition'] == partition), key=lambda row: row['id'])
        random.Random(config['membership_seed']).shuffle(candidates)
        members.extend(candidates[:config['systems'][partition]])
    args.output.mkdir(parents=True, exist_ok=False)
    for partition in ('train', 'validation'):
        (args.output/partition).mkdir()
    frozen = dict(config, input_version=VERSION, selected_members=[dict(id=r['id'], partition=r['partition']) for r in members])
    (args.output/'config.json').write_text(json.dumps(frozen, indent=2))
    rows = []
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for row in pool.map(prepare_system, [(row, config, str(args.output.resolve())) for row in members]):
            rows.append(row)
            with (args.output/'rows.jsonl').open('a') as file:
                file.write(json.dumps(row)+'\n')
            print(json.dumps({'completed': len(rows), 'total': len(members), **row}), flush=True)
    assert len(rows) == len(members) and all(len(row['records']) == 3 for row in rows)
    (args.output/'manifest.json').write_text(json.dumps(dict(completed=True, input_version=VERSION, config=frozen, rows=rows), indent=2))


if __name__ == '__main__':
    main()
