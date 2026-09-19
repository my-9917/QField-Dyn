"""Review complete generation shards and rank checkpoints by the frozen selection rule."""
import argparse
import filecmp
import json
import math
from pathlib import Path
import numpy as np
import torch
from sampling import case_seed


RANKING = {
    'feature_energy_score': 'Probability.feature_energy_score',
    'ligand_mean_rmsd_angstrom': 'Geo.mean_rmsd_angstrom',
    'coordinate_energy_score_angstrom': 'Geo.coordinate_energy_score_angstrom',
}


def flatten_metrics(metrics, prefix=''):
    result = {}
    for key, value in metrics.items():
        name = prefix+'.'+key if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(value), name
            result[name] = float(value)
        elif name == 'Dyn.correlations':
            for row in value:
                result.update(flatten_metrics({k: v for k, v in row.items() if k != 'lag_ps'},
                    name+'.'+str(row['lag_ps'])+'ps'))
    return result


def aggregate_rows(rows, paths_per_prefix=32):
    grouped = {}
    # Fixed reduction order gives identical means across worker/shard completion orders.
    for row in sorted(rows,key=lambda item:(item['id'],item['tier'])):
        group = grouped.setdefault(row['id'], {})
        assert row['tier'] not in group and row['paths'] == paths_per_prefix
        group[row['tier']] = flatten_metrics(row['metrics'])
    assert all(set(group) == {'T1', 'T2', 'T3'} for group in grouped.values())
    names = sorted({name for group in grouped.values() for metrics in group.values() for name in metrics})
    by_system = {identifier: {} for identifier in grouped}
    summary = {}
    for name in names:
        defined = 0
        for identifier, group in grouped.items():
            values = [metrics[name] for metrics in group.values() if name in metrics]
            defined += len(values)
            by_system[identifier][name] = float(np.mean(values)) if values else None
        values = [scores[name] for scores in by_system.values() if scores[name] is not None]
        summary[name] = dict(mean=float(np.mean(values)), defined_systems=len(values), defined_prefixes=defined)
    by_tier = {tier: {} for tier in ['T1', 'T2', 'T3']}
    for tier in by_tier:
        for name in names:
            values = [group[tier][name] for group in grouped.values() if name in group[tier]]
            by_tier[tier][name] = dict(mean=float(np.mean(values)) if values else None, defined_systems=len(values))
    return dict(systems=len(grouped), prefixes=len(rows), summary=summary, by_system=by_system, by_tier=by_tier,
        weighting='equal systems; equal defined T1/T2/T3 within each system; defined counts reported')


def collect(directory, manifest, shards, selection, paths_per_prefix=32):
    members = [row for row in manifest['rows'] if row['partition'] == 'validation']
    expected = {(row['id'], tier): file for row in members for tier, file in zip(['T1', 'T2', 'T3'], row['records'])}
    all_rows, seen, first = [], set(), None
    for index in range(shards):
        output = directory/f'shard_{index}'
        review = json.loads((output/'review.json').read_text())
        assert review['completed'] and review['shard'] == index and review['shards'] == shards
        assert review['paths_per_prefix'] == paths_per_prefix and review['seed'] == selection['evaluation_seed']
        assert review['expected_prefixes'] == len(review['rows']) == 3*len(members[index::shards])
        if first is None:
            first = review
        else:
            for key in ['statistics', 'seed', 'paths_per_prefix', 'solver_steps', 'evaluator_version', 'path_batch_size']:
                assert review[key] == first[key], key
            assert filecmp.cmp(review['checkpoint'], first['checkpoint'], shallow=False)
        for row in review['rows']:
            key = row['id'], row['tier']
            assert key in expected and key not in seen and row['paths'] == paths_per_prefix
            artifact = torch.load(output/(row['id']+'_'+row['tier']+'.pt'), weights_only=False, map_location='cpu')
            assert artifact['source_record'] == expected[key] and artifact['metrics'] == row['metrics']
            assert artifact['seeds'] == [case_seed(review['seed']+i, artifact['meta']) for i in range(paths_per_prefix)]
            assert artifact['paths'].shape[:2] == (paths_per_prefix, {'T1': 10, 'T2': 20, 'T3': 80}[row['tier']])
            assert np.isfinite(artifact['paths']).all()
            assert artifact['checkpoint'] == review['checkpoint']
            all_rows.append(row); seen.add(key)
    assert seen == set(expected)
    return dict(**aggregate_rows(all_rows, paths_per_prefix), checkpoint=first['checkpoint'], directory=str(directory.resolve()),
        protocol={key: first[key] for key in ['statistics', 'seed', 'solver_steps', 'evaluator_version', 'path_batch_size']},
        generation_seconds=sum(row['generation_seconds'] for row in all_rows),
        scoring_seconds=sum(row['scoring_seconds'] for row in all_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='append', required=True, help='name=directory containing shard_0, shard_1, ...')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.cache/'manifest.json').read_text()); assert manifest['completed']
    selection = json.loads((Path(__file__).parent/'configs/selection_v1.json').read_text())
    assert selection['ranking'] == list(RANKING)
    results = {}
    for candidate in args.candidate:
        name, directory = candidate.split('=', 1)
        assert name not in results
        results[name] = collect(Path(directory), manifest, args.shards, selection)
    names = list(results)
    first = results[names[0]]
    for result in results.values():
        assert result['protocol'] == first['protocol'] and result['by_system'].keys() == first['by_system'].keys()
    ranking = sorted(names, key=lambda name: tuple(results[name]['summary'][key]['mean'] for key in RANKING.values()))
    pairs = {}
    for name in names[1:]:
        scores = {}
        for key in first['summary'].keys() & results[name]['summary'].keys():
            differences = {identifier: result[key]-first['by_system'][identifier][key]
                for identifier, result in results[name]['by_system'].items()
                if result[key] is not None and first['by_system'][identifier][key] is not None}
            scores[key] = dict(mean_difference=float(np.mean(list(differences.values()))) if differences else None,
                defined_systems=len(differences), differences_by_system=differences)
        pairs[name+'_minus_'+names[0]] = scores
    report = dict(completed=True, candidates=results, ranking=ranking, ranked_candidate=ranking[0],
        paired_differences=pairs, selection=selection, architecture_and_physical_acceptance_pending=True,
        interpretation='complete development-set generation comparison; production and submission have separate acceptance criteria')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'review.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(ranking=ranking, systems=first['systems'], prefixes=first['prefixes']), indent=2))


if __name__ == '__main__':
    main()
