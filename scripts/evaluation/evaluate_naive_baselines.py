"""Last-frame and last-five-frame velocity baselines in raw and file coordinates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time

import numpy as np
from rdkit.Chem import GetPeriodicTable
import torch

from compare_generation import aggregate_rows
from semiflexible_scores import score_paths, VERSION
from projection_encoding import encode_coordinates

ROOT = Path(os.environ.get('QFIELD_RESULTS_ROOT', 'results'))


def predict(observed, dt_ps, horizon):
    """Fit velocity with an intercept; extrapolate from the actual last observation."""
    observed = np.asarray(observed, dtype=np.float64)
    assert len(observed) >= 5 and dt_ps > 0
    t = np.arange(-4, 1, dtype=float)*dt_ps
    centred = t-t.mean()
    velocity = np.einsum('t,tni->ni', centred, observed[-5:])/np.dot(centred, centred)
    last = np.broadcast_to(observed[-1], (horizon, *observed.shape[1:])).copy()
    future_time = np.arange(1, horizon+1, dtype=float)*dt_ps
    return dict(Static=last, Linear=last+future_time[:, None, None]*velocity), velocity


def motion_scores(paths, record, detail):
    """Report displacement amplitude and whole-ligand centre-vector errors."""
    table = GetPeriodicTable()
    weights = np.array([table.GetAtomicWeight(int(z)) for z in record['inputs']['ligand_graph']['atomic_numbers']])
    weights /= weights.sum()
    last = np.asarray(record['inputs']['X_obs'])[-1]
    a = np.einsum('mhni,n->mhi', np.asarray(paths)-last, weights)
    b = np.einsum('hni,n->hi', np.asarray(record['X_future'])-last, weights)
    return dict(relative_displacement_amplitude_mae_angstrom=float(np.abs(
        detail['features'][..., 0]-detail['reference_features'][None, ..., 0]).mean()),
        centre_displacement_vector_mae_angstrom=float(np.linalg.norm(a-b, axis=-1).mean()))


def verify_definition():
    initial = np.arange(6, dtype=float).reshape(2, 3)
    velocity = np.array([[.01, -.02, .03], [-.04, .05, -.06]])
    observed = initial+np.arange(10)[:, None, None]*80*velocity
    values, fitted = predict(observed, 80., 20)
    np.testing.assert_allclose(fitted, velocity, rtol=0, atol=1e-14)
    np.testing.assert_allclose(values['Linear'], initial+np.arange(10, 30)[:, None, None]*80*velocity, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(values['Static'], np.repeat(observed[-1: ], 20, axis=0))
    return dict(known_velocity_and_physical_times=True, static_last_frame=True)


def evaluate_case(task):
    file, scales, output, replicas = task
    torch.set_num_threads(2)
    record = torch.load(file, weights_only=False, map_location='cpu')
    meta, inputs = record['meta'], record['inputs']
    assert meta['dt_ps'] == 80 and inputs['dt_ps'] == meta['dt_ps']
    assert (meta['n_obs'], meta['n_pred']) == {'T1':(10,10), 'T2':(80,20), 'T3':(20,80)}[meta['tier']]
    predictions, velocity = predict(inputs['X_obs'], meta['dt_ps'], meta['n_pred'])
    t = np.arange(-4, 1, dtype=float)*meta['dt_ps']
    independent = np.linalg.lstsq(np.column_stack([np.ones(5), t]),
        np.asarray(inputs['X_obs'])[-5:].reshape(5, -1), rcond=None)[0][1].reshape(velocity.shape)
    np.testing.assert_allclose(velocity, independent, rtol=1e-10, atol=1e-12)
    rows = []
    for name, path in predictions.items():
        started = time.perf_counter()
        paths = np.repeat(path[None], replicas, axis=0)
        metrics, detail = score_paths(paths, record, scales)
        features, truth = detail['features'], detail['reference_features']
        active = np.isfinite(features).all((0,1)) & np.isfinite(truth).all(0)
        direct_es = np.sqrt(np.mean(np.square((features[0][:,active]-truth[:,active])/scales[active])))
        np.testing.assert_allclose(metrics['Probability']['feature_energy_score'], direct_es, rtol=1e-12, atol=1e-12)
        if name == 'Static': assert abs(metrics['Dyn']['predicted_mean_atom_rmsf_angstrom']) < 1e-10
        metrics = dict(metrics, Motion=motion_scores(paths, record, detail))
        encoded,encoded_record=encode_coordinates(path[None],record)
        encoded_paths=np.repeat(encoded,replicas,axis=0)
        encoded_metrics,encoded_detail=score_paths(encoded_paths,encoded_record,scales)
        encoded_metrics=dict(encoded_metrics,Motion=motion_scores(encoded_paths,encoded_record,encoded_detail))
        row = dict(method=name, id=meta['id'], tier=meta['tier'], paths=replicas, independent_paths=1,
            source_record=str(file), metrics=metrics, seconds=time.perf_counter()-started,
            encoded_metrics=encoded_metrics,
            deterministic_es_direct=float(direct_es), independent_velocity_fit_passed=True)
        destination = Path(output)/name/(meta['id']+'_'+meta['tier']+'.pt')
        torch.save(dict(meta=meta, path=path, velocity=velocity if name == 'Linear' else np.zeros_like(velocity),
            replicas=replicas, independent_paths=1, source_record=str(file), metrics=metrics,
            encoded_path=encoded[0],encoded_metrics=encoded_metrics), destination)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT/'validation16_v2')
    parser.add_argument('--statistics', type=Path, default=ROOT/'feature_statistics_v2.json')
    parser.add_argument('--paths',type=int,choices=[8,32],default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    verification = verify_definition()
    manifest = json.loads((args.cache/'manifest.json').read_text())
    members = [row for row in manifest['rows'] if row['partition'] == 'validation']
    assert manifest['completed'] and len(members)>0
    statistics = json.loads(args.statistics.read_text())
    assert statistics['partition'] == 'train' and statistics['version'] == VERSION
    scales = np.asarray(statistics['scale'])
    files = [file for row in members for file in row['records']]
    assert len(files) == 3*len(members)
    out = args.output.resolve(); out.mkdir()
    source = out/'source'; source.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path, source/path.name)
    for name in ('Static', 'Linear'): (out/name).mkdir()
    protocol = dict(systems=[row['id'] for row in members], tiers=['T1','T2','T3'], paths=args.paths,
        independent_paths=1, window=5, source_cache=str(args.cache), statistics=statistics,
        history_only_prediction=True, projection='plain deterministic baseline; raw and common XTC encoding both reported',
        uncertainty_unit='system', coordinate_frame='same cached observed protein reference',
        cpu_affinity=sorted(os.sched_getaffinity(0)), workers=2, threads_per_worker=2,
        definitions=verification, selection_rules='diagnostic comparison; training continuation follows model learning trend',
        final_coordinate_definition='complete_original_atom_order_common_MDA_writer',independent_set='sealed')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    started = time.perf_counter(); rows = []
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        for result in pool.map(evaluate_case, [(file, scales, str(out),args.paths) for file in files]):
            rows.extend(result)
            with (out/'rows.jsonl').open('a') as stream:
                for row in result: stream.write(json.dumps(row, allow_nan=False)+'\n')
            state = dict(completed_prefixes=len(rows)//2, expected_prefixes=len(files), elapsed_seconds=time.perf_counter()-started)
            (out/'status.json').write_text(json.dumps(state)); print(json.dumps(state), flush=True)
    methods = {name: aggregate_rows([row for row in rows if row['method'] == name], args.paths) for name in ('Static','Linear')}
    encoded_methods={name:aggregate_rows([dict(row,metrics=row['encoded_metrics']) for row in rows if row['method']==name],args.paths)
                     for name in ('Static','Linear')}
    report = dict(completed=True, protocol=protocol, rows=rows, methods=methods,encoded_methods=encoded_methods,
        verification=dict(definition=verification, true_cases=len(files), methods=2,
            independent_velocity_fit_passed=True, deterministic_ES_formula_passed=True),
        seconds=time.perf_counter()-started, training_performed=False, independent_set='sealed')
    (out/'review.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(completed=True, systems=len(members), prefixes=len(files), seconds=report['seconds'])), flush=True)


if __name__ == '__main__': main()
