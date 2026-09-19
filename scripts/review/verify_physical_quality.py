"""Independent numerical and corruption checks for the geometry-tail evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from physical_quality import measurements, score_quality, VERSION, FAMILIES, chemical_checks, distribution
from projection_encoding import encode_coordinates


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True); a = p.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((a.root/'inputs/manifest.json').read_text())
    record = torch.load(manifest['train'][0]['records'][0], map_location='cpu', weights_only=False)
    g = record['geometry']; observed = np.asarray(record['inputs']['X_obs']); x = np.asarray(record['X_future'])[None]
    m = measurements(x, g)
    triples = np.asarray(g['angles']); u = x[..., triples[:, 0], :]-x[..., triples[:, 1], :]
    v = x[..., triples[:, 2], :]-x[..., triples[:, 1], :]
    direct = np.rad2deg(np.arccos(np.clip((u*v).sum(-1)/np.linalg.norm(u, axis=-1)/np.linalg.norm(v, axis=-1), -1, 1)))
    np.testing.assert_allclose(m['values']['angle'], direct.reshape(-1, len(triples)), atol=1e-10, rtol=0)
    # Dense independent environment distances cross-check KDTree maxima.
    d = np.linalg.norm(x[..., np.asarray(g['heavy']), None, :]-np.asarray(g['environment']), axis=-1)
    compressed = np.maximum(np.asarray(g['cross_limits'])-d, 0)/np.asarray(g['cross_limits'])
    np.testing.assert_allclose(m['frame_max']['environment_compression'], compressed.max((-1, -2)), atol=1e-12)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([7., -3., 2.])
    changed = dict(g, environment=np.asarray(g['environment'])@rotation+shift)
    transformed = measurements(x@rotation+shift, changed)
    for k in FAMILIES: np.testing.assert_allclose(m['frame_max'][k], transformed['frame_max'][k], atol=1e-11, rtol=0)
    permutation = np.arange(x.shape[-2])[::-1]; inverse = np.argsort(permutation)
    changed = dict(g)
    for k in ('bonds', 'angles', 'self_pairs', 'heavy'): changed[k] = inverse[np.asarray(g[k])]
    reordered = measurements(x[..., permutation, :], changed)
    for k in FAMILIES: np.testing.assert_allclose(m['frame_max'][k], reordered['frame_max'][k], atol=1e-12, rtol=0)
    calibration = json.loads(a.calibration.read_text()); assert calibration['completed']
    sparse = np.array([.1, .3, .4]); dense = np.r_[np.zeros(17), sparse]
    actual = distribution(sparse, total_count=20)['all']
    for key, q in [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99)]:
        np.testing.assert_allclose(actual[key], np.quantile(dense, q), atol=1e-14)
    clean = observed[-1:][None].copy(); corrupt = clean.copy()
    atom = int(np.asarray(g['bonds'])[0, 0]); corrupt[0, 0, atom] += np.array([10., 0., 0.])
    assert measurements(corrupt, g)['frame_max']['bond_relative'].item() > 1.
    assert measurements(corrupt, g)['frame_max']['bond_relative'].item() > calibration['thresholds']['bond_relative']
    a0, b0, c0 = np.asarray(g['angles'])[0]
    corrupt = clean.copy(); corrupt[0, 0, c0] = clean[0, 0, b0]+2*(clean[0, 0, a0]-clean[0, 0, b0])
    assert measurements(corrupt, g)['frame_max']['angle_degrees'].item() > 20.
    assert measurements(corrupt, g)['frame_max']['angle_degrees'].item() > calibration['thresholds']['angle_degrees']
    a0, b0 = np.asarray(g['self_pairs'])[0]
    corrupt = clean.copy(); corrupt[0, 0, b0] = corrupt[0, 0, a0]
    assert measurements(corrupt, g)['frame_max']['self_compression'].item() == 1.
    assert measurements(corrupt, g)['frame_max']['self_compression'].item() > calibration['thresholds']['self_compression']
    corrupt = clean.copy(); heavy_atom = int(np.asarray(g['heavy'])[0])
    corrupt[0, 0, heavy_atom] = np.asarray(g['environment'])[0]
    assert measurements(corrupt, g)['frame_max']['environment_compression'].item() > calibration['thresholds']['environment_compression']
    corrupt = np.repeat(x, 2, axis=0); corrupt[0, 0, 0, 0] = np.nan
    result = score_quality(corrupt, record, calibration)
    assert result['finite_frame_fraction'] == 1-1/(2*x.shape[1])
    assert result['assessed_tail_valid_frame_fraction'] <= result['finite_frame_fraction']
    decoded, encoded_record = encode_coordinates(x, record)
    encoded = measurements(decoded, encoded_record['geometry'])
    assert encoded['finite'].all()
    chemistry = json.loads((a.root/'chemistry_combined_r3'/(record['meta']['id']+'.json')).read_text())
    chiral, _ = chemical_checks(x, record, chemistry)
    chemistry_coverage = []
    for member in manifest['train']+manifest['development']:
        filename = a.root/'chemistry_combined_r3'/(member['id']+'.json')
        if filename.exists():
            r = torch.load(member['records'][0], map_location='cpu', weights_only=False)
            info, _ = chemical_checks(np.asarray(r['X_future'])[None], r, json.loads(filename.read_text()))
            chemistry_coverage.append(dict(id=member['id'], **info))
    report = dict(passed=True, checks=['independent angle formula', 'dense environment distance', 'rigid transform',
                  'atom permutation', '10 Angstrom displacement', 'angle collapse', 'nonbonded atom coincidence',
                  'nonfinite retained in denominator', 'actual XTC roundtrip', 'chemical atom identity',
                  'all available training and development chemical centres', 'sparse quantiles include zeros',
                  'frozen train thresholds detect corruptions', 'environment atom coincidence'],
                  source_record=manifest['train'][0]['records'][0], metadata=record['meta'],
                  record_fields=list(record), manifest_fields=list(manifest), chemical_check=chiral,
                  chemistry_coverage=chemistry_coverage,
                  coordinate_encoding_max_error_angstrom=float(np.max(np.abs(decoded-x))))
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
