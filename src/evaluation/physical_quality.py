"""Geometry-tail diagnostics with train-calibrated frame thresholds.

The calibrated tail flag measures departure from training MD. It is a geometry
proxy, not a force-field energy or a proof of chemical bond breaking.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from rdkit import Chem

VERSION = 'physical_quality_train_tail_v2'
FAMILIES = ('bond_relative', 'angle_degrees', 'self_compression', 'environment_compression')


def distribution(values, total_count=None):
    x = np.asarray(values).ravel()
    x = x[np.isfinite(x)]
    positive = x[x > 1e-7]
    def summary(v):
        return dict(count=len(v), mean=float(v.mean()), p50=float(np.quantile(v, .5)),
                    p90=float(np.quantile(v, .9)), p95=float(np.quantile(v, .95)),
                    p99=float(np.quantile(v, .99)), maximum=float(v.max())) if len(v) else dict(count=0)
    all_values = summary(x)
    if total_count is not None:
        # Quantiles of a sparse nonnegative array include its implicit zeros.
        ordered = np.sort(x); zeros = total_count-len(x)
        assert zeros >= 0 and (x >= 0).all()
        def at(index):
            return 0. if index < zeros else float(ordered[index-zeros])
        all_values = dict(count=total_count, mean=float(x.sum()/total_count), maximum=float(x.max(initial=0)))
        for name, q in [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99)]:
            index = (total_count-1)*q; low = int(np.floor(index)); high = int(np.ceil(index))
            all_values[name] = at(low)+(index-low)*(at(high)-at(low))
    return dict(all=all_values, positive=summary(positive))


def measurements(paths, geometry):
    """Exact arrays for bonds/angles and sparse overlap arrays including implicit zeros."""
    x = np.asarray(paths, dtype=np.float64)
    shape = x.shape[:2]
    finite = np.isfinite(x).all((-1, -2))
    clean = x[finite]
    g = {k: np.asarray(v) for k, v in geometry.items() if k not in ('calibration_version',)}
    bonds, triples = g['bonds'], g['angles']
    distance = np.linalg.norm(clean[:, bonds[:, 0]]-clean[:, bonds[:, 1]], axis=-1)
    a, b, c = triples.T
    u, v = clean[:, a]-clean[:, b], clean[:, c]-clean[:, b]
    angle = np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u*v).sum(-1))
    values, physical, samples = {}, {}, {}
    for name, y in [('bond', distance), ('angle', angle)]:
        excess = np.maximum(g[name+'_min']-y, 0)+np.maximum(y-g[name+'_max'], 0)
        physical[name] = excess if name == 'bond' else np.rad2deg(excess)
        key = 'bond_relative' if name == 'bond' else 'angle_degrees'
        samples[key] = excess/g['bond_scale'] if name == 'bond' else physical[name]
        values[name] = y if name == 'bond' else np.rad2deg(y)
    heavy, pairs = g['heavy'], g['self_pairs']
    d = np.linalg.norm(clean[:, pairs[:, 0]]-clean[:, pairs[:, 1]], axis=-1)
    samples['self_compression'] = np.maximum(g['self_limits']-d, 0)/g['self_limits']
    physical['self_depth_angstrom'] = np.maximum(g['self_limits']-d, 0)
    env, limits = g['environment'], g['cross_limits']
    tree = cKDTree(env)
    env_values, env_depth, events = [], [], []
    maxima, means, counts = [], [], []
    for frame_index, frame in enumerate(clean):
        neighbours = tree.query_ball_point(frame[heavy], r=float(limits.max()))
        ia = np.repeat(np.arange(len(heavy)), [len(row) for row in neighbours])
        ib = np.asarray([j for row in neighbours for j in row], dtype=int)
        dist = np.linalg.norm(frame[heavy[ia]]-env[ib], axis=-1)
        depth = np.maximum(limits[ia, ib]-dist, 0)
        overlap = depth/limits[ia, ib]
        positive = overlap > 0
        env_values.append(overlap[positive]); env_depth.append(depth[positive])
        maxima.append(float(overlap.max(initial=0)))
        means.append(float(np.square(overlap).sum()/len(heavy)))
        counts.append(int((overlap > 1e-7).sum()))
        if np.any(positive):
            j = int(overlap.argmax())
            member, time_index = np.argwhere(finite)[frame_index]
            events.append(dict(member=int(member), future_frame_index=int(time_index), ligand_atom=int(heavy[ia[j]]),
                               environment_heavy_atom=int(ib[j]), distance_angstrom=float(dist[j]),
                               limit_angstrom=float(limits[ia[j], ib[j]]), compression=float(overlap[j])))
    samples['environment_compression'] = np.concatenate(env_values) if env_values else np.empty(0)
    physical['environment_depth_angstrom'] = np.concatenate(env_depth) if env_depth else np.empty(0)
    frame_max, frame_mse = {}, {}
    for key in FAMILIES:
        maximum = np.full(shape, np.inf); mean = np.full(shape, np.inf)
        if key == 'environment_compression':
            maximum[finite], mean[finite] = maxima, means
        else:
            s = samples[key]
            maximum[finite] = s.max(-1, initial=0)
            mean[finite] = np.square(s).sum(-1)/max(1, s.shape[-1])
        frame_max[key], frame_mse[key] = maximum, mean
    return dict(finite=finite, samples=samples, physical=physical, values=values,
                frame_max=frame_max, frame_mse=frame_mse, environment_events=events,
                environment_pair_count=int(len(clean)*limits.size),
                environment_violations=counts, heavy_atoms=len(heavy))


def chemical_checks(paths, record, chemistry):
    """Check supplied stereocentres; keep anonymous tetrahedral proxies separate."""
    if chemistry is None:
        return dict(available=False, stereochemistry='unassigned', checked_centres=0), None
    graph = record['inputs']['ligand_graph']
    np.testing.assert_array_equal(chemistry['numbers'], graph['atomic_numbers'])
    np.testing.assert_array_equal(chemistry['names'], graph['atom_names'])
    assert set(map(tuple, np.sort(chemistry['bonds'], axis=1))) == set(map(tuple, np.sort(graph['bonds'], axis=1)))
    neighbours = [[] for _ in chemistry['numbers']]
    for a, b in chemistry['bonds']:
        neighbours[a].append(b); neighbours[b].append(a)
    mol = Chem.RWMol()
    for z, charge in zip(chemistry['numbers'], chemistry['formal_charges']):
        atom = Chem.Atom(int(z)); atom.SetFormalCharge(int(charge)); atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    orders = {1.: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC, 2.: Chem.BondType.DOUBLE, 3.: Chem.BondType.TRIPLE}
    for (i, j), order in zip(chemistry['bonds'], chemistry['bond_orders']):
        mol.AddBond(int(i), int(j), orders[float(order)])
    mol = mol.GetMol(); Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(len(chemistry['numbers']))
    for i, xyz in enumerate(np.asarray(record['inputs']['X_obs'])[-1]): conformer.SetAtomPosition(i, xyz)
    mol.AddConformer(conformer); Chem.AssignStereochemistryFrom3D(mol)
    centres = [i for i, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)]
    tetra = np.asarray([(i, *sorted(neighbours[i])[:3]) for i in centres], dtype=int).reshape(-1, 4)
    # Three-coordinate pyramidal centres (e.g. stereogenic sulfoxide S) use
    # the same oriented volume; an explicit fourth bonded atom is unnecessary.
    assert all(len(neighbours[i]) in (3, 4) for i in centres)
    def volumes(x):
        c, a, b, d = tetra.T
        return np.einsum('...ni,...ni->...n', x[..., a, :]-x[..., c, :],
                        np.cross(x[..., b, :]-x[..., c, :], x[..., d, :]-x[..., c, :]))
    observed = volumes(np.asarray(record['inputs']['X_obs']))
    stable = observed.min(0)*observed.max(0) > 0
    signed = volumes(np.asarray(paths))*np.sign(observed[-1])
    # Zero volume is a degenerate configuration, not a valid stereocentre.
    bad = (signed[..., stable] <= 0).any(-1)
    return dict(available=True, checked_centres=int(stable.sum()),
                three_coordinate_centres=sum(len(neighbours[i]) == 3 for i in centres),
                observed_unstable_centres=int((~stable).sum()),
                stereochemistry_source=chemistry.get('provenance', chemistry['version']),
                inversion_or_degeneracy_frame_fraction=float(bad.mean()),
                ring_count=len(chemistry['ring_atoms']),
                restricted_connections=len(chemistry['restricted_links']),
                restricted_torsion_qualification='reported by existing Dyn; absent from this tail gate'), bad


def score_quality(paths, record, calibration, chemistry=None, reference=None):
    assert calibration['version'] == VERSION and calibration['partition'] == 'train'
    m = measurements(paths, record['geometry'])
    finite = m['finite']; failure = ~finite; internal_failure = ~finite
    result = dict(version=VERSION, interpretation='train-calibrated geometry-tail proxy',
                  heavy_atoms=m['heavy_atoms'], finite_frame_fraction=float(finite.mean()), families={})
    for key in FAMILIES:
        threshold = calibration['thresholds'][key]
        flagged = m['frame_max'][key] > threshold+1e-7
        failure = failure | flagged
        if key != 'environment_compression': internal_failure = internal_failure | flagged
        s = m['samples'][key]
        item = dict(threshold=threshold, anomaly_frame_fraction=float(flagged.mean()),
                    finite_mean_squared_severity=float(m['frame_mse'][key][finite].mean()) if finite.any() else None,
                    frame_maximum_distribution=distribution(m['frame_max'][key]),
                    constraint_distribution=distribution(s))
        if key == 'environment_compression':
            # Sparse KDTree stores positive entries. Count implicit zeros explicitly.
            item['constraint_distribution'] = distribution(s, total_count=m['environment_pair_count']) if m['environment_pair_count'] else distribution(s)
            item['constraint_anomaly_fraction'] = float((s > threshold+1e-7).sum()/max(1, m['environment_pair_count']))
        else:
            item['constraint_anomaly_fraction'] = float((s > threshold+1e-7).mean()) if s.size else 0.
        result['families'][key] = item
    chemical, flips = chemical_checks(paths, record, chemistry)
    if chemistry is not None:
        numbers = chemistry['numbers']
        types = {':'.join(map(str, (*sorted((numbers[i], numbers[j])), float(order))))
                 for (i, j), order in zip(chemistry['bonds'], chemistry['bond_orders'])}
        chemical['bond_types_outside_calibration'] = sorted(types-set(calibration['chemical_types']))
        chemical['stereocentre_identification'] = 'RDKit chemical symmetry and observed 3D coordinates'
    result['chemistry'] = chemical
    if flips is not None:
        failure |= flips
        internal_failure |= flips
    result['assessed_tail_valid_frame_fraction'] = float((~failure).mean())
    result['intramolecular_tail_valid_frame_fraction'] = float((~internal_failure).mean())
    result['fixed_environment_valid_frame_fraction'] = float((finite & (m['frame_max']['environment_compression'] <= calibration['thresholds']['environment_compression']+1e-7)).mean())
    result['qualification_scope'] = 'geometric tails plus available assigned stereocentres; force-field energy and bond reaction excluded'
    result['physical_excess_distributions'] = {k: distribution(v) for k, v in m['physical'].items()}
    if m['environment_pair_count']:
        result['physical_excess_distributions']['environment_depth_angstrom'] = distribution(
            m['physical']['environment_depth_angstrom'], total_count=m['environment_pair_count'])
    result['worst_environment_events'] = sorted(m['environment_events'], key=lambda r: r['compression'], reverse=True)[:5]
    result['windows'] = {name: dict(valid_frame_fraction=float((~failure[:, idx]).mean()))
                         for name, idx in zip(('early', 'middle', 'late'), np.array_split(np.arange(failure.shape[1]), 3))}
    if reference is not None:
        ref = measurements(np.asarray(reference)[None], record['geometry'])
        result['bonded_distribution_comparison'] = {}
        for key in ('bond', 'angle'):
            p, q = m['values'][key], ref['values'][key]
            distances = [wasserstein_distance(p[:, i], q[:, i]) for i in range(p.shape[-1])] if len(p) else []
            result['bonded_distribution_comparison'][key] = dict(
                mean_wasserstein=float(np.mean(distances)) if distances else None,
                predicted_constraint_variance_mean=float(np.var(p, axis=0).mean()) if p.size else None,
                predicted_within_path_variance_mean=float(np.var(p.reshape(*finite.shape, -1), axis=1).mean()) if finite.all() and p.size else None,
                reference_constraint_variance_mean=float(np.var(q, axis=0).mean()) if q.size else None,
                units='Angstrom' if key == 'bond' else 'degrees', pooling='paths and frames; time dynamics reported separately')
        result['extra_mean_squared_severity'] = {k: float(m['frame_mse'][k][finite].mean()-ref['frame_mse'][k].mean())
                                                  if finite.any() else None for k in FAMILIES}
    return result
