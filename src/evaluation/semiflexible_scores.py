"""Fixed-protein adaptation of the frozen ligand probability and trajectory metrics."""
import numpy as np
from scipy.spatial import cKDTree
import torch
import joint_observables as frozen
from molecular_dynamics_scores import torsion_phase, defined_mean
from path_scores import energy_score, displacement_variogram
from trajectory_metrics import trajectory_metrics
from ligand_geometry import angles, signed_volume

FEATURES = ('ligand_relative_displacement', 'ligand_gyration', 'pocket_contact_strength')
VERSION = 'semiflexible_ligand_scores_v1'


def observables(paths, record):
    inputs = record['inputs']
    graph, protein = inputs['ligand_graph'], inputs['protein_topology']
    lh, ph = np.asarray(graph['atomic_numbers']) > 1, np.asarray(protein['atomic_numbers']) > 1
    environment = np.asarray(inputs['P0'])[ph]
    residues, slots = np.unique(np.asarray(protein['residue_indices'])[ph], return_inverse=True)
    last = np.asarray(inputs['X_obs'])[-1, lh]
    initial = cKDTree(last).query(environment)[0]
    pocket_slots = np.asarray(record['pocket_slots']) if 'pocket_slots' in record else np.unique(slots[initial <= 6.])
    distances = np.empty((*paths.shape[:2], len(residues)))
    for m, h in np.ndindex(paths.shape[:2]):
        values = cKDTree(paths[m, h, lh]).query(environment)[0]
        distances[m, h] = np.inf
        np.minimum.at(distances[m, h], slots, values)
    ligand = paths[:, :, lh]
    displacement = np.sqrt(np.square(ligand-last).sum(-1).mean(-1))
    gyration = np.sqrt(np.square(ligand-ligand.mean(-2, keepdims=True)).sum(-1).mean(-1))
    contact = ((1/(1+(distances[..., pocket_slots]/4.5)**6)).mean(-1)
               if len(pocket_slots) else np.full(paths.shape[:2], np.nan))
    return dict(features=np.stack([displacement, gyration, contact], axis=-1),
                distances=distances, residues=residues, pocket_slots=pocket_slots)


def probability_scores(samples, truth, scales):
    if len(samples) == 1:
        available = np.isfinite(samples).all((0, 1)) & np.isfinite(truth).all(0)
        residual = samples[0]-truth
        return dict(feature_energy_score=float(np.sqrt(np.square(residual[..., available]/scales[available]).mean())),
                    defined_features=[name for name, active in zip(FEATURES, available) if active],
                    features={name: dict(crps=float(np.abs(residual[..., i]).mean())) if available[i] else None
                              for i, name in enumerate(FEATURES)},
                    interpretation='single-realization deterministic-form score')
    # Preserve the frozen fair-ensemble formulas. Selecting the three ligand features
    # removes constant protein channels from both numerator and dimension normalization.
    padded = np.zeros((*samples.shape[:-1], len(frozen.FEATURES)))
    reference = np.zeros((*truth.shape[:-1], len(frozen.FEATURES)))
    available = np.isfinite(samples).all((0, 1)) & np.isfinite(truth).all(0)
    selected = np.array([0, 2, 4])[available]
    padded[..., selected], reference[..., selected] = samples[..., available], truth[..., available]
    expanded_scales = np.ones(len(frozen.FEATURES)); expanded_scales[selected] = scales[available]
    scores = frozen.ensemble_scores(padded, reference, expanded_scales)
    return dict(feature_energy_score=scores['feature_energy_score']*np.sqrt(len(frozen.FEATURES)/int(available.sum())),
                defined_features=[name for name, active in zip(FEATURES, available) if active],
                features={name: scores['features'][name] if active else None for name, active in zip(FEATURES, available)})


def physics_scores(paths, geometry):
    x = torch.as_tensor(paths, dtype=torch.float64)
    g = geometry
    frames = {}
    for kind, value in [('bond', torch.linalg.vector_norm(x[..., g['bonds'][:, 0], :]-x[..., g['bonds'][:, 1], :], dim=-1)),
                        ('angle', angles(x, g['angles']))]:
        excess = ((g[kind+'_min']-value).clamp_min(0)+(value-g[kind+'_max']).clamp_min(0))/g[kind+'_scale']
        excess = excess.numpy()
        frames[kind+'_violations'] = (excess > 1e-7).sum(-1)
        frames[kind+'_mean_squared_excess'] = np.square(excess).mean(-1) if excess.shape[-1] else np.zeros(paths.shape[:2])
    volume = signed_volume(x, g['tetrahedra']).numpy()*np.asarray(g['tetrahedron_sign'])
    frames['local_configuration_flips'] = (volume < -1e-7).sum(-1)
    environment, heavy, limits = np.asarray(g['environment']), np.asarray(g['heavy']), np.asarray(g['cross_limits'])
    tree = cKDTree(environment)
    pairs, self_limits = np.asarray(g['self_pairs']), np.asarray(g['self_limits'])
    self_count, cross_count, severity = [], [], []
    for frame in np.asarray(paths).reshape(-1, paths.shape[-2], 3):
        compression = np.maximum(self_limits-np.linalg.norm(frame[pairs[:, 0]]-frame[pairs[:, 1]], axis=-1), 0)/self_limits
        self_count.append(int((compression > 1e-7).sum()))
        total = np.square(compression).sum()
        neighbours = tree.query_ball_point(frame[heavy], r=float(limits.max()))
        a = np.repeat(np.arange(len(heavy)), [len(row) for row in neighbours])
        b = np.array([j for row in neighbours for j in row], dtype=int)
        threshold = limits[a, b]
        compression = np.maximum(threshold-np.linalg.norm(frame[heavy[a]]-environment[b], axis=-1), 0)/threshold
        cross_count.append(int((compression > 1e-7).sum()))
        severity.append(float((total+np.square(compression).sum())/len(heavy)))
    frames['ligand_self_overlap_violations'] = np.asarray(self_count).reshape(paths.shape[:2])
    frames['ligand_environment_overlap_violations'] = np.asarray(cross_count).reshape(paths.shape[:2])
    frames['severe_overlap_mean_squared'] = np.asarray(severity).reshape(paths.shape[:2])
    frames['assessed_geometry_valid'] = sum(v for k, v in frames.items() if k.endswith('_violations') or k.endswith('_flips')) == 0
    return {key: float(value.mean()) for key, value in frames.items()}, frames


def score_paths(paths, record, feature_scales):
    paths = np.asarray(paths, dtype=float)
    truth = np.asarray(record['X_future'], dtype=float)
    assert paths.ndim == 4 and len(paths) >= 1 and paths.shape[1:] == truth.shape
    assert np.isfinite(paths).all()
    heavy = np.asarray(record['inputs']['ligand_graph']['atomic_numbers']) > 1
    x, y = paths[:, :, heavy], truth[:, heavy]
    last = np.asarray(record['inputs']['X_obs'])[-1, heavy]
    dt = record['meta']['dt_ps']
    error = np.sqrt(np.square(x-y).sum(-1).mean(-1))
    dyn = [trajectory_metrics(path, y, last, dt) for path in x]
    mask = torch.ones((1, x.shape[-2]), dtype=torch.bool)
    tx, ty = torch.from_numpy(x)[None], torch.from_numpy(y)[None]
    geo = dict(mean_rmsd_angstrom=float(error.mean()), final_rmsd_angstrom=float(error[:, -1].mean()),
               auc_angstrom_ps=float(np.trapz(error, dx=dt, axis=1).mean()),
               coordinate_energy_score_angstrom=(float(np.sqrt(np.square(x-y).sum(-1).mean()))
                   if len(paths) == 1 else float(energy_score(tx, ty, mask)[0])))
    dynamics = {key: float(np.mean([row[key] for row in dyn])) for key, value in dyn[0].items() if isinstance(value, float)}
    dynamics['displacement_variogram'] = float(displacement_variogram(tx, ty, mask, tuple(i for i in (1, 2, 4, 8) if i < len(y)))[0])
    dynamics['correlations'] = [{key: defined_mean([row['correlations'][i][key] for row in dyn])
                                for key in ['lag_ps', 'predicted', 'truth', 'absolute_error']}
                               for i in range(len(dyn[0]['correlations']))]
    pred, actual = observables(paths, record), observables(truth[None], record)
    indices = frozen.graph_torsions(record['inputs']['ligand_graph'])
    q, v = torsion_phase(paths, indices); qt, vt = torsion_phase(truth, indices)
    dynamics['contacts'] = frozen.contact_scores(pred, actual, record['inputs'])
    dynamics['ligand_torsions'] = frozen.periodic_scores(q, v, qt, vt, dt)
    physical, pf = physics_scores(paths, record['geometry'])
    reference, rf = physics_scores(truth[None], record['geometry'])
    stability = {row['window']: {key: float(np.mean([r['windows'][i][key] for r in dyn]))
                 for key in row if key != 'window'} for i, row in enumerate(dyn[0]['windows'])}
    for name, index in zip(['early', 'middle', 'late'], np.array_split(np.arange(len(y)), 3)):
        stability[name]['geometry'] = {key: float(value[:, index].mean()) for key, value in pf.items()}
        stability[name]['reference_geometry'] = {key: float(value[:, index].mean()) for key, value in rf.items()}
    metrics = dict(version=VERSION, Geo=geo, Dyn=dynamics, Phys=physical, Reference_Phys=reference,
                   Stab=stability, Probability=probability_scores(pred['features'], actual['features'][0], feature_scales))
    detail = dict(rmsd_by_frame=error, dynamics_by_path=dyn, physics_by_frame=pf, reference_physics_by_frame=rf,
                  features=pred['features'], reference_features=actual['features'][0], feature_scales=feature_scales,
                  contact_distances=pred['distances'], reference_contact_distances=actual['distances'][0])
    return metrics, detail
