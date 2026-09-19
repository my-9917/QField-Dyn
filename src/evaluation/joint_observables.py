"""Observed-reference molecular features and ensemble scores in physical units."""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from molecular_dynamics_scores import torsion_phase, retention, defined_mean


FEATURES = ('ligand_relative_displacement', 'pocket_relative_displacement',
            'ligand_gyration', 'pocket_gyration', 'pocket_contact_strength',
            'chi_sin_change', 'chi_cos_change', 'common_translation')


def future_coordinates(record):
    shift = np.asarray(record['labels']['translation_future'])
    return {key+'_gen': (np.asarray(record['labels'][key+'_future'])+shift[:, None])[None]
            for key in ('X', 'P', 'I')}


def observables(generated, inputs, flexible):
    x, p = (np.asarray(generated[key], dtype=float) for key in ('X_gen', 'P_gen'))
    lh = np.asarray(inputs['ligand_graph']['atomic_numbers']) > 1
    ph = np.asarray(inputs['protein_topology']['atomic_numbers']) > 1
    pocket = np.asarray(flexible['encoder_inputs']['protein_indices'])
    pocket = pocket[ph[pocket]]
    residues, slots = np.unique(np.asarray(inputs['protein_topology']['residue_indices'])[ph], return_inverse=True)
    pocket_slots = np.searchsorted(residues, np.unique(np.asarray(inputs['protein_topology']['residue_indices'])[pocket]))
    distances = np.empty((*x.shape[:2], len(residues)))
    for m, h in np.ndindex(x.shape[:2]):
        atom_distance = cKDTree(x[m, h, lh]).query(p[m, h, ph])[0]
        distances[m, h] = np.inf
        np.minimum.at(distances[m, h], slots, atom_distance)
    centre, c0 = p.mean(-2, keepdims=True), np.asarray(inputs['P0']).mean(0)
    xr, pr = x[:, :, lh]-centre, p[:, :, pocket]-centre
    x0 = np.asarray(inputs['X_obs'])[-1, lh]-c0
    p0 = np.asarray(inputs['P0'])[pocket]-c0
    ligand_displacement = np.sqrt(np.square(xr-x0).sum(-1).mean(-1))
    pocket_displacement = np.sqrt(np.square(pr-p0).sum(-1).mean(-1))
    rg = [np.sqrt(np.square(a-a.mean(-2, keepdims=True)).sum(-1).mean(-1)) for a in (xr, pr)]
    indices = np.asarray(flexible['sidechains']['chi_atoms'], dtype=int).reshape(-1, 4)
    q, valid = torsion_phase(p, indices)
    q0, valid0 = torsion_phase(np.asarray(inputs['P0']), indices)
    phase = q*q0.conj()
    if len(indices):
        sin, cos = phase.imag.mean(-1), phase.real.mean(-1)
    else:
        sin, cos = np.zeros(x.shape[:2]), np.ones(x.shape[:2])
    feature = np.stack((ligand_displacement, pocket_displacement, *rg,
        (1/(1+(distances[..., pocket_slots]/4.5)**6)).mean(-1), sin, cos,
        np.linalg.norm(centre[..., 0, :]-c0, axis=-1)), -1)
    return {'features': feature, 'distances': distances, 'pocket_slots': pocket_slots,
            'residues': residues, 'chi_phase': q, 'chi_valid': valid & valid0,
            'ligand_relative': xr, 'pocket_relative': pr, 'ligand_last': x0, 'pocket_last': p0}


def ensemble_scores(samples, truth, scales):
    """Fair ensemble ES/CRPS; truth is one realized future, samples are independent."""
    x, y = np.asarray(samples, dtype=float), np.asarray(truth, dtype=float)
    m = len(x)
    assert m >= 2 and x.shape[1:] == y.shape and np.all(np.asarray(scales) > 0)
    z, reference = (x/scales).reshape(m, -1), (y/scales).ravel()
    target = np.linalg.norm(z-reference, axis=-1).mean()
    pairs = sum(np.linalg.norm(z[i]-z[j]) for i in range(m) for j in range(i))
    es = (target-pairs/(m*(m-1)))/np.sqrt(reference.size)
    absolute = np.abs(x-y).mean(0)
    pair = sum(np.abs(x[i]-x[j]) for i in range(m) for j in range(i))/(m*(m-1))
    result = {'feature_energy_score': float(es), 'features': {}}
    for i, name in enumerate(FEATURES):
        values = {'crps': float((absolute-pair)[..., i].mean())}
        for coverage in (.5, .9):
            low, high = np.quantile(x[..., i], [(1-coverage)/2, (1+coverage)/2], axis=0)
            label = str(int(100*coverage))
            values['coverage_'+label] = float(((y[..., i] >= low) & (y[..., i] <= high)).mean())
            values['width_'+label] = float((high-low).mean())
        result['features'][name] = values
    return result


def contact_scores(pred, truth, inputs):
    d, ref = pred['distances'], truth['distances'][0]
    contact, target = d < 4.5, ref < 4.5
    dt = float(inputs['dt_ps'])
    result = {}
    for name, indices in (('all_residues', np.arange(d.shape[-1])), ('observed_pocket', pred['pocket_slots'])):
        if len(indices) == 0:
            result[name] = {'residues': 0, 'defined': False, 'brier': None, 'distance_w1_angstrom': None,
                            'predicted_contact_fraction': None, 'true_contact_fraction': None}
            continue
        a, b = contact[..., indices], target[..., indices]
        result[name] = {'residues': len(indices), 'brier': float(((a.mean(0)-b)**2).mean()),
            'distance_w1_angstrom': float(np.mean([wasserstein_distance(d[..., i].ravel(), ref[:, i]) for i in indices])),
            'predicted_contact_fraction': float(a.mean()), 'true_contact_fraction': float(b.mean())}
        for lag in (1, 2, 4, 8):
            if lag >= len(ref): continue
            pa = (a[:, lag:] & a[:, :-lag]).mean((0, 1))
            pb = (b[lag:] & b[:-lag]).mean(0)
            result[name]['joint_occurrence_mae_lag_'+str(lag*dt)+'ps'] = float(np.abs(pa-pb).mean())
    numbers = np.asarray(inputs['protein_topology']['atomic_numbers'])
    heavy = numbers > 1
    rids = np.asarray(inputs['protein_topology']['residue_indices'])[heavy]
    initial = np.full(len(pred['residues']), np.inf)
    lh = np.asarray(inputs['ligand_graph']['atomic_numbers']) > 1
    values = cKDTree(np.asarray(inputs['X_obs'])[-1, lh]).query(np.asarray(inputs['P0'])[heavy])[0]
    np.minimum.at(initial, np.searchsorted(pred['residues'], rids), values)
    active = initial < 4.5
    result['initial_contact_residues'] = int(active.sum())
    if active.any():
        span, censored = retention(contact[..., active], dt)
        reference_span, reference_censored = retention(target[..., active], dt)
        result['retention'] = {'generated_span_ps': float(span.mean()), 'true_span_ps': float(reference_span.mean()),
            'span_mae_ps': float(np.abs(span.mean(0)-reference_span).mean()),
            'generated_right_censored_fraction': float(censored.mean()),
            'true_right_censored_fraction': float(reference_censored.mean())}
    return result


def graph_torsions(graph):
    numbers = np.asarray(graph['atomic_numbers'])
    neighbours = [set() for _ in numbers]
    for a, b in graph['bonds']:
        if numbers[a] > 1 and numbers[b] > 1:
            neighbours[a].add(b); neighbours[b].add(a)
    rows = set()
    for b, c in graph['bonds']:
        for a, d in product(neighbours[b]-{c}, neighbours[c]-{b}):
            if a != d:
                row = (int(a), int(b), int(c), int(d))
                rows.add(min(row, row[::-1]))
    return np.asarray(sorted(rows), dtype=int).reshape(-1, 4)


def periodic_scores(q, valid, target, target_valid, dt):
    """Circular histograms and autocorrelations, with invalid-angle coverage explicit."""
    count = q.shape[-1]
    result = {'torsions': count, 'invalid_fraction': float((~valid).mean()) if count else None,
              'true_invalid_fraction': float((~target_valid).mean()) if count else None}
    if not count: return result
    bins = np.linspace(-np.pi, np.pi, 37)
    js = []
    for i in range(count):
        a, b = np.angle(q[..., i][valid[..., i]]), np.angle(target[..., i][target_valid[..., i]])
        if len(a) and len(b):
            ha, hb = np.histogram(a, bins)[0], np.histogram(b, bins)[0]
            js.append(float(jensenshannon(ha/ha.sum(), hb/hb.sum())**2))
    result.update(histogram_js=defined_mean(js), histogram_defined_torsions=len(js))
    for lag in (1, 2, 4, 8):
        if lag >= q.shape[1]: continue
        keep = valid[:, lag:] & valid[:, :-lag] & target_valid[None, lag:] & target_valid[None, :-lag]
        complete = keep.all((0, 1))
        a = (q[:, lag:]*q[:, :-lag].conj()).real.mean((0, 1))
        b = (target[lag:]*target[:-lag].conj()).real.mean(0)
        result['correlation_'+str(lag*dt)+'ps'] = {
            'mae': defined_mean(np.abs(a[complete]-b[complete]).tolist()),
            'defined_torsions': int(complete.sum())}
    return result


def motion_coupling(pred, truth, dt):
    signals = []
    for observed in (pred, truth):
        channels = []
        for role in ('ligand', 'pocket'):
            path, last = observed[role+'_relative'], observed[role+'_last']
            delta = np.diff(np.concatenate((np.broadcast_to(last, (len(path), 1, *last.shape)), path), axis=1), axis=1)
            channels.append(np.sqrt(np.square(delta).sum(-1).mean(-1)))
        signals.append(channels)
    result = []
    for lag in (0, 1, 2, 4, 8):
        if lag >= pred['features'].shape[1]: continue
        values = []
        for ligand, pocket in signals:
            a, b = (ligand[:, :-lag], pocket[:, lag:]) if lag else (ligand, pocket)
            a, b = a-a.mean(-1, keepdims=True), b-b.mean(-1, keepdims=True)
            den = np.sqrt(np.square(a).sum(-1)*np.square(b).sum(-1))
            values.append([float(v/d) if d > 1e-12 else None for v, d in zip((a*b).sum(-1), den)])
        mean, ref = defined_mean(values[0]), values[1][0]
        result.append({'pocket_lag_ps': lag*dt, 'predicted': mean, 'truth': ref,
            'absolute_error': abs(mean-ref) if mean is not None and ref is not None else None,
            'defined_paths': sum(v is not None for v in values[0])})
    return result
