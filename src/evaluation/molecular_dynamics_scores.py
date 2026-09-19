"""Ring geometry, time correlations and input-residue contact retention in float64."""
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

VERSION = 'molecular_dynamics_v1'


def defined_mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def torsion_indices(molecule):
    """All heavy four-atom paths around non-triple central bonds, unique up to reversal."""
    rows = []
    for bond in molecule.GetBonds():
        j, k = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        a, b = molecule.GetAtomWithIdx(j), molecule.GetAtomWithIdx(k)
        if min(a.GetAtomicNum(), b.GetAtomicNum()) <= 1 or bond.GetBondTypeAsDouble() == 3:
            continue
        # Triple-bond endpoints define collinear planes, so their dihedral is undefined.
        if any(e.GetBondTypeAsDouble() == 3 for atom in (a, b) for e in atom.GetBonds()):
            continue
        for i in sorted(n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != k):
            for l in sorted(n.GetIdx() for n in b.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in (j, i)):
                rows.append((i, j, k, l))
    return np.array(rows, dtype=int).reshape(-1, 4)


def torsion_phase(xyz, indices):
    """Unit complex dihedrals; invalid collinear or coincident geometry stays explicit."""
    a, b, c, d = indices.T
    axis = xyz[..., c, :]-xyz[..., b, :]
    first = np.cross(xyz[..., b, :]-xyz[..., a, :], axis)
    second = np.cross(axis, xyz[..., d, :]-xyz[..., c, :])
    axis_norm = np.linalg.norm(axis, axis=-1)
    plane_norm = np.linalg.norm(first, axis=-1)*np.linalg.norm(second, axis=-1)
    valid = (axis_norm > 1e-12) & (plane_norm > 1e-12)
    real = (first*second).sum(-1)
    imaginary = np.divide((np.cross(first, second)*axis).sum(-1), axis_norm,
        out=np.zeros_like(real), where=valid)
    phase = np.divide(real+1j*imaginary, plane_norm, out=np.zeros(real.shape, complex), where=valid)
    return phase, valid


def displacement_correlation(paths, lag):
    """Normalized lagged displacement dot product per path; stationary paths are undefined."""
    delta = np.diff(paths, axis=1)
    a, b = delta[:, :-lag], delta[:, lag:]
    numerator = (a*b).sum(axis=(1, 2, 3))
    denominator = np.sqrt(np.square(a).sum(axis=(1, 2, 3))*np.square(b).sum(axis=(1, 2, 3)))
    return [float(n/d) if d > 0 else None for n, d in zip(numerator, denominator)]


def structure_scores(paths, target, molecule, lags):
    x, y = np.asarray(paths, dtype=float), np.asarray(target, dtype=float)
    heavy = np.array([a.GetAtomicNum() > 1 for a in molecule.GetAtoms()])
    rings = [list(r) for r in molecule.GetRingInfo().AtomRings()]
    aromatic = [r for r in rings if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    ring_errors, generated_planes, reference_planes = [], [], []
    for ring in rings:
        a, b = np.array(list(combinations(ring, 2))).T
        dx = np.linalg.norm(x[:, :, a]-x[:, :, b], axis=-1)
        dy = np.linalg.norm(y[:, a]-y[:, b], axis=-1)
        ring_errors.append(float(np.abs(dx-dy[None]).mean()))
    for ring in aromatic:
        px, py = x[:, :, ring], y[:, ring]
        generated_planes.append(float((np.linalg.svd(px-px.mean(-2, keepdims=True), compute_uv=False)[..., -1]/np.sqrt(len(ring))).mean()))
        reference_planes.append(float((np.linalg.svd(py-py.mean(-2, keepdims=True), compute_uv=False)[..., -1]/np.sqrt(len(ring))).mean()))
    indices = torsion_indices(molecule)
    qx, vx = torsion_phase(x, indices)
    qy, vy = torsion_phase(y[None], indices)
    result = {'ring_count': len(rings), 'aromatic_ring_count': len(aromatic), 'torsion_count': len(indices),
        'ring_pair_distance_mae_angstrom': defined_mean(ring_errors),
        'generated_aromatic_plane_rms_angstrom': defined_mean(generated_planes),
        'reference_aromatic_plane_rms_angstrom': defined_mean(reference_planes),
        'generated_torsion_invalid_fraction': float((~vx).mean()) if len(indices) else None,
        'reference_torsion_invalid_fraction': float((~vy).mean()) if len(indices) else None}
    for lag in lags:
        generated = displacement_correlation(x[:, :, heavy], lag)
        reference, = displacement_correlation(y[None, :, heavy], lag)
        average = defined_mean(generated)
        result.update({f'generated_displacement_corr_lag{lag}': average,
            f'reference_displacement_corr_lag{lag}': reference,
            f'displacement_corr_defined_fraction_lag{lag}': sum(v is not None for v in generated)/len(generated),
            f'displacement_corr_error_lag{lag}': abs(average-reference) if average is not None and reference is not None else None})
        # A torsion is compared only when all sampled and reference planes are defined.
        # Invalid coverage is reported above and remains a physical-quality diagnostic.
        valid = vx[:, :-lag] & vx[:, lag:] & vy[:, :-lag] & vy[:, lag:]
        complete = valid.all(axis=(0, 1))
        cx = (qx[:, lag:]*qx[:, :-lag].conj()).real.mean(axis=(0, 1))
        cy = (qy[:, lag:]*qy[:, :-lag].conj()).real.mean(axis=(0, 1))
        result.update({f'generated_torsion_corr_lag{lag}': defined_mean(cx[complete].tolist()),
            f'reference_torsion_corr_lag{lag}': defined_mean(cy[complete].tolist()),
            f'torsion_corr_mae_lag{lag}': defined_mean(np.abs(cx[complete]-cy[complete]).tolist()),
            f'torsion_corr_defined_fraction_lag{lag}': float(complete.mean()) if len(indices) else None})
    return result


def residue_contacts(ligand, protein, residue_slots, count, cutoff):
    """Boolean [frames,residues] contact map for fixed protein coordinates."""
    frames, atoms, _ = ligand.shape
    near = cKDTree(protein).query_ball_point(ligand.reshape(-1, 3), cutoff, return_sorted=False)
    lengths = np.fromiter((len(n) for n in near), dtype=int, count=len(near))
    protein_index = np.concatenate(near).astype(int)
    frame_index = np.repeat(np.arange(len(near))//atoms, lengths)
    contacts = np.zeros((frames, count), dtype=bool)
    contacts[frame_index, residue_slots[protein_index]] = True
    return contacts


def retention(contact, dt_ps):
    """Remaining span of an observed initial contact, sampled at future endpoints."""
    retained = np.logical_and.accumulate(contact, axis=-2)
    return retained.sum(axis=-2)*dt_ps, retained[..., -1, :]


def static_contact_scores(paths, target, raw, cutoff=4.5):
    """Compare static generated protein contacts with matching moving-reference residues."""
    x = raw['inputs']; y = raw['labels']
    heavy = x['ligand_graph']['atomic_numbers'] > 1
    topology = x['protein_topology']
    protein_heavy = topology['atomic_numbers'] > 1
    distances = cKDTree(x['X_obs'][-1, heavy]).query(x['P0'])[0]
    residues = np.unique(topology['residue_indices'][protein_heavy & (distances <= cutoff)])
    names = ('generated_initial_contact_span_ps', 'reference_initial_contact_span_ps',
        'initial_contact_span_mae_ps', 'reference_static_contact_span_mae_ps',
        'generated_contact_right_censored_fraction', 'reference_contact_right_censored_fraction',
        'reference_contact_residue_displacement_rms_angstrom')
    if len(residues) == 0:
        return {'initial_contact_residues': 0, **dict.fromkeys(names)}
    selected = np.flatnonzero(protein_heavy & np.isin(topology['residue_indices'], residues))
    slots = np.searchsorted(residues, topology['residue_indices'][selected])
    generated = np.asarray(paths, dtype=float)[:, :, heavy]
    reference = np.asarray(target, dtype=float)[:, heavy]
    m, h, n, _ = generated.shape
    contact = residue_contacts(generated.reshape(m*h, n, 3), x['P0'][selected], slots, len(residues), cutoff).reshape(m, h, -1)
    ref_contact = np.array([residue_contacts(frame[None], protein[selected], slots, len(residues), cutoff)[0]
        for frame, protein in zip(reference, y['P_future'])])
    ref_static = residue_contacts(reference, x['P0'][selected], slots, len(residues), cutoff)
    span, censored = retention(contact, x['dt_ps'])
    ref_span, ref_censored = retention(ref_contact, x['dt_ps'])
    static_span, _ = retention(ref_static, x['dt_ps'])
    values = (float(span.mean()), float(ref_span.mean()), float(np.abs(span.mean(0)-ref_span).mean()),
        float(np.abs(static_span-ref_span).mean()), float(censored.mean()), float(ref_censored.mean()),
        float(np.sqrt(np.square(y['P_future'][:, selected]-x['P0'][selected]).sum(-1).mean())))
    return {'initial_contact_residues': len(residues), **dict(zip(names, values))}


def molecular_dynamics_scores(paths, target, raw, lags):
    return {**structure_scores(paths, target, raw['inputs']['ligand_graph']['molecule'], lags),
        **static_contact_scores(paths, target, raw)}


def summarize_dynamics(records):
    """Average applicable values; carry the denominator for every metric."""
    return {name: {'mean': defined_mean([r[name] for r in records]),
        'defined_records': sum(r[name] is not None for r in records), 'total_records': len(records)}
        for name in records[0]}
