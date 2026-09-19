"""Validate prefix budgets and geometry after actual XTC coordinate encoding."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from trajectory_delivery import read_xtc, write_xtc
from semiflexible_scores import observables, physics_scores
from prefix_selection import select_candidates


def encode_coordinates(selected, record):
    """Return the actual XTC ligand/environment coordinates in the model frame."""
    m, h, n = selected.shape[:3]
    origin = np.asarray(record['transform']['origin'])
    inputs = record['inputs']
    graph, protein, ions = (inputs[k] for k in ('ligand_graph', 'protein_topology', 'ion_topology'))
    li, pi, ii = (np.asarray(t['source_atom_indices'], dtype=int) for t in (graph, protein, ions))
    indices = np.concatenate((li, pi, ii))
    assert np.array_equal(np.sort(indices), np.arange(len(indices)))
    template = np.empty((len(indices), 3))
    template[pi], template[ii] = np.asarray(inputs['P0'])+origin, np.asarray(inputs['I0'])+origin
    xyz = np.broadcast_to(template, (m*h, *template.shape)).copy()
    xyz[:, li] = selected.reshape(m*h, n, 3)+origin
    with TemporaryDirectory(prefix='qmem_prefix_xtc_') as temporary:
        filename = Path(temporary)/'coordinates.xtc'
        frame = np.arange(m*h)
        write_xtc(filename, xyz, frame*float(inputs['dt_ps']), frame, np.zeros((3, 3)))
        decoded = read_xtc(filename)['coordinates_angstrom'].reshape(m, h, -1, 3)-origin
    fixed = np.concatenate((pi, ii))
    assert np.all(decoded[:, :, fixed] == decoded[0, 0, fixed])
    inputs = dict(inputs)
    ph = np.asarray(inputs['protein_topology']['atomic_numbers']) > 1
    ih = np.asarray(inputs['ion_topology']['atomic_numbers']) > 1
    inputs['P0'], inputs['I0'] = decoded[0, 0, pi], decoded[0, 0, ii]
    env = np.concatenate((inputs['P0'][ph], inputs['I0'][ih]))
    pockets = observables(selected[:1, :1], record)['pocket_slots']
    encoded_record = dict(record, inputs=inputs, pocket_slots=pockets,
        geometry=dict(record['geometry'], environment=torch.as_tensor(env)))
    return decoded[:, :, li], encoded_record


def review_encoded_prefix(raw, selected, record, limit, prior_state=None):
    m, h = selected.shape[:2]
    decoded, encoded_record = encode_coordinates(selected, record)
    before = observables(raw, record)
    # Pocket membership belongs to the original observed condition in both score versions.
    after = observables(decoded, encoded_record)
    raw_phys = physics_scores(raw, record['geometry'])[1]
    final_phys = physics_scores(decoded, encoded_record['geometry'])[1]
    legal = all((final_phys[k] == 0).all() for k in ['bond_violations', 'angle_violations', 'local_configuration_flips'])
    rc = raw_phys['ligand_self_overlap_violations']+raw_phys['ligand_environment_overlap_violations']
    fc = final_phys['ligand_self_overlap_violations']+final_phys['ligand_environment_overlap_violations']
    contacts = [a[None] for a in (after['distances'] < 4.5).reshape(m*h, -1)]
    _, report = select_candidates(before['distances'] < 4.5, rc, contacts,
        [np.array([c]) for c in fc.ravel()], [np.array([0.])]*(m*h),
        before['pocket_slots'], limit, prior_state)
    report.update(geometry_passed=bool(legal),
        coordinate_roundtrip_max_angstrom=float(np.abs(decoded-selected).max()))
    return report
