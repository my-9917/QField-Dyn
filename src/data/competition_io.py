"""Official observation-only reads and one-trajectory XTC submissions."""
import json
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.guesser.default_guesser import DefaultGuesser
import numpy as np
from rdkit.Chem import GetPeriodicTable
from trajectory_delivery import XTC_PRECISION, write_xtc


def public_cases(root):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text())
    for tier, spec in protocol['tiers'].items():
        rows = [json.loads(line) for line in (root/tier/'manifest.jsonl').read_text().splitlines()]
        assert len(rows) == spec['n_systems']
        for row in rows:
            meta = json.loads((root/tier/row['id']/'meta.json').read_text())
            for key in ('n_obs', 'n_pred', 'dt_ps', 'ligand_resname'):
                assert row[key] == meta[key] == spec[key], (row['id'], key)
            assert meta['tier'] == tier and meta['id'] == row['id']
            assert meta['obs_index_0based'] == [0, spec['n_obs']]
            assert meta['pred_index_0based'] == [spec['n_obs'], spec['n_obs']+spec['n_pred']]
            yield row, meta


def read_observation(root, row, meta):
    root = Path(root)
    observed = mda.Universe(str(root/row['top']), str(root/row['obs']))
    assert (len(observed.trajectory), len(observed.atoms)) == (meta['n_obs'], meta['n_atoms'])
    frames, times = [], []
    for ts in observed.trajectory:
        frames.append(ts.positions.copy())
        times.append(ts.time)
    np.testing.assert_allclose(times, np.arange(meta['n_obs'])*meta['dt_ps'], rtol=0, atol=.01)
    inferred_elements = np.flatnonzero(observed.atoms.elements == '')
    if len(inferred_elements):
        elements = observed.atoms.elements.copy()
        elements[inferred_elements] = DefaultGuesser(observed).guess_types(observed.atoms.names[inferred_elements])
        observed.atoms.elements = elements
    atoms = observed.atoms
    numbers = np.array([GetPeriodicTable().GetAtomicNumber(element) for element in atoms.elements])
    assert (numbers > 0).all()
    ligand = atoms.resnames == meta['ligand_resname']
    protein = np.isin(atoms.indices, observed.select_atoms('protein or resname ACE NME').indices)
    ions = np.array([len(a.residue.atoms) == 1 and a.element in ('Na', 'Cl', 'K', 'Ca', 'Mg', 'Zn', 'Mn', 'Fe') for a in atoms])
    assert ligand.any() and protein.any()
    assert np.all(ligand.astype(int)+protein.astype(int)+ions.astype(int) == 1), meta['id']
    return {'meta':meta, 'trajectory':observed, 'coordinates_angstrom':np.asarray(frames,dtype=float),
            'atomic_numbers':numbers, 'elements_from_atom_names':inferred_elements,
            'ligand_indices':np.flatnonzero(ligand), 'protein_indices':np.flatnonzero(protein),
            'ion_indices':np.flatnonzero(ions)}


def write_prediction(case, coordinates_angstrom, output):
    """The caller supplies one complete path in original atom order and frame."""
    meta, observed = case['meta'], case['trajectory']
    xyz = np.asarray(coordinates_angstrom)
    assert xyz.shape == (meta['n_pred'], meta['n_atoms'], 3) and np.isfinite(xyz).all()
    time = np.arange(meta['n_obs'], meta['n_obs']+meta['n_pred'])*meta['dt_ps']
    path = Path(output)/f"{meta['id']}_pred.xtc"
    box = observed.trajectory.ts.triclinic_dimensions
    write_xtc(path, xyz, time, np.arange(meta['n_obs'], meta['n_obs']+meta['n_pred']),
              np.zeros((3, 3)) if box is None else box)
    return path


def assemble_prediction(case, generated, origin):
    """Restore the fixed last-observed frame and the official atom order."""
    meta = case['meta']
    xyz = np.empty((meta['n_pred'], meta['n_atoms'], 3))
    for key, indices in (('X_gen', 'ligand_indices'), ('P_gen', 'protein_indices'), ('I_gen', 'ion_indices')):
        values = generated[key]
        assert values.shape == (1, meta['n_pred'], len(case[indices]), 3)
        xyz[:, case[indices]] = values[0].detach().cpu().numpy()+origin
    return xyz


def training_slice(coordinates, tier, protocol):
    """Training caller supplies an authorized native trajectory in source units."""
    spec = protocol['tiers'][tier]
    k, h = spec['n_obs'], spec['n_pred']
    assert len(coordinates) >= k+h
    return coordinates[:k].copy(), coordinates[k:k+h].copy()
