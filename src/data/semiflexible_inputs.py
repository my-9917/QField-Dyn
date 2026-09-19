"""Ligand targets and fixed observed environments in one physical coordinate frame."""
import numpy as np
import torch

from molecular_inputs import model_inputs
from native_coordinates import topology_universe, unwrap_source

VERSION = 'semiflexible_protein_aligned_v2'


def protein_alignment(xyz, reference, protein_heavy):
    """Proper Kabsch rotations map each protein frame to the observed reference."""
    moving = np.asarray(xyz[:, protein_heavy], dtype=np.float64)
    fixed = np.asarray(reference[protein_heavy], dtype=np.float64)
    centres = moving.mean(1)
    origin = fixed.mean(0)
    covariance = np.einsum('tni,nj->tij', moving-centres[:, None], fixed-origin)
    u, _, vh = np.linalg.svd(covariance)
    sign = np.linalg.det(u@vh)
    u[:, :, -1] *= sign[:, None]
    rotation = u@vh
    aligned = np.einsum('tni,tij->tnj', xyz-centres[:, None], rotation)+origin
    return aligned, rotation, centres


def observation_inputs(case):
    """Align observed frames to their final protein frame and retain fixed context."""
    atoms = case['trajectory'].atoms
    xyz = case['coordinates_angstrom']
    pi, li, ii = (case[key] for key in ('protein_indices', 'ligand_indices', 'ion_indices'))
    numbers = case['atomic_numbers']
    aligned, rotation, centres = protein_alignment(xyz, xyz[-1], pi[numbers[pi] > 1])
    origin = xyz[-1, pi].mean(0)
    p0, i0 = xyz[-1, pi]-origin, xyz[-1, ii]-origin
    source_bonds = np.asarray(atoms.bonds.indices, dtype=int).reshape(-1, 2)
    source_bonds = source_bonds[np.isin(source_bonds, li).all(1)]
    reverse = np.full(len(atoms), -1, dtype=int); reverse[li] = np.arange(len(li))
    bonds = np.unique(np.sort(reverse[source_bonds], axis=1), axis=0)
    assert len(bonds) > 0 and (np.bincount(bonds.ravel(), minlength=len(li)) > 0).all()
    inputs = {'input_version': VERSION, 'X_obs': aligned[:, li]-origin, 'P0': p0, 'I0': i0,
        'dt_ps': case['meta']['dt_ps'],
        'protein_topology': {'source_atom_indices': pi, 'atomic_numbers': numbers[pi],
            'atom_names': atoms.names[pi], 'residue_indices': atoms.resindices[pi],
            'residue_names': atoms.resnames[pi]},
        'ligand_graph': {'source_atom_indices': li, 'atomic_numbers': numbers[li],
            'atom_names': atoms.names[li], 'bonds': bonds},
        'ion_topology': {'source_atom_indices': ii, 'atomic_numbers': numbers[ii],
            'charge_e': np.array([{'Na': 1., 'Cl': -1., 'K': 1., 'Ca': 2., 'Mg': 2.,
                                  'Zn': 2., 'Mn': 2., 'Fe': 2.}[v] for v in atoms.elements[ii]])},
        'md_protocol': {'TIMESTEP': None, 'TEMP': None, 'ENSEMBLE': None, 'FF': '', 'WAT': None}}
    # The complete environment remains accessible when a ligand moves beyond its initial pocket.
    encoded = model_inputs({'inputs': inputs}, pocket_radius=6., protein_indices=np.arange(len(pi)))
    return inputs, encoded, {'origin': origin, 'reference': 'last_observed',
        'coordinate_version': VERSION, 'units': 'Angstrom', 'alignment': 'proper_Kabsch_protein_heavy',
        'observed_rotations': rotation, 'observed_centres': centres}


def training_record(group, topology, spec, identifier):
    """Prepare a complete native task directly from coordinates and ligand connectivity."""
    numbers = group['atoms_number'][:]
    np.testing.assert_array_equal(numbers, topology['numbers'])
    start = int(group['molecules_begin_atom_index'][-1])
    pi, li, ii = np.arange(start), np.arange(start, len(numbers)), np.empty(0, dtype=int)
    assert set(topology['labels'][li]) == {'MOL'}
    k, h = spec['n_obs'], spec['n_pred']
    assert k+h <= group['trajectory_coordinates'].shape[0]
    universe = topology_universe(topology)
    xyz = unwrap_source(group['trajectory_coordinates'][:k+h], universe, pi)
    universe.delete_bonds(universe.bonds)
    universe.add_bonds(topology['bonds'][(topology['bonds'] >= start).all(1)])
    case = {'meta': dict(spec, id=identifier), 'trajectory': universe, 'coordinates_angstrom': xyz[:k],
        'atomic_numbers': numbers, 'protein_indices': pi, 'ligand_indices': li, 'ion_indices': ii}
    inputs, encoded, transform = observation_inputs(case)
    aligned, _, _ = protein_alignment(xyz[k:k+h], xyz[k-1], pi[numbers[pi] > 1])
    target = torch.as_tensor(aligned[:, li]-transform['origin'], dtype=torch.float64)
    return {'meta': case['meta'], 'input_version': VERSION, 'inputs': inputs,
        'encoder_inputs': encoded, 'X_future': target, 'transform': transform,
        'source_static_box': topology['dimensions']}


def assemble_prediction(case, ligand, origin):
    """Write every nonligand atom from the last observation, preserving its stored coordinates."""
    xyz = np.broadcast_to(case['coordinates_angstrom'][-1],
                          (case['meta']['n_pred'], case['meta']['n_atoms'], 3)).copy()
    assert ligand.shape == (case['meta']['n_pred'], len(case['ligand_indices']), 3)
    xyz[:, case['ligand_indices']] = np.asarray(ligand)+origin
    return xyz
