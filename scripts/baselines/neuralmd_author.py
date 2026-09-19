"""Observation-only transfer of the pinned author NeuralMD ODE checkpoint."""
import csv
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ASSETS = Path(os.environ.get('QFIELD_NEURALMD_ASSETS', 'external/neuralmd'))
VENDOR = ASSETS/'vendor/NeuralMD-a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe'


def load_author_model(device):
    sys.path[:0] = [str(ASSETS/'deps'), str(VENDOR), str(VENDOR/'examples')]
    from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01
    args = SimpleNamespace(model_3d_ligand='FrameNet01', model_3d_protein='FrameNetProtein03',
        emb_dim=128, FrameNet_cutoff=5, FrameNet_num_layers=4, FrameNet_complex_layer=1,
        FrameNet_num_radial=100, FrameNet_rbf_type='RBF_repredding_01', FrameNet_gamma=None,
        FrameNet_readout='mean', NeuralMD_velocity_refined_value_coefficient=1e-6,
        use_MLP_velocity=False)
    model = NeuralMD_Binding01(args)
    state = torch.load(ASSETS/'assets/model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state['binding_model'], strict=True)
    return model.to(device).eval()


def prepare_inputs(inputs, device):
    """Use the observed complex centre and last two observed ligand frames."""
    observed = np.asarray(inputs['X_obs'])
    protein = np.asarray(inputs['P0'])
    centre = np.concatenate((protein, np.asarray(inputs['I0']), observed[-1])).mean(0)
    numbers = np.asarray(inputs['ligand_graph']['atomic_numbers'])
    heavy = numbers > 1
    top = inputs['protein_topology']
    names = np.asarray(top['atom_names'])
    residues = np.asarray(top['residue_indices'])
    residue_names = np.asarray(top['residue_names'])
    indices = [np.flatnonzero(names == name) for name in ('N', 'CA', 'C')]
    np.testing.assert_array_equal(residues[indices[0]], residues[indices[1]])
    np.testing.assert_array_equal(residues[indices[1]], residues[indices[2]])
    with (VENDOR/'NeuralMD/datasets/MISATO/utils/atoms_residue_map.pickle').open('rb') as stream:
        residue_codes = {value: key-1 for key, value in pickle.load(stream).items()}
    with (VENDOR/'NeuralMD/datasets/periodic_table.csv').open() as stream:
        masses = {int(row['AtomicNumber']): float(row['AtomicMass']) for row in csv.DictReader(stream)}
    codes=[]
    for index in indices[1]:
        name=residue_names[index]
        if name=='HIS':
            atoms=np.flatnonzero(residues==residues[index]);labels=names[atoms]
            assert 'HE2' in labels and 'HD1' not in labels, 'Author HIE requires an observed NE2 hydrogen'
            h=atoms[np.flatnonzero(labels=='HE2')[0]];n=atoms[np.flatnonzero(labels=='NE2')[0]]
            assert np.linalg.norm(protein[h]-protein[n])<1.3
            name='HIE'
        codes.append(residue_codes[name])
    f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    i = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long, device=device)
    condition = (i(numbers[heavy]-1), i(np.zeros(heavy.sum())),
        f([masses[int(z)] for z in numbers[heavy]]),
        *(f(protein[index]-centre) for index in indices),
        i(codes), i(np.zeros(len(indices[1]))))
    # The released weight was trained with unscaled adjacent-frame differences.
    initial = (f(observed[-1, heavy]-observed[-2, heavy]), f(observed[-1, heavy]-centre))
    return initial, condition, centre, heavy


@torch.no_grad()
def predict(model, inputs, future_frames, device):
    from torchdiffeq import odeint
    initial, condition, centre, heavy = prepare_inputs(inputs, device)
    assert float(inputs['dt_ps']) == 80., 'supplemental protocol uses80ps task windows'
    times = torch.arange(future_frames+1, device=device, dtype=torch.float32)/100.
    _, positions = odeint(lambda t, state: model(t, state, condition), initial, times,
        method='euler', options={'step_size': .1})
    return positions[1:].cpu().numpy().astype(np.float64)+centre, heavy
