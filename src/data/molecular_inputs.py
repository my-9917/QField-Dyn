"""Build model tensors from observed ligand, protein and ion histories."""
import numpy as np
from scipy.spatial import cKDTree
import torch
INPUT_VERSION = 'semiflexible_protein_aligned_v2'

def model_inputs(sample, pocket_radius, device='cpu', protein_indices=None):
    x = sample['inputs']
    graph, topology = x['ligand_graph'], x['protein_topology']
    ligand_numbers = graph['atomic_numbers']
    protein_numbers = topology['atomic_numbers']
    if protein_indices is None:
        distance = cKDTree(x['X_obs'][-1, ligand_numbers > 1]).query(x['P0'])[0]
        residues = np.unique(topology['residue_indices'][(distance <= pocket_radius) & (protein_numbers > 1)])
        selected = np.flatnonzero(np.isin(topology['residue_indices'], residues))
    else:
        selected = np.asarray(protein_indices,dtype=int)
    bonds = np.zeros((len(ligand_numbers), len(ligand_numbers)))
    for a,b in graph['bonds']:bonds[a,b]=bonds[b,a]=1.
    degree=bonds.sum(1)
    separation=np.linalg.norm(x['X_obs'][-1,:,None]-x['X_obs'][-1,None,:],axis=-1)
    bond_mean=(bonds*separation).sum(1)/degree
    bond_variance=(bonds*(separation-bond_mean[:,None])**2).sum(1)/degree
    # Connectivity and geometry are available in both training and anonymous inputs.
    chemistry=np.stack(((bonds[:,ligand_numbers>1]).sum(1)/4.,
                        (bonds[:,ligand_numbers==1]).sum(1)/4.,bond_mean/2.,bond_variance),axis=-1)
    md = x['md_protocol']
    temperature = md['TEMP']
    ensemble = str(md['ENSEMBLE']).upper()
    # Explicit units and known-value indicators preserve missing source metadata.
    condition = [x['dt_ps']/200., 0. if md['TIMESTEP'] is None else md['TIMESTEP']/2.,
        0. if temperature is None else float(temperature)/300., float(temperature is not None),
        float('NVT' in ensemble), float('NPT' in ensemble), float(md['ENSEMBLE'] is not None),
        float('Amber ff14SB' in md['FF']), float('GAFF' in md['FF']), float(md['WAT'] == 'TIP3P')]
    floating = {'X_obs': x['X_obs'], 'P0': x['P0'][selected], 'I0': x['I0'],
        'ion_charge_e': x['ion_topology']['charge_e'], 'chemistry': chemistry,
        'bonds': bonds, 'condition': condition, 'dt_ps': x['dt_ps']}
    result = {k: torch.as_tensor(np.array(v, copy=True), dtype=torch.float32, device=device) for k, v in floating.items()}
    result.update({'ligand_numbers': torch.as_tensor(ligand_numbers, dtype=torch.long, device=device),
        'protein_numbers': torch.as_tensor(protein_numbers[selected], dtype=torch.long, device=device),
        'protein_indices': torch.as_tensor(selected, dtype=torch.long, device=device),
        'ion_numbers': torch.as_tensor(x['ion_topology']['atomic_numbers'], dtype=torch.long, device=device),
        'ion_source_indices': torch.as_tensor(x['ion_topology']['source_atom_indices'], dtype=torch.long, device=device),
        'heavy_mask': torch.as_tensor(ligand_numbers > 1, device=device)})
    return result
