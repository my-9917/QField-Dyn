"""Observed ligand geometry and fixed-environment contacts with train-only margins."""
from itertools import combinations
import numpy as np
import torch
from rdkit import Chem


def angles(x, indices):
    a, b, c = indices.T
    u, v = x[..., a, :]-x[..., b, :], x[..., c, :]-x[..., b, :]
    return torch.atan2(torch.linalg.vector_norm(torch.cross(u, v, dim=-1), dim=-1), (u*v).sum(-1))


def signed_volume(x, indices):
    centre, a, b, c = indices.T
    u, v, w = (x[..., i, :]-x[..., centre, :] for i in (a, b, c))
    return (u*torch.cross(v, w, dim=-1)).sum(-1)


def build_geometry(inputs, calibration):
    assert calibration['completed'] and calibration['partition'] == 'train'
    x = torch.as_tensor(inputs['X_obs'], dtype=torch.float64)
    graph = inputs['ligand_graph']
    numbers = np.asarray(graph['atomic_numbers'])
    bonds = torch.as_tensor(graph['bonds'], dtype=torch.long)
    neighbours = [[] for _ in numbers]
    for a, b in bonds.tolist():
        neighbours[a].append(b); neighbours[b].append(a)
    triples = torch.tensor([(a, b, c) for b, near in enumerate(neighbours)
                            for a, c in combinations(sorted(near), 2)], dtype=torch.long).reshape(-1, 3)
    tetra = torch.tensor([(i, *sorted(near)[:3]) for i, near in enumerate(neighbours)
                          if numbers[i] == 6 and len(near) == 4], dtype=torch.long).reshape(-1, 4)
    volume = signed_volume(x, tetra)
    stable = volume.amin(0)*volume.amax(0) > 0
    tetra, volume = tetra[stable], volume[:, stable]
    result = {'bonds': bonds, 'angles': triples, 'tetrahedra': tetra,
              'tetrahedron_sign': volume[-1].sign(), 'tetrahedron_scale': volume[-1].abs(),
              'calibration_version': calibration['version']}
    for kind, value in [('bond', torch.linalg.vector_norm(x[:, bonds[:, 0]]-x[:, bonds[:, 1]], dim=-1)),
                        ('angle', angles(x, triples))]:
        low, high = value.amin(0), value.amax(0)
        scale = (low+high)/2 if kind == 'bond' else torch.ones_like(low)
        margin = calibration['margins']['ligand_'+kind]*scale
        result[kind+'_min'] = (low-margin).clamp_min(0)
        result[kind+'_max'] = high+margin if kind == 'bond' else (high+margin).clamp_max(torch.pi)
        result[kind+'_scale'] = scale
    heavy = torch.as_tensor(np.flatnonzero(numbers > 1))
    table = Chem.GetPeriodicTable()
    radii = torch.tensor([table.GetRcovalent(int(z)) for z in numbers], dtype=x.dtype)[heavy]
    excluded = {tuple(sorted(v)) for v in bonds.tolist()}
    excluded.update(tuple(sorted(v)) for near in neighbours for v in combinations(near, 2))
    pairs = torch.tensor([(a, b) for a, b in combinations(heavy.tolist(), 2)
                          if (a, b) not in excluded], dtype=torch.long).reshape(-1, 2)
    all_radii = torch.tensor([table.GetRcovalent(int(z)) for z in numbers], dtype=x.dtype)
    limit = torch.minimum(all_radii[pairs[:, 0]]+all_radii[pairs[:, 1]],
                          torch.linalg.vector_norm(x[:, pairs[:, 0]]-x[:, pairs[:, 1]], dim=-1).amin(0))
    environment = torch.as_tensor(np.concatenate((inputs['P0'], inputs['I0'])), dtype=x.dtype)
    en = np.concatenate((inputs['protein_topology']['atomic_numbers'], inputs['ion_topology']['atomic_numbers']))
    environment = environment[en > 1]
    er = torch.tensor([table.GetRcovalent(int(z)) for z in en[en > 1]], dtype=x.dtype)
    cross_limit = radii[:, None]+er[None]
    for frame in x:
        cross_limit = torch.minimum(cross_limit, torch.cdist(frame[heavy], environment,
                                                            compute_mode='donot_use_mm_for_euclid_dist'))
    result.update(heavy=heavy, self_pairs=pairs, self_limits=limit,
                  environment=environment, cross_limits=cross_limit)
    return result


def geometry_terms(x, reference, split_overlap=False):
    r = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in reference.items()}
    bonds, triples = r['bonds'], r['angles']
    result = {}
    for kind, value in [('bond', torch.linalg.vector_norm(x[..., bonds[:, 0], :]-x[..., bonds[:, 1], :], dim=-1)),
                        ('angle', angles(x, triples))]:
        excess = (r[kind+'_min']-value).clamp_min(0)+(value-r[kind+'_max']).clamp_min(0)
        result[kind+'_strain'] = (excess/r[kind+'_scale']).square().sum(-1)/max(1, value.shape[-1])
    volume = signed_volume(x, r['tetrahedra'])*r['tetrahedron_sign']
    result['configuration_flip'] = (-volume/r['tetrahedron_scale']).clamp_min(0).square().sum(-1)/max(1, volume.shape[-1])
    pairs = r['self_pairs']
    distance = torch.linalg.vector_norm(x[..., pairs[:, 0], :]-x[..., pairs[:, 1], :], dim=-1)
    self_overlap = ((r['self_limits']-distance).clamp_min(0)/r['self_limits']).square().sum(-1)
    distance = torch.cdist(x[..., r['heavy'], :], r['environment'].to(x), compute_mode='donot_use_mm_for_euclid_dist')
    cross_overlap = ((r['cross_limits']-distance).clamp_min(0)/r['cross_limits']).square().sum((-1, -2))
    if split_overlap:
        result['self_overlap'] = self_overlap/len(r['heavy'])
        result['environment_overlap'] = cross_overlap/len(r['heavy'])
    else:
        result['severe_overlap'] = (self_overlap+cross_overlap)/len(r['heavy'])
    return result
