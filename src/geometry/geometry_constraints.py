"""Shared calibrated geometric inequalities, including coordinate encoding margins."""
import numpy as np
import torch
from ligand_geometry import angles, signed_volume
from trajectory_delivery import coordinate_encoding_error_bound


def geometric_values(flat, geometry, padding, config):
    x = flat.reshape(-1, 3)
    residuals = []
    for name in ['bond', 'angle']:
        indices = geometry['bonds' if name == 'bond' else 'angles']
        value = (torch.linalg.vector_norm(x[indices[:, 0]]-x[indices[:, 1]], dim=-1)
                 if name == 'bond' else angles(x, indices))
        scale = geometry[name+'_scale']
        low = geometry[name+'_min']+padding[name]
        high = geometry[name+'_max']-padding[name]
        if name == 'angle':
            low = torch.where(geometry['angle_min'] == 0, geometry['angle_min'], low)
            high = torch.where(geometry['angle_max'] == torch.pi, geometry['angle_max'], high)
        residuals.extend(((value-low)/scale, (high-value)/scale))
    residuals.append(signed_volume(x, geometry['tetrahedra'])*geometry['tetrahedron_sign']/
        geometry['tetrahedron_scale']-config['configuration_min_fraction']-
        padding['volume']/geometry['tetrahedron_scale'])
    return torch.cat(residuals)


def connected_encoding_padding(geometry, record, centre, config):
    slots = {tuple(sorted(pair)): i for i, pair in enumerate(geometry['bonds'].tolist())}
    legs = torch.tensor([[slots[tuple(sorted((a, b)))], slots[tuple(sorted((c, b)))]]
        for a, b, c in geometry['angles'].tolist()], dtype=torch.long).reshape(-1, 2)
    tetra = torch.tensor([[slots[tuple(sorted((ct, a)))] for a in (a, b, c)]
        for ct, a, b, c in geometry['tetrahedra'].tolist()], dtype=torch.long).reshape(-1, 3)
    origin = np.asarray(record['transform']['origin'])
    extent = max(np.abs(np.concatenate((record['inputs']['P0'], record['inputs']['I0']))+origin).max(),
        np.abs(centre+origin).max()+float(geometry['bond_max'].sum()))
    axis_error = coordinate_encoding_error_bound(extent)
    pair_error = 2*np.sqrt(3)*axis_error
    angle_error = torch.asin(pair_error/geometry['bond_min'][legs]).sum(-1)
    edge = geometry['bond_max'][tetra]
    volume_error = (pair_error*(edge[:, 0]*edge[:, 1]+edge[:, 0]*edge[:, 2]+edge[:, 1]*edge[:, 2])
        +pair_error**2*edge.sum(-1)+pair_error**3)
    padding = dict(bond=pair_error+2*config['constraint_tolerance']*geometry['bond_scale'],
        angle=angle_error+2*config['constraint_tolerance']*geometry['angle_scale'], volume=volume_error)
    return padding, axis_error, pair_error
