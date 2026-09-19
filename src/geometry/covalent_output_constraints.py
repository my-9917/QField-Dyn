"""Exact covalent bounds and analytic Jacobians for final output feasibility."""
import numpy as np
import torch
from geometry_constraints import connected_encoding_padding


def output_constraints(context, record, centre):
    cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in context.items()}
    padding, axis_error, pair_error = connected_encoding_padding(cpu, record, centre,
                                                               {'constraint_tolerance': 1e-7})
    device = context['weights'].device
    padding = {k: v.to(device) for k, v in padding.items()}
    g = dict(context)
    g['output_bond_min'] = g['bond_min'] + padding['bond']
    g['output_bond_max'] = g['bond_max'] - padding['bond']
    low = torch.where(g['angle_min'] == 0, g['angle_min'], g['angle_min'] + padding['angle'])
    high = torch.where(g['angle_max'] == torch.pi, g['angle_max'], g['angle_max'] - padding['angle'])
    assert (g['output_bond_min'] < g['output_bond_max']).all() and (low < high).all()
    g['output_cos_min'], g['output_cos_max'] = torch.cos(high), torch.cos(low)
    g['output_chiral_min'] = .1 + padding['volume'] / g['tetrahedron_scale']
    # Restricted distances use the same writer error bound as covalent distances.
    g['output_locked_min'] = g['locked_min'] + pair_error
    g['output_locked_max'] = g['locked_max'] - pair_error
    assert (g['output_locked_min'] < g['output_locked_max']).all()
    slots = {tuple(sorted(pair)): i for i, pair in enumerate(g['bonds'].tolist())}
    legs = torch.tensor([[slots[tuple(sorted((c, a)))] for a in near]
                         for c, *near in g['planar'].tolist()], device=device, dtype=torch.long).reshape(-1, 3)
    lengths = g['bond_max'][legs]
    planar_error = (pair_error*(lengths[:, 0]*lengths[:, 1]+lengths[:, 0]*lengths[:, 2]+lengths[:, 1]*lengths[:, 2])
                    + pair_error**2*lengths.sum(-1)+pair_error**3)/g['planar_scale']
    g['output_planar_min'] = g['planar_min']+planar_error
    g['output_planar_max'] = g['planar_max']-planar_error
    assert (g['output_planar_min'] < g['output_planar_max']).all()
    g['encoding_axis_error'] = float(axis_error)
    return g


def values_jacobian(x, g):
    """Return one scalar per constraint, bounds, dense Jacobian and atom membership."""
    batch, atoms, _ = x.shape
    blocks = []
    for pairs, low, high, scale in (
            (g['bonds'], g['output_bond_min'], g['output_bond_max'], g['bond_scale']),
            (g['locked_pairs'], g['output_locked_min'], g['output_locked_max'], torch.ones_like(g['locked_min']))):
        vector = x[:, pairs[:, 0]] - x[:, pairs[:, 1]]
        length = vector.norm(dim=-1).clamp_min(1e-12)
        direction = vector / length[..., None] / scale[None, :, None]
        blocks.append((length / scale, low / scale, high / scale,
                       torch.stack((direction, -direction), -2), pairs))
    a, b, c = g['angles'].T
    u, v = x[:, a] - x[:, b], x[:, c] - x[:, b]
    lu, lv = u.norm(dim=-1).clamp_min(1e-12), v.norm(dim=-1).clamp_min(1e-12)
    cosine = (u*v).sum(-1)/(lu*lv)
    du = v/(lu*lv)[..., None]-cosine[..., None]*u/lu[..., None].square()
    dv = u/(lu*lv)[..., None]-cosine[..., None]*v/lv[..., None].square()
    blocks.append((cosine, g['output_cos_min'], g['output_cos_max'],
                   torch.stack((du, -du-dv, dv), -2), g['angles']))
    for indices, scale, low, high, sign in (
            (g['tetrahedra'], g['tetrahedron_scale'], g['output_chiral_min'],
             torch.full_like(g['tetrahedron_scale'], float('inf')), g['tetrahedron_sign']),
            (g['planar'], g['planar_scale'], g['output_planar_min'], g['output_planar_max'], torch.ones_like(g['planar_scale']))):
        c, a, b, d = indices.T
        u, v, w = x[:, a]-x[:, c], x[:, b]-x[:, c], x[:, d]-x[:, c]
        value = (u*torch.cross(v, w, dim=-1)).sum(-1)*sign/scale
        ga, gb, gd = torch.cross(v, w, dim=-1), torch.cross(w, u, dim=-1), torch.cross(u, v, dim=-1)
        grad = torch.stack((-ga-gb-gd, ga, gb, gd), -2)*(sign/scale)[None, :, None, None]
        blocks.append((value, low, high, grad, indices))
    values, lows, highs, jacobians, members = [], [], [], [], []
    for value, low, high, grad, indices in blocks:
        matrix = x.new_zeros((batch, len(indices), atoms, 3))
        matrix.scatter_add_(2, indices[None, :, :, None].expand(batch, -1, -1, 3), grad)
        values.append(value); lows.append(low); highs.append(high); jacobians.append(matrix.flatten(-2))
        member = torch.zeros((len(indices), atoms), dtype=torch.bool, device=x.device)
        member.scatter_(1, indices, True); members.append(member)
    return torch.cat(values, 1), torch.cat(lows), torch.cat(highs), torch.cat(jacobians, 1), torch.cat(members)


def constraint_residual(value, low, high):
    return value - torch.maximum(low, torch.minimum(high, value))
