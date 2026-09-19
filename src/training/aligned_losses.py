"""Differentiable generated-path physics and the frozen fair feature Energy Score."""
import numpy as np
import torch
from ligand_geometry import geometry_terms


def feature_context(record, device):
    inputs = record['inputs']
    heavy = np.flatnonzero(np.asarray(inputs['ligand_graph']['atomic_numbers']) > 1)
    protein = inputs['protein_topology']
    ph = np.asarray(protein['atomic_numbers']) > 1
    residues, slots = np.unique(np.asarray(protein['residue_indices'])[ph], return_inverse=True)
    environment = torch.as_tensor(np.asarray(inputs['P0'])[ph], dtype=torch.float64, device=device)
    last = torch.as_tensor(np.asarray(inputs['X_obs'])[-1, heavy], dtype=torch.float64, device=device)
    initial = torch.cdist(last, environment, compute_mode='donot_use_mm_for_euclid_dist').amin(0)
    selected = np.unique(slots[(initial <= 6.).cpu().numpy()])
    keep = np.isin(slots, selected)
    return dict(heavy=torch.as_tensor(heavy, device=device), last=last,
        environment=environment[keep], groups=[torch.as_tensor(np.flatnonzero(slots[keep] == s), device=device)
                                              for s in selected])


def trajectory_features(paths, context):
    x = paths.double()[..., context['heavy'], :]
    displacement = torch.linalg.vector_norm((x-context['last']).flatten(-2), dim=-1)/(x.shape[-2]**.5)
    gyration = torch.linalg.vector_norm((x-x.mean(-2, keepdim=True)).flatten(-2), dim=-1)/(x.shape[-2]**.5)
    features = [displacement, gyration]
    if context['groups']:
        distances = torch.cdist(x, context['environment'], compute_mode='donot_use_mm_for_euclid_dist').amin(-2)
        residues = torch.stack([distances[..., indices].amin(-1) for indices in context['groups']], -1)
        features.append((1/(1+(residues/4.5)**6)).mean(-1))
    return torch.stack(features, -1)


def feature_energy_score(samples, truth, scales):
    """Fair M>=2 iid estimate, normalized over all times and defined features."""
    members = len(samples)
    assert members >= 2 and samples.shape[1:] == truth.shape
    x = (samples/scales).reshape(members, -1)
    y = (truth/scales).flatten()
    target = torch.linalg.vector_norm(x-y, dim=-1).mean()
    pair = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist').sum()/(2*members*(members-1))
    return (target-pair)/(y.numel()**.5)


def coordinate_energy_score(paths, target, heavy, scale):
    """Fair full-path coordinate ES in the fixed protein frame, in atom-RMS units."""
    x = paths.double()[..., heavy, :]
    y = target.double()[..., heavy, :]
    return feature_energy_score(x, y, scale)*(3.**.5)


def rollout_objectives(paths, target, geometry, context, scales, environment_epsilon=0., coordinate_scale=None):
    physics = geometry_terms(paths, geometry, split_overlap=True)
    with torch.no_grad():
        reference = geometry_terms(target, geometry, split_overlap=True)['environment_overlap']
        truth_features = trajectory_features(target, context)
    physics['environment_overlap'] = (physics['environment_overlap']-reference-environment_epsilon).clamp_min(0)
    with torch.no_grad():
        features = trajectory_features(paths, context)
        feature_es = feature_energy_score(features, truth_features, scales[:features.shape[-1]])
    result = dict(rollout_physics=sum(value.mean() for value in physics.values()), feature_es=feature_es)
    if coordinate_scale is not None:
        result['coordinate_es'] = coordinate_energy_score(paths, target, context['heavy'], coordinate_scale)
    return result
