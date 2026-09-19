"""Stage1 v4 path scores in Angstrom with explicit heavy-atom masks."""
import torch


def energy_score(samples, reference, heavy_mask):
    """Return one score per condition: [B,M,H,N,3], [B,H,N,3], [B,N]."""
    samples, reference = samples.double(), reference.double()
    batch, members, horizon, _, _ = samples.shape
    assert members >= 2
    weights = heavy_mask[:, None, None, :, None].double()
    scale = (horizon*heavy_mask.sum(-1)).double().sqrt()
    x = (samples*weights).reshape(batch, members, -1)/scale[:, None, None]
    y = (reference*weights[:, 0]).reshape(batch, 1, -1)/scale[:, None, None]
    target_term = torch.linalg.vector_norm(x-y, dim=-1).mean(-1)
    pair_term = torch.cdist(x, x, compute_mode='donot_use_mm_for_euclid_dist').sum((-1, -2))/(2*members*(members-1))
    return target_term-pair_term


def displacement_variogram(samples, reference, heavy_mask, lags=(1, 2, 4, 8), power=.5):
    """Equal weight for every valid atom/time pair across the specified lags."""
    samples, reference = samples.double(), reference.double()
    total = torch.zeros(samples.shape[0], dtype=torch.float64, device=samples.device)
    count = 0
    for lag in lags:
        observed = torch.linalg.vector_norm(reference[:, lag:]-reference[:, :-lag], dim=-1).pow(power)
        predicted = torch.linalg.vector_norm(samples[:, :, lag:]-samples[:, :, :-lag], dim=-1).pow(power).mean(1)
        total += ((observed-predicted).square()*heavy_mask[:, None]).sum((-1, -2))
        count += reference.shape[1]-lag
    return total/(heavy_mask.sum(-1)*count)


def system_mean(values, system_ids):
    """Equal weighting of systems after averaging their conditions."""
    return torch.stack([values[system_ids == identifier].mean() for identifier in torch.unique(system_ids)]).mean()
