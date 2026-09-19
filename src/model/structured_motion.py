"""Finite rigid source motion and consistently centred CFM errors."""
import torch


def internal_component(value, reference):
    """Orthogonal projection away from translation and infinitesimal rotation."""
    r = reference-reference.mean(-2, keepdim=True)
    centred = value-value.mean(-2, keepdim=True)
    inertia = r.square().sum((-1, -2))[..., None, None]*torch.eye(3, device=r.device, dtype=r.dtype)
    inertia = inertia-r.transpose(-1, -2)@r
    torque = torch.cross(r, centred, dim=-1).sum(-2)
    omega = (torch.linalg.pinv(inertia, hermitian=True, rtol=1e-6)@torque[..., None]).squeeze(-1)
    return centred-torch.cross(omega[..., None, :].expand_as(r), r, dim=-1)


def rigid_source(last, paths, horizon, generator, scales):
    r = last-last.mean(0)
    translation = torch.randn(paths, horizon, 1, 3, generator=generator,
        device=last.device, dtype=last.dtype)*scales['translation']
    omega = torch.randn(paths, horizon, 3, generator=generator,
        device=last.device, dtype=last.dtype)*scales['rotation']
    a, b, c = omega.unbind(-1)
    zero = torch.zeros_like(a)
    skew = torch.stack((zero, -c, b, c, zero, -a, -b, a, zero), -1).reshape(paths, horizon, 3, 3)
    rotation = torch.linalg.matrix_exp(skew)
    rotated = r@rotation.transpose(-1, -2)
    eta = torch.randn(paths, horizon, len(last), 3, generator=generator,
        device=last.device, dtype=last.dtype)*scales['internal']
    return translation+rotated-r+internal_component(eta, rotated)


def component_loss(error, scales):
    translation = error.mean(-2, keepdim=True)
    centred = error-translation
    return (translation/scales['translation']).square().mean()+(centred/scales['centred']).square().mean()
