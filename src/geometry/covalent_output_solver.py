"""Adaptive final-coordinate feasibility with explicit residual-based completion."""
import time
import numpy as np
import torch
from covalent_output_constraints import values_jacobian, constraint_residual, output_constraints
from local_covalent_solve import solve_local


@torch.no_grad()
def adaptive_covalent(x, g, max_iterations=128):
    x = x.double().clone()
    centre = (x*g['weights'][:, None]).sum(-2, keepdim=True)
    damping = x.new_full((len(x),), 1e-3)
    iterations = torch.zeros(len(x), device=x.device, dtype=torch.long)
    remaining = torch.arange(len(x), device=x.device)
    eye = torch.eye(x.shape[-2]*3, device=x.device, dtype=x.dtype)
    for step in range(max_iterations):
        if not len(remaining): break
        current = x[remaining]
        value, low, high, jac, _ = values_jacobian(current, g)
        residual = constraint_residual(value, low, high)
        done = residual.abs().amax(-1) <= 1e-9
        remaining = remaining[~done]
        if not len(remaining): break
        current, residual, jac = current[~done], residual[~done], jac[~done]
        jac = jac*(residual != 0)[..., None]
        jt = jac.transpose(-1, -2)
        delta = torch.linalg.solve(jt@jac+damping[remaining, None, None]*eye,
                                   -(jt@residual[..., None])).reshape_as(current)
        delta *= (.25/delta.norm(dim=-1).amax(-1).clamp_min(.25))[:, None, None]
        trial = current+delta
        trial = trial-(trial*g['weights'][:, None]).sum(-2, keepdim=True)+centre[remaining]
        new_value, low, high, _, _ = values_jacobian(trial, g)
        new_residual = constraint_residual(new_value, low, high)
        accepted = new_residual.square().sum(-1) < residual.square().sum(-1)
        x[remaining[accepted]] = trial[accepted]
        damping[remaining] = torch.where(accepted, damping[remaining]*.3, damping[remaining]*10).clamp(1e-10, 1e12)
        iterations[remaining] = step+1
        stalled = (damping[remaining] >= 1e12) | ((delta.abs().amax((-1, -2)) < 1e-12) & ~accepted)
        remaining = remaining[~stalled]
    value, low, high, _, _ = values_jacobian(x, g)
    residual = constraint_residual(value, low, high).abs().amax(-1)
    return x, dict(passed=(residual <= 1e-9).cpu().numpy(), iterations=iterations.cpu().numpy(), residual=residual.cpu().numpy())


@torch.no_grad()
def solve_output_path(path, record, context, local_hard=True, chunk_frames=8, initial_coordinates=None):
    """One independent path; accepted repairs follow physical frame order."""
    began = time.perf_counter(); original = torch.as_tensor(path, device=context['weights'].device, dtype=torch.float64)
    centres = (original*context['weights'][:, None]).sum(-2)
    centre = centres[centres.abs().amax(-1).argmax()].cpu().numpy()
    g = output_constraints(context, record, centre)
    outputs, details = [], []
    for chunk in original.split(chunk_frames):
        result, detail = adaptive_covalent(chunk, g)
        outputs.append(result); details.append(detail)
    final = torch.cat(outputs).cpu().numpy()
    passed = np.concatenate([d['passed'] for d in details])
    records = []
    reference = original.cpu().numpy()
    if local_hard:
        for frame in np.flatnonzero(~passed):
            previous = final[frame-1]-reference[frame-1] if frame else np.zeros_like(final[frame])
            warm = final[frame-1] if frame else (
                np.asarray(record['inputs']['X_obs'])[-1] if initial_coordinates is None else initial_coordinates)
            candidate, detail = solve_local(warm, reference[frame], previous, g)
            records.append(dict(frame=int(frame), **detail))
            if detail['passed']:
                final[frame] = candidate; passed[frame] = True
            else:
                # A failed frame terminates this path before its coordinates
                # can become the next frame's warm start or temporal target.
                break
    return final, dict(completed=True, passed=bool(passed.all()), failed_frames=np.flatnonzero(~passed).tolist(),
        adaptive_passed_frames=int(sum(d['passed'].sum() for d in details)), frames=len(final),
        adaptive_iterations=np.concatenate([d['iterations'] for d in details]).tolist(),
        local_solves=records, seconds=time.perf_counter()-began, encoding_axis_error=g['encoding_axis_error'])
