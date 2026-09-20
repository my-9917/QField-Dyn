"""Correct each final-coordinate frame with the same constrained objective."""
import time
import numpy as np
import torch
from covalent_output_constraints import output_constraints
from local_covalent_solve import solve_local


@torch.no_grad()
def solve_output_path(path, record, context, initial_coordinates=None):
    """Solve in physical frame order using the last accepted molecular state."""
    began = time.perf_counter()
    original = torch.as_tensor(path, device=context['weights'].device, dtype=torch.float64)
    centres = (original*context['weights'][:, None]).sum(-2)
    centre = centres[centres.abs().amax(-1).argmax()].cpu().numpy()
    constraints = output_constraints(context, record, centre)
    reference = original.cpu().numpy()
    final = reference.copy()
    passed = np.zeros(len(final), dtype=bool)
    records = []
    initial = np.asarray(record['inputs']['X_obs'])[-1] if initial_coordinates is None else initial_coordinates
    for frame in range(len(final)):
        previous = final[frame-1]-reference[frame-1] if frame else np.zeros_like(final[frame])
        warm = final[frame-1] if frame else initial
        candidate, detail = solve_local(warm, reference[frame], previous, constraints)
        records.append(dict(frame=frame, **detail))
        if not detail['passed']:
            break
        final[frame] = candidate
        passed[frame] = True
    return final, dict(completed=True, passed=bool(passed.all()),
        failed_frames=np.flatnonzero(~passed).tolist(), frames=len(final),
        local_solves=records, seconds=time.perf_counter()-began,
        objective='per-atom squared correction with previous-frame correction coefficient 0.5',
        encoding_axis_error=constraints['encoding_axis_error'])
