"""Restore an admissible molecular state before an autoregressive T4 update."""
import math
import numpy as np
import torch
from covalent_output_solver import solve_output_path
from projection_encoding import encode_coordinates
from physical_quality import measurements


class FeedbackSolveFailure(RuntimeError):
    """Keep the actual failed numerical problem available to the delivery caller."""
    def __init__(self,path,record,geometry,repaired,detail,initial_coordinates):
        super().__init__(f"T4 covalent solve failed at frames {detail['failed_frames']}")
        self.payload=dict(path=path.cpu(),record=record,geometry=geometry,result=repaired,detail=detail,
                          initial_coordinates=initial_coordinates)


def feedback_geometry(geometry, calibration):
    assert calibration['partition'] == 'train' and calibration['completed']
    limits = calibration['thresholds']; g = dict(geometry)
    distance_margin = limits['bond_relative']*g['bond_scale']
    angle_margin = math.radians(limits['angle_degrees'])
    g['bond_min'] = (g['bond_min']-distance_margin).clamp_min(0)
    g['bond_max'] = g['bond_max']+distance_margin
    g['angle_min'] = (g['angle_min']-angle_margin).clamp_min(0)
    g['angle_max'] = (g['angle_max']+angle_margin).clamp_max(math.pi)
    return g


@torch.no_grad()
def restore_feedback(path, record, geometry, calibration, initial_coordinates=None):
    repaired, detail = solve_output_path(path.cpu().numpy(), record, geometry,initial_coordinates=initial_coordinates)
    if not detail['passed']:
        raise FeedbackSolveFailure(path,record,geometry,repaired,detail,initial_coordinates)
    decoded, encoded_record = encode_coordinates(repaired[None], record)
    assessed = measurements(decoded, encoded_record['geometry'])
    for key in ('bond_relative', 'angle_degrees'):
        assert (assessed['frame_max'][key] <= calibration['thresholds'][key]+1e-7).all(), key
    delta = repaired-path.cpu().numpy()
    detail.update(correction_rms_angstrom=float(np.sqrt(np.square(delta).sum(-1).mean())),
                  maximum_atom_correction_angstrom=float(np.linalg.norm(delta, axis=-1).max()),
                  definition='train-calibrated covalent bounds; mass centre retained; no pocket confinement')
    return torch.as_tensor(repaired, device=path.device), decoded[0], detail
