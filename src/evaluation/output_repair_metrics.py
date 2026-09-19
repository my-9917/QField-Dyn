"""Paired full-path diagnostics for a saved final-coordinate repair."""
import numpy as np
import joint_observables as frozen
from molecular_dynamics_scores import torsion_phase
from projection_encoding import encode_coordinates
from semiflexible_scores import observables, physics_scores
from trajectory_metrics import trajectory_metrics


def compare_repair(before, after, record, weights):
    heavy = np.asarray(record['inputs']['ligand_graph']['atomic_numbers']) > 1
    indices = frozen.graph_torsions(record['inputs']['ligand_graph'])
    comparisons, phases, validity = {}, {}, {}
    for label, coordinates in [('before', before), ('after', after)]:
        encoded, encoded_record = encode_coordinates(np.asarray(coordinates)[None], record)
        truth = np.asarray(encoded_record['X_future'])
        last = np.asarray(encoded_record['inputs']['X_obs'])[-1, heavy]
        values = trajectory_metrics(encoded[0][:, heavy], truth[:, heavy],
                                    last, record['meta']['dt_ps'])
        predicted, actual = observables(encoded, encoded_record), observables(truth[None], encoded_record)
        phase, valid = torsion_phase(encoded, indices)
        target, target_valid = torsion_phase(truth, indices)
        comparisons[label] = dict(trajectory=values,
            contacts=frozen.contact_scores(predicted, actual, encoded_record['inputs']),
            torsions=frozen.periodic_scores(phase, valid, target, target_valid, record['meta']['dt_ps']),
            physics=physics_scores(encoded, encoded_record['geometry'])[0])
        phases[label], validity[label] = phase, valid
    delta = np.asarray(after) - np.asarray(before)
    valid = validity['before'] & validity['after']
    angular_change = np.abs(np.angle(phases['after']*phases['before'].conjugate()))
    correction = dict(rms_angstrom=float(np.sqrt(np.square(delta).sum(-1).mean())),
        step_rms_angstrom=float(np.sqrt(np.square(np.diff(delta, axis=0)).sum(-1).mean())),
        maximum_COM_change_angstrom=float(np.linalg.norm((delta*weights[None, :, None]).sum(-2), axis=-1).max()),
        torsion_mean_absolute_change_degrees=float(np.rad2deg(angular_change[valid]).mean()) if valid.any() else None,
        torsion_valid_comparisons=int(valid.sum()), torsion_total_comparisons=int(valid.size))
    return dict(comparisons=comparisons, correction=correction,
                scope='single saved path; both sides use actual XTC encoding; diagnostic comparison')
