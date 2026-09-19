"""Paired raw/final metrics and acceptance, separate from solving and artifact I/O."""
import numpy as np
from rdkit import Chem
from semiflexible_scores import score_paths, observables
from projection_encoding import encode_coordinates
from prefix_selection import brier_bound



def evaluate_projection(raw, corrected, record, scales, evidence, rules):
    pre_encoding, _ = score_paths(corrected, record, scales)
    decoded, encoded_record = encode_coordinates(corrected, record)
    projected, detail = score_paths(decoded, encoded_record, scales)
    raw_metrics, _ = score_paths(raw, record, scales)
    centre_error = max(row['mass_centre_error_angstrom'] for row in evidence)
    masses = np.array([Chem.GetPeriodicTable().GetAtomicWeight(int(z))
                       for z in record['inputs']['ligand_graph']['atomic_numbers']])
    fractions = masses/masses.sum()
    encoded_centre_error = np.linalg.norm(np.einsum('mhni,n->mhi',decoded-raw,fractions),axis=-1)
    checks = projection_acceptance(raw_metrics, projected, centre_error, rules)
    heavy = np.asarray(record['inputs']['ligand_graph']['atomic_numbers']) > 1
    displacement = np.sqrt(np.square((corrected-raw)[:, :, heavy]).sum(-1).mean(-1))
    velocity_correction = np.diff((decoded-raw)[:, :, heavy], axis=1)/(record['meta']['dt_ps']/1000.)
    before, after = observables(raw, record), observables(decoded, encoded_record)
    worst = brier_bound((before['distances'] < 4.5).mean(0), (after['distances'] < 4.5).mean(0))
    probability_bounds = {name: float(worst[:, slots].mean()) if len(slots) else None
        for name, slots in [('all_residues', np.arange(worst.shape[-1])), ('observed_pocket', before['pocket_slots'])]}
    summary = dict(raw=raw_metrics, projected=projected, pre_encoding=pre_encoding,
        acceptance_checks=checks, projection_adoption_passed=all(v for v in checks.values() if v is not None),
        final_coordinate_layer='full_atom_xtc_decoded', ensemble_brier_worst_increase=probability_bounds,
        ensemble_acceptance='actual ensemble Brier against truth; per-path budget recorded independently',
        projection_velocity_change_rms_angstrom_per_ns=float(np.sqrt(np.square(velocity_correction).sum(-1).mean())),
        mass_centre_acceptance_layer='pre_encoding constrained solve; original 1e-9 Angstrom rule',
        mass_centre_error_max_angstrom=centre_error,
        encoded_mass_centre_error_max_angstrom=float(encoded_centre_error.max()),
        projection_displacement_angstrom=dict(mean=float(displacement.mean()),p95=float(np.quantile(displacement,.95)),
            maximum=float(displacement.max()),definition='pre-encoding heavy-atom RMS per frame'))
    return decoded, detail, summary


def projection_acceptance(raw, projected, centre_error, rule):
    before, after = raw['Phys'], projected['Phys']
    checks = {}
    for kind in ['bond', 'angle']:
        checks[kind+'_excess_reduction'] = after[kind+'_mean_squared_excess'] <= rule[kind+'_squared_excess_ratio_max']*before[kind+'_mean_squared_excess']
        checks[kind+'_count_reduction'] = after[kind+'_violations'] <= rule[kind+'_violation_ratio_max']*before[kind+'_violations']
    checks['rmsd_preserved'] = projected['Geo']['mean_rmsd_angstrom']-raw['Geo']['mean_rmsd_angstrom'] <= rule['rmsd_increase_max_angstrom']
    checks['rmsf_error_preserved'] = projected['Dyn']['rmsf_mae_angstrom']-raw['Dyn']['rmsf_mae_angstrom'] <= rule['rmsf_mae_increase_max_angstrom']
    for name in ['observed_pocket', 'all_residues']:
        a, b = raw['Dyn']['contacts'][name]['brier'], projected['Dyn']['contacts'][name]['brier']
        checks[name+'_contact_preserved'] = None if a is None else b-a <= rule['contact_brier_increase_max']
    checks['mass_centre_preserved'] = centre_error < rule['mass_centre_error_max_angstrom']
    checks['configuration_preserved'] = after['local_configuration_flips'] <= before['local_configuration_flips']
    checks['overlap_preserved'] = sum(after[key] for key in ['ligand_self_overlap_violations', 'ligand_environment_overlap_violations']) <= sum(before[key] for key in ['ligand_self_overlap_violations', 'ligand_environment_overlap_violations'])
    return checks
