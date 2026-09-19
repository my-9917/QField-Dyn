"""Observed-geometry candidates with explicit pair constraints and fixed mass centre."""
import numpy as np
import torch
from rdkit.Chem import GetPeriodicTable
from scipy.linalg import null_space
from scipy.optimize import minimize
from geometry_constraints import geometric_values, connected_encoding_padding
from observable_constraints import ObservableConstraints
from semiflexible_scores import observables, physics_scores


def solve_candidate(raw, initial, record, config, preserve_pocket, preserve_collisions=True):
    g = {k: v.contiguous() if torch.is_tensor(v) else v for k, v in record['geometry'].items()}
    inputs = record['inputs']; n = len(raw)
    mass = np.array([GetPeriodicTable().GetAtomicWeight(int(z)) for z in inputs['ligand_graph']['atomic_numbers']])
    mass /= mass.sum(); centre = mass@raw; basis = null_space(mass[None])
    padding, axis_error, pair_error = connected_encoding_padding(g, record, centre, config)
    ob = ObservableConstraints(raw, inputs, g, pair_error+2*config['constraint_tolerance'])
    if not preserve_collisions:
        # The prefix selector enforces the collision-count budget. This candidate
        # preserves pocket contacts while the objective penalizes collision depth.
        ob.self_floor.fill(0.)
        ob.cross_floor.fill(0.)
    pockets = observables(raw[None, None], record)['pocket_slots'] if preserve_pocket else np.array([], dtype=int)
    contacts = [int(k) for k in pockets if ob.contact_sign[k] < 0]
    # A noncontact is a conjunction of all pair lower bounds; a contact is an existence condition.
    for k in pockets:
        if ob.contact_sign[k] > 0:
            group = ob.residue_groups[k]
            ob.cross_floor[:, group] = np.maximum(ob.cross_floor[:, group], 4.5+ob.padding)
    N, ct, ref = torch.from_numpy(basis), torch.from_numpy(centre), torch.from_numpy(raw)
    pairs, heavy, env = torch.as_tensor(ob.pairs), torch.as_tensor(ob.heavy), torch.as_tensor(ob.environment)
    sf, cf = torch.from_numpy(ob.self_floor), torch.from_numpy(ob.cross_floor)
    tolerance = config['constraint_tolerance']

    def coordinates(q):
        return basis@q.reshape(n-1, 3)+centre

    def residuals(x):
        values = [geometric_values(x.ravel(), g, padding, config),
            torch.linalg.vector_norm(x[pairs[:, 0]]-x[pairs[:, 1]], dim=-1)-sf]
        cross = torch.cdist(x[heavy], env, compute_mode='donot_use_mm_for_euclid_dist')
        values.append((cross-cf).ravel())
        for k in contacts:
            values.append((4.5-ob.padding-cross[:, ob.residue_groups[k]].min()).reshape(1))
        return values

    def feasibility(q):
        qt = torch.tensor(q.reshape(n-1, 3), requires_grad=True)
        value = sum(.5*v.clamp_max(0).square().sum() for v in residuals(N@qt+ct))
        grad, = torch.autograd.grad(value, qt)
        return float(value.detach()), grad.numpy().ravel()

    def objective(q):
        qt = torch.tensor(q.reshape(n-1, 3), requires_grad=True); x = N@qt+ct
        sd = torch.linalg.vector_norm(x[pairs[:, 0]]-x[pairs[:, 1]], dim=-1)
        cd = torch.cdist(x[heavy], env, compute_mode='donot_use_mm_for_euclid_dist')
        overlap = ((g['self_limits']-sd).clamp_min(0)/g['self_limits']).square().sum()
        overlap = overlap+((g['cross_limits']-cd).clamp_min(0)/g['cross_limits']).square().sum()
        value = .5*(x-ref).square().sum()+.5*config['overlap_weight']*overlap
        grad, = torch.autograd.grad(value, qt)
        return float(value.detach()), grad.numpy().ravel()

    def excess(q):
        return float(max(0., *[-v.min().item() for v in residuals(torch.from_numpy(coordinates(q))) if v.numel()]))

    q = (basis.T@(initial-centre)).ravel()
    phase = dict(required=excess(q) > tolerance)
    if phase['required']:
        result = minimize(feasibility, q, jac=True, method='L-BFGS-B', options=dict(
            maxiter=np.iinfo(np.int32).max, maxfun=np.iinfo(np.int32).max, maxls=60, maxcor=40,
            ftol=1e-16, gtol=1e-11))
        q = result.x
        phase.update(status=int(result.status), message=str(result.message), iterations=int(result.nit),
            evaluations=int(result.nfev), objective=float(result.fun), residual_max=excess(q))
    ds, dc = ob.pair_gaps(coordinates(q))
    active_s = set(np.flatnonzero(ds < .5).tolist())
    active_c = set(map(tuple, np.argwhere(dc < .5)))
    best, best_value, rounds = None, float('inf'), []

    def consider(current):
        nonlocal best, best_value
        if excess(current) <= tolerance:
            value = objective(current)[0]
            if value < best_value:
                best, best_value = current.copy(), value

    consider(q)
    while True:
        si = np.array(sorted(active_s), dtype=int)
        cp = np.array(sorted(active_c), dtype=int).reshape(-1, 2)
        cached = [None, None]

        def values_jac(current):
            if cached[0] is None or not np.array_equal(current, cached[0]):
                x = coordinates(current)
                flat = torch.from_numpy(x.ravel())
                gv = geometric_values(flat, g, padding, config).numpy()
                gj = torch.autograd.functional.jacobian(lambda a: geometric_values(a, g, padding, config),
                    flat, vectorize=True).numpy()
                ov, oj = ob.explicit_values_jacobian(x, si, cp, contacts)
                val = np.concatenate((gv, ov))
                jac = np.concatenate((gj, oj)).reshape(len(val), n, 3)
                cached[:] = [current.copy(), (val, np.einsum('mnc,nk->mkc', jac, basis).reshape(len(val), -1))]
            return cached[1]

        result = minimize(objective, q, jac=True, method='SLSQP', callback=consider,
            constraints=[dict(type='ineq', fun=lambda a: values_jac(a)[0], jac=lambda a: values_jac(a)[1])],
            options=dict(maxiter=np.iinfo(np.int32).max, ftol=1e-10))
        q = result.x; consider(q)
        ds, dc = ob.pair_gaps(coordinates(q))
        rounds.append(dict(status=int(result.status), message=str(result.message), iterations=int(result.nit),
            constraints=len(values_jac(q)[0]), residual_max=excess(q)))
        added_s = set(np.flatnonzero(ds < 1e-4).tolist())-active_s
        added_c = set(map(tuple, np.argwhere(dc < 1e-4)))-active_c
        if best is not None or not (added_s or added_c):
            break
        active_s |= added_s; active_c |= added_c
    kind = 'collision_pocket' if preserve_pocket and preserve_collisions else ('pocket' if preserve_pocket else 'collision')
    evidence = dict(kind=kind,
        feasible_point_found=best is not None, phase=phase, exchanges=rounds,
        coordinate_error_bound_angstrom=float(axis_error), pair_distance_error_bound_angstrom=float(pair_error))
    if best is None:
        evidence.update(terminal_coordinates=coordinates(q), global_feasibility='unresolved')
        return None, evidence
    x = coordinates(best)
    evidence.update(success=rounds[-1]['status'] == 0, constraint_excess_max=excess(best),
        mass_centre_error_angstrom=float(np.linalg.norm(mass@x-centre)), returned_objective=best_value)
    return x, evidence


def frame_candidates(raw, record, config, base=None, base_evidence=None, previous_pool=None):
    from terminal_geometry import project_frames
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    if base is None:
        result, evidence = project_frames(raw[None, None], record, config['base_projection'])
        base, base_evidence = result[0, 0], evidence[0]
    base = np.ascontiguousarray(base, dtype=np.float64)
    if previous_pool is None:
        coordinates = [base]; evidence = [dict(base_evidence, kind='original_v2', feasible_point_found=True)]
        attempts = []
    else:
        coordinates = list(previous_pool['coordinates'])
        evidence = list(previous_pool['evidence'])
        attempts = list(previous_pool['attempts'])
    completed = {row['kind'] for row in evidence+attempts}
    obs = observables(np.stack((raw, base))[:, None], record)
    physics = physics_scores(np.stack((raw, base))[:, None], record['geometry'])[1]
    counts = physics['ligand_self_overlap_violations']+physics['ligand_environment_overlap_violations']
    changed = np.any((obs['distances'][0] < 4.5) != (obs['distances'][1] < 4.5))
    if changed or counts[1, 0] > counts[0, 0]:
        constraints = {'collision': (False, True), 'collision_pocket': (True, True), 'pocket': (True, False)}
        for kind in config['candidate_scopes']:
            if kind in completed:
                continue
            point, detail = solve_candidate(raw, base, record, config, *constraints[kind])
            attempts.append(detail)
            if point is not None:
                coordinates.append(point); evidence.append(detail)
    return dict(coordinates=np.stack(coordinates), evidence=evidence, attempts=attempts)
