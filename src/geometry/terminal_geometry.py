"""Observed ligand constraints and fixed-environment contacts with exact mass-centre preservation."""
import numpy as np
from rdkit.Chem import GetPeriodicTable
from scipy.optimize import minimize, least_squares
from scipy.spatial import cKDTree
import torch
from trajectory_delivery import XTC_PRECISION, coordinate_encoding_error_bound
from geometry_constraints import geometric_values


from projection_errors import ProjectionFeasibilityError


def project_frames(paths, record, config):
    geometry = {key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in record['geometry'].items()}
    inputs = record['inputs']
    assert config['xtc_precision'] == XTC_PRECISION
    xyz = np.ascontiguousarray(paths, dtype=np.float64)
    mass = np.array([GetPeriodicTable().GetAtomicWeight(int(z)) for z in inputs['ligand_graph']['atomic_numbers']])
    mass /= mass.sum()
    mass_jac = np.zeros((3, xyz.shape[-2], 3))
    for axis in range(3): mass_jac[axis, :, axis] = mass
    mass_jac = mass_jac.reshape(3, -1)
    environment = np.ascontiguousarray(geometry['environment'])
    tree = cKDTree(environment)
    heavy, limits = np.asarray(geometry['heavy']), np.asarray(geometry['cross_limits'])
    pairs, self_limits = geometry['self_pairs'], geometry['self_limits']
    bond_slots = {tuple(sorted(pair)): i for i, pair in enumerate(geometry['bonds'].tolist())}
    angle_legs = torch.tensor([[bond_slots[tuple(sorted((a, b)))], bond_slots[tuple(sorted((c, b)))]]
        for a, b, c in geometry['angles'].tolist()], dtype=torch.long).reshape(-1, 2)
    tetra_legs = torch.tensor([[bond_slots[tuple(sorted((centre, atom)))] for atom in (a, b, c)]
        for centre, a, b, c in geometry['tetrahedra'].tolist()], dtype=torch.long).reshape(-1, 3)
    origin = np.asarray(record['transform']['origin'])
    fixed_extent = np.abs(np.concatenate((inputs['P0'], inputs['I0']))+origin).max()
    preserve_observables = config.get('preserve_raw_observables', False)

    def constraints(flat):
        return geometric_values(flat, geometry, padding, config)

    def inequality(flat):
        return constraints(torch.from_numpy(flat)).numpy()

    def inequality_jac(flat):
        return torch.autograd.functional.jacobian(constraints, torch.from_numpy(flat), vectorize=True).numpy()

    corrected, evidence = xyz.copy(), []
    for member, frame in np.ndindex(xyz.shape[:2]):
        original = xyz[member, frame]
        centre = mass@original
        reference = torch.from_numpy(original)

        def objective(flat):
            current = flat.reshape(-1, 3)
            neighbours = tree.query_ball_point(current[heavy], r=float(limits.max()))
            a = np.repeat(np.arange(len(heavy)), [len(row) for row in neighbours])
            b = np.array([j for row in neighbours for j in row], dtype=int)
            x = torch.tensor(current, requires_grad=True)
            distance = torch.linalg.vector_norm(x[pairs[:, 0]]-x[pairs[:, 1]], dim=-1)
            overlap = ((self_limits-distance).clamp_min(0)/self_limits).square().sum()
            threshold = torch.from_numpy(limits[a, b])
            distance = torch.linalg.vector_norm(x[heavy[a]]-torch.from_numpy(environment[b]), dim=-1)
            overlap = overlap+((threshold-distance).clamp_min(0)/threshold).square().sum()
            value = .5*(x-reference).square().sum()+.5*config['overlap_weight']*overlap
            gradient, = torch.autograd.grad(value, x)
            return float(value.detach()), gradient.detach().numpy().ravel()

        if config['initialization'] == 'raw_prediction':
            assert preserve_observables
            # Contact membership and pair-distance floors are defined at the raw
            # prediction; begin in that contact component while repairing geometry.
            initial = original.copy()
        else:
            assert config['initialization'] == 'aligned_last_observation'
            # Canonical layout fixes reduction order before the SVD and solver.
            last = np.ascontiguousarray(inputs['X_obs'][-1], dtype=np.float64)
            centred = last-mass@last
            u, _, vt = np.linalg.svd((centred*mass[:, None]).T@(original-centre))
            initial = centred@(u@np.diag([1., 1., np.linalg.det(u@vt)])@vt)+centre
        assert config['output_rule'] == 'lowest_objective_feasible_iterate'
        best = initial.ravel().copy()
        initial_value = objective(best)[0]; best_value = initial_value
        # The returned objective is at most its feasible initialization. Its quadratic
        # displacement term bounds every accepted coordinate before encoding.
        coordinate_extent = max(fixed_extent, np.abs(original+origin).max()+np.sqrt(2*initial_value))
        if preserve_observables:
            # Fixed mass centre and connected, bounded bond lengths bound every
            # feasible atom, including solutions above the initialization objective.
            coordinate_extent = max(fixed_extent, np.abs(centre+origin).max()+float(geometry['bond_max'].sum()))
        axis_error = coordinate_encoding_error_bound(coordinate_extent)
        pair_error = 2*np.sqrt(3)*axis_error
        assert pair_error < float(geometry['bond_min'].min())
        angle_error = torch.asin(pair_error/geometry['bond_min'][angle_legs]).sum(-1)
        edge = geometry['bond_max'][tetra_legs]
        volume_error = (pair_error*(edge[:, 0]*edge[:, 1]+edge[:, 0]*edge[:, 2]+edge[:, 1]*edge[:, 2])
            +pair_error**2*edge.sum(-1)+pair_error**3)
        padding = dict(bond=pair_error+2*config['constraint_tolerance']*geometry['bond_scale'],
            angle=angle_error+2*config['constraint_tolerance']*geometry['angle_scale'], volume=volume_error)
        if not preserve_observables:
            assert inequality(initial.ravel()).min(initial=0.) >= -config['constraint_tolerance']
        additional = []
        if preserve_observables:
            from observable_constraints import ObservableConstraints
            observable = ObservableConstraints(original, inputs, geometry,
                pair_error+2*config['constraint_tolerance'])
            additional = [{'type': 'ineq', 'fun': observable.fun, 'jac': observable.jac}]
            best, best_value = None, float('inf')

        def consider_feasible(flat):
            nonlocal best, best_value
            feasible = inequality(flat).min(initial=0.) >= -config['constraint_tolerance']
            feasible = feasible and np.linalg.norm(mass@flat.reshape(-1, 3)-centre) < 1e-9
            if preserve_observables:
                feasible = feasible and observable.fun(flat).min(initial=0.) >= -config['constraint_tolerance']
            if feasible:
                value = objective(flat)[0]
                if value < best_value: best, best_value = flat.copy(), value

        if preserve_observables: consider_feasible(initial.ravel())
        feasibility = None
        if config.get('feasible_start', False) and best is None:
            # A least-squares feasibility problem admits inconsistent linearized
            # constraints. Mass centre is eliminated exactly throughout this phase.
            def centred_coordinates(flat):
                x = flat.reshape(-1, 3)
                return (x-mass@x+centre).ravel()

            def feasibility_values(flat):
                current = centred_coordinates(flat)
                return np.minimum(np.concatenate((inequality(current), observable.fun(current))), 0.)

            def feasibility_jac(flat):
                current = centred_coordinates(flat)
                values = np.concatenate((inequality(current), observable.fun(current)))
                jac = np.concatenate((inequality_jac(current), observable.jac(current)))
                jac = jac.reshape(len(values), -1, 3)
                jac = jac-jac.sum(1, keepdims=True)*mass[None, :, None]
                return (jac*(values < 0)[:, None, None]).reshape(len(values), -1)

            prepared = least_squares(feasibility_values, initial.ravel(), jac=feasibility_jac,
                method='trf', ftol=config['ftol'], xtol=1e-12, gtol=1e-12,
                max_nfev=np.inf if config['feasibility_max_evaluations'] is None else config['feasibility_max_evaluations'])
            initial = centred_coordinates(prepared.x).reshape(-1, 3)
            consider_feasible(initial.ravel())
            feasibility = dict(success=bool(prepared.success), status=int(prepared.status),
                evaluations=int(prepared.nfev), message=prepared.message,
                residual_max=float(np.abs(feasibility_values(prepared.x)).max(initial=0.)))
            if best is None:
                raise ProjectionFeasibilityError(dict(member=member, frame=frame,
                    success=False, status=int(prepared.status), iterations=int(prepared.nfev),
                    message='Joint feasibility stage: '+prepared.message, feasibility=feasibility,
                    terminal_coordinates=initial,
                    geometry_excess_max=float(np.maximum(-inequality(initial.ravel()), 0).max(initial=0.)),
                    observable_excess_max=float(np.maximum(-observable.fun(initial.ravel()), 0).max(initial=0.)),
                    scope='Feasibility optimization found no joint feasible point; global feasibility remains unresolved'))
        result = minimize(objective, initial.ravel(), jac=True, method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda a: mass@a.reshape(-1, 3)-centre, 'jac': lambda a: mass_jac},
                         {'type': 'ineq', 'fun': inequality, 'jac': inequality_jac}]+additional,
            callback=consider_feasible, options={'maxiter': config['iterations'], 'ftol': config['ftol']})
        consider_feasible(result.x)
        if best is None:
            raise ProjectionFeasibilityError(dict(member=member, frame=frame, success=bool(result.success),
                status=int(result.status), iterations=int(result.nit), message=result.message,
                terminal_coordinates=result.x.reshape(-1, 3),
                geometry_excess_max=float(np.maximum(-inequality(result.x), 0).max(initial=0.)),
                observable_excess_max=float(np.maximum(-observable.fun(result.x), 0).max(initial=0.)),
                scope='Local constrained solve found no jointly feasible iterate; global feasibility remains unresolved'))
        corrected[member, frame] = best.reshape(-1, 3)
        evidence.append(dict(member=member, frame=frame, success=bool(result.success), status=int(result.status),
            iterations=int(result.nit), message=result.message,
            constraint_excess_max=float(np.maximum(-inequality(best), 0).max(initial=0.)),
            optimizer_terminal_constraint_excess_max=float(np.maximum(-inequality(result.x), 0).max(initial=0.)),
            initial_objective=initial_value, returned_objective=best_value,
            xtc_precision=XTC_PRECISION, coordinate_error_bound_angstrom=float(axis_error),
            pair_distance_error_bound_angstrom=float(pair_error),
            maximum_angle_error_bound_radians=float(angle_error.max()) if len(angle_error) else 0.,
            mass_centre_error_angstrom=float(np.linalg.norm(mass@corrected[member, frame]-centre))))
        if preserve_observables:
            evidence[-1]['observable_constraint_excess_max'] = float(np.maximum(-observable.fun(best), 0).max(initial=0.))
            evidence[-1]['feasibility_stage'] = feasibility
    return corrected, evidence
