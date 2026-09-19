"""Finite candidate selection with truth-free empirical-contact and collision budgets."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def brier_bound(p, q):
    return np.maximum(q*q-p*p, (1-q)**2-(1-p)**2)


def select_candidates(raw_contacts, raw_counts, contacts, counts, costs, pockets, limit, state=None, encoded=None):
    """Each list row is one (member, frame); each candidate contributes R contact bits."""
    m, h, r = raw_contacts.shape
    prior = state or dict(frames=0, raw_collisions=0, final_collisions=0, all_sum=0., pocket_sum=0.)
    groups = [('all', np.arange(r)), ('pocket', np.asarray(pockets, dtype=int))]
    collision_views = [counts]
    if encoded is not None:
        assert [len(c) for c in encoded['contacts']] == [len(c) for c in contacts]
        raw_contacts = np.concatenate((raw_contacts, raw_contacts), axis=-1)
        contacts = [np.concatenate((a,b),axis=-1) for a,b in zip(contacts,encoded['contacts'])]
        groups += [('encoded_all', np.arange(r,2*r)), ('encoded_pocket', np.asarray(pockets,dtype=int)+r)]
        collision_views.append(encoded['counts'])
        r *= 2
    base = np.stack([c[0] for c in contacts]).reshape(m, h, r)
    p, q0 = raw_contacts.mean(0), base.mean(0)
    variable_frames = [i for i, c in enumerate(contacts) if len(c) > 1]
    offsets, nz = {}, 0
    cells = {}
    for i in variable_frames:
        offsets[i] = nz
        delta = (contacts[i].astype(float)-contacts[i][0])/m
        for k, residue in np.argwhere(delta != 0):
            cells.setdefault((i % h, int(residue)), []).append((nz+int(k), delta[k, residue]))
        nz += len(contacts[i])
    changed_cells = sorted(cells)
    initial_bound = brier_bound(p, q0)
    report = dict(ensemble_size=m, frames=h, total_candidates=sum(map(len, contacts)),
        variable_frames=len(variable_frames), variable_probability_cells=len(changed_cells),
        scope='finite candidate pool; empirical probabilities; future truth excluded')

    def assess(chosen):
        final = np.stack([c[k] for c, k in zip(contacts, chosen)]).reshape(m, h, r)
        w = brier_bound(p, final.mean(0))
        collision_sum = sum(int(c[k]) for c, k in zip(counts, chosen))
        updated = dict(frames=prior['frames']+h, raw_collisions=prior['raw_collisions']+int(raw_counts.sum()),
            final_collisions=prior['final_collisions']+collision_sum)
        for name, slots in groups:
            updated[name+'_sum'] = prior[name.removeprefix('encoded_')+'_sum']+float(w[:, slots].sum())
        collision_sums = [sum(int(c[k]) for c,k in zip(view,chosen)) for view in collision_views]
        certified = all(prior['final_collisions']+v <= updated['raw_collisions'] for v in collision_sums) and all(
            updated[name+'_sum'] <= limit*updated['frames']*len(slots) for name, slots in groups)
        return dict(certified=certified, ledger=updated, collision_count=collision_sum,
            brier_bounds={name: float(w[:, slots].mean()) if len(slots) else None for name, slots in groups},
            displacement_cost=float(sum(c[k] for c, k in zip(costs, chosen))),
            **(dict(encoded_collision_count=collision_sums[1]) if encoded is not None else {}))

    if nz == 0:
        chosen = np.zeros(m*h, dtype=int)
        check = assess(chosen)
        report.update(check, solver_status='single_combination')
        return (chosen if check['certified'] else None), report
    row_indices, columns, values, lower, upper = [], [], [], [], []

    def add(entries, lo, hi):
        index = len(lower)
        for col, value in entries:
            if value != 0:
                row_indices.append(index); columns.append(col); values.append(value)
        lower.append(lo); upper.append(hi)

    objective = np.zeros(nz+len(changed_cells))
    for i in variable_frames:
        start = offsets[i]; size = len(contacts[i])
        add([(start+k, 1.) for k in range(size)], 1., 1.)
        objective[start:start+size] = costs[i]-min(costs[i])
    for view in collision_views:
        collision_entries = [(offsets[i]+k,int(view[i][k])-int(view[i][0]))
            for i in variable_frames for k in range(len(view[i]))]
        add(collision_entries, -np.inf,
            prior['raw_collisions']+int(raw_counts.sum())-prior['final_collisions']-sum(int(c[0]) for c in view))
    for index, cell in enumerate(changed_cells):
        grid = np.arange(m+1)/m
        w = brier_bound(p[cell], grid)
        slopes = m*np.diff(w); intercepts = w[:-1]-slopes*grid[:-1]
        for slope, intercept in zip(slopes, intercepts):
            add([(nz+index, 1.)]+[(col, -slope*value) for col, value in cells[cell]],
                slope*q0[cell]+intercept, np.inf)
    for name, slots in groups:
        selected = [(j, cell) for j, cell in enumerate(changed_cells) if cell[1] in slots]
        fixed = float(initial_bound[:, slots].sum())-sum(float(initial_bound[cell]) for _, cell in selected)
        allowed = limit*(prior['frames']+h)*len(slots)-prior[name.removeprefix('encoded_')+'_sum']-fixed
        add([(nz+j, 1.) for j, _ in selected], -np.inf, allowed)
    matrix = coo_matrix((values, (row_indices, columns)), shape=(len(lower), len(objective))).tocsc()
    result = milp(objective, integrality=np.r_[np.ones(nz), np.zeros(len(changed_cells))],
        bounds=Bounds(np.zeros(len(objective)), np.r_[np.ones(nz), np.full(len(changed_cells), np.inf)]),
        constraints=LinearConstraint(matrix, lower, upper), options=dict(mip_rel_gap=0.))
    report.update(solver_status=int(result.status), solver_message=str(result.message))
    if result.status == 2:
        report.update(certified=False, outcome='finite_pool_has_no_certified_combination',
            original_brier_acceptance='requires_development_truth', continuous_feasibility='unresolved')
        return None, report
    assert result.status == 0, report
    chosen = np.zeros(m*h, dtype=int)
    for i in variable_frames:
        start = offsets[i]
        chosen[i] = int(result.x[start:start+len(contacts[i])].argmax())
    check = assess(chosen)
    report.update(check, mip_gap=float(result.mip_gap))
    assert check['certified'], report
    return chosen, report
