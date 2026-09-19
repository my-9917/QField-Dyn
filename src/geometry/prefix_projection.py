"""Generate per-frame candidates, then select a complete predicted prefix."""
import numpy as np
from projection_candidates import frame_candidates
from prefix_selection import select_candidates
from semiflexible_scores import observables, physics_scores
from projection_errors import ProjectionFeasibilityError


def select_prefix(paths, pools, record, config, budget_state=None):
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    m, h = paths.shape[:2]
    if config.get('selection_scope') == 'independent_paths' and m > 1:
        assert budget_state is None
        corrected, evidence = [], []
        for member in range(m):
            path, rows = select_prefix(paths[member:member+1], pools[member*h:(member+1)*h], record, config)
            corrected.append(path)
            evidence.extend(dict(row, member=member, selection_scope='independent_paths') for row in rows)
        return np.concatenate(corrected), evidence
    raw = observables(paths, record)
    raw_physics = physics_scores(paths, record['geometry'])[1]
    raw_counts = raw_physics['ligand_self_overlap_violations']+raw_physics['ligand_environment_overlap_violations']
    encoded_views = None
    if config.get('selection_representation') == 'coordinate_and_xtc':
        from projection_encoding import encode_coordinates
        sizes = [len(pool['coordinates']) for pool in pools]
        encoded, encoded_record = encode_coordinates(np.concatenate([pool['coordinates'] for pool in pools])[None],record)
        encoded_pools = np.split(encoded[0],np.cumsum(sizes)[:-1])
        encoded_views = dict(contacts=[],counts=[])
        filtered=[]
        for index,(pool,coordinates) in enumerate(zip(pools,encoded_pools)):
            phys = physics_scores(coordinates[:,None],encoded_record['geometry'])[1]
            keep = np.flatnonzero((phys['bond_violations'][:,0] == 0) &
                (phys['angle_violations'][:,0] == 0) & (phys['local_configuration_flips'][:,0] == 0))
            if not len(keep):
                raise ProjectionFeasibilityError(dict(stage='encoded_frame_pool',frame=index,
                    outcome='finite_candidate_pool_has_no_encoded_geometric_solution'))
            filtered.append(dict(pool,coordinates=pool['coordinates'][keep],
                evidence=[dict(pool['evidence'][k],source_candidate_index=int(k)) for k in keep]))
            encoded_views['contacts'].append(observables(coordinates[keep,None],encoded_record)['distances'][:,0] < 4.5)
            encoded_views['counts'].append((phys['ligand_self_overlap_violations']+phys['ligand_environment_overlap_violations'])[keep,0])
        pools = filtered
    contacts, counts, costs = [], [], []
    for index, pool in enumerate(pools):
        member, frame = divmod(index, h)
        candidates = pool['coordinates']
        ob = observables(candidates[:, None], record)
        phys = physics_scores(candidates[:, None], record['geometry'])[1]
        contacts.append(ob['distances'][:, 0] < 4.5)
        counts.append((phys['ligand_self_overlap_violations']+phys['ligand_environment_overlap_violations'])[:, 0])
        costs.append(.5*np.square(candidates-paths[member, frame]).sum((1, 2)))
        assert (phys['bond_violations'] == 0).all() and (phys['angle_violations'] == 0).all()
        assert (phys['local_configuration_flips'] == 0).all()
    chosen, report = select_candidates(raw['distances'] < 4.5, raw_counts,
        contacts, counts, costs, raw['pocket_slots'], config['contact_brier_increase_max'], budget_state,
        encoded=encoded_views)
    if chosen is None:
        raise ProjectionFeasibilityError(report)
    result, evidence = np.empty_like(paths), []
    for index, (pool, k) in enumerate(zip(pools, chosen)):
        member, frame = divmod(index, h)
        result[member, frame] = pool['coordinates'][k]
        evidence.append(dict(pool['evidence'][k], member=member, frame=frame, candidate_index=int(k)))
    from projection_encoding import review_encoded_prefix
    encoded = review_encoded_prefix(paths, result, record, config['contact_brier_increase_max'], budget_state)
    if not (encoded['certified'] and encoded['geometry_passed']):
        raise ProjectionFeasibilityError(dict(stage='encoded_prefix', pre_encoding=report, encoded=encoded))
    evidence[0]['prefix_selection'] = report
    evidence[0]['encoded_prefix_selection'] = encoded
    if budget_state is not None:
        budget_state.update(encoded['ledger'])
    return result, evidence


def project_prefix(paths, record, config, budget_state=None):
    paths = np.ascontiguousarray(paths, dtype=np.float64)
    pools = [frame_candidates(raw, record, config) for raw in paths.reshape(-1, *paths.shape[-2:])]
    return select_prefix(paths, pools, record, config, budget_state)
