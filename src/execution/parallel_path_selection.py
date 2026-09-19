"""Execute the frozen independent-path selector concurrently, in member order."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import torch
from prefix_projection import select_prefix


def select_member(task):
    member, path, pools, record, config = task
    torch.set_num_threads(2)
    corrected, evidence = select_prefix(path, pools, record, config)
    return corrected, [dict(row, member=member, selection_scope='independent_paths')
                       for row in evidence]


def select_paths_parallel(paths, pools, record, config, workers):
    assert config['selection_scope'] == 'independent_paths'
    members, horizon = paths.shape[:2]
    tasks = [(member, paths[member:member+1], pools[member*horizon:(member+1)*horizon],
              record, config) for member in range(members)]
    with ProcessPoolExecutor(max_workers=min(workers, members),
                             mp_context=get_context('spawn')) as pool:
        rows = list(pool.map(select_member, tasks))
    return np.concatenate([row[0] for row in rows]), [item for row in rows for item in row[1]]
