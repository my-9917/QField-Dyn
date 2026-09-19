"""Generate frame outputs or candidate pools and assemble complete-prefix decisions."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from projection import project_paths
from projection_errors import ProjectionFeasibilityError


@lru_cache(maxsize=1)
def context(artifact_path, config_path):
    torch.set_num_threads(2)
    artifact = torch.load(artifact_path, weights_only=False, map_location='cpu')
    record = torch.load(artifact['source_record'], weights_only=False, map_location='cpu')
    config = json.loads(Path(config_path).read_text())
    return artifact, record, config


def project_chunk(task):
    artifact_path, config_path, output, begin, end = task
    artifact, record, config = context(artifact_path, config_path)
    paths = np.ascontiguousarray(artifact['paths'])
    horizon = paths.shape[1]
    flat = paths.reshape(-1, *paths.shape[-2:])
    if config['method'] == 'prefix_candidate_selection':
        from projection_candidates import frame_candidates
        pools = [frame_candidates(raw, record, config) for raw in flat[begin:end]]
        torch.save(dict(source_artifact=str(Path(artifact_path).resolve()), begin=begin, end=end,
            candidates=pools), output)
        return str(Path(output).parent), begin, end
    try:
        corrected, evidence = project_paths(flat[begin:end][None], record, config)
    except ProjectionFeasibilityError as error:
        failure = error.args[0]
        failure['member'], failure['frame'] = divmod(begin+failure['frame'], horizon)
        failure['source_artifact'] = artifact_path
        torch.save(dict(source_artifact=artifact_path, begin=begin, end=end, failure=failure),
            Path(output).with_suffix('.failed.pt'))
        raise
    for row in evidence:
        row['member'], row['frame'] = divmod(begin+row['frame'], horizon)
    torch.save(dict(source_artifact=str(Path(artifact_path).resolve()), begin=begin, end=end,
        paths=corrected[0], evidence=evidence), output)
    return str(Path(output).parent), begin, end


def collect_chunks(artifact_path, paths, chunks, record=None, config=None, selection_workers=1):
    plan = json.loads((chunks/'manifest.json').read_text())
    total = paths.shape[0]*paths.shape[1]
    assert plan['source_artifact'] == str(artifact_path.resolve()) and plan['frames'] == total
    result = np.empty(paths.shape, dtype=np.float64)
    flat = result.reshape(-1, *paths.shape[-2:])
    seen = np.zeros(total, dtype=bool)
    evidence = [None]*total
    candidate_mode = config is not None and config['method'] == 'prefix_candidate_selection'
    pools = [None]*total if candidate_mode else None
    for item in plan['chunks']:
        saved = torch.load(chunks/item['file'], weights_only=False, map_location='cpu')
        begin, end = saved['begin'], saved['end']
        assert saved['source_artifact'] == plan['source_artifact']
        assert (begin, end) == (item['begin'], item['end']) and 0 <= begin < end <= total
        assert not seen[begin:end].any()
        if candidate_mode:
            assert len(saved['candidates']) == end-begin
            pools[begin:end] = saved['candidates']; seen[begin:end] = True
            continue
        assert len(saved['evidence']) == end-begin
        flat[begin:end] = saved['paths']; seen[begin:end] = True
        for index, row in enumerate(saved['evidence'], begin):
            assert (row['member'], row['frame']) == divmod(index, paths.shape[1])
            evidence[index] = row
    assert seen.all()
    if candidate_mode:
        if selection_workers > 1 and paths.shape[0] > 1:
            from parallel_path_selection import select_paths_parallel
            return select_paths_parallel(paths, pools, record, config, selection_workers)
        from prefix_projection import select_prefix
        return select_prefix(paths, pools, record, config)
    return result, evidence
