"""Reuse M-independent frame pools while rebuilding the target ensemble selection."""
import json
from pathlib import Path
import numpy as np
import torch

NUMERIC_FILES = ('geometry_constraints', 'projection_candidates', 'prefix_selection', 'prefix_projection',
                 'projection_encoding', 'observable_constraints', 'terminal_geometry', 'semiflexible_scores')
FRAME_POOL_FILES = ('geometry_constraints', 'projection_candidates',
                    'observable_constraints', 'terminal_geometry', 'semiflexible_scores')


def reuse_pools(previous, artifact_path, config_path, folder):
    key = artifact_path.stem
    if (previous/'projection_config.json').exists():
        previous_config = json.loads((previous/'projection_config.json').read_text())
    else:
        previous_config = json.loads((previous/'manifest.json').read_text())['config']
    current_config = json.loads(config_path.read_text())
    extending = previous_config['candidate_scopes'] != current_config['candidate_scopes']
    if extending:
        assert previous_config['candidate_scopes'] == ['original_v2', 'collision', 'collision_pocket']
        assert current_config['candidate_scopes'] == previous_config['candidate_scopes']+['pocket']
    selection_metadata = {'version', 'selection_scope', 'selection_objective', 'selection_status',
                          'selection_representation'}
    if extending:
        selection_metadata.add('candidate_scopes')
    assert {k:v for k,v in previous_config.items() if k not in selection_metadata} == {
        k:v for k,v in current_config.items() if k not in selection_metadata}
    # Frame pools precede selection; a new independently verified selector consumes identical pools.
    for name in FRAME_POOL_FILES:
        if extending and name == 'projection_candidates':
            # The verified extension preserves the three existing candidates and
            # appends only the new pocket-constrained solve from the same base.
            continue
        assert (previous/'source'/(name+'.py')).read_bytes() == (Path(__file__).parent/(name+'.py')).read_bytes(), name
    plan = json.loads((previous/'chunks'/key/'manifest.json').read_text())
    old = torch.load(plan['source_artifact'], weights_only=False, map_location='cpu')
    current = torch.load(artifact_path, weights_only=False, map_location='cpu')
    m = len(old['paths'])
    assert old['source_record'] == current['source_record'] and old['meta'] == current['meta']
    assert old['seeds'] == current['seeds'][:m]
    assert np.array_equal(old['paths'], current['paths'][:m])
    assert m <= len(current['paths']) and plan['frames'] == m*current['paths'].shape[1]
    chunks, seen = [], np.zeros(plan['frames'], dtype=bool)
    for item in plan['chunks']:
        saved_path = previous/'chunks'/key/item['file']
        # Interrupted construction preserves its complete leading chunks; the suffix is rebuilt.
        if not saved_path.exists(): break
        saved = torch.load(saved_path, weights_only=False, map_location='cpu')
        begin, end = item['begin'], item['end']
        assert (saved['begin'], saved['end']) == (begin, end) and begin == int(seen.sum())
        assert saved['source_artifact'] == plan['source_artifact'] and len(saved['candidates']) == end-begin
        name = ('seed_' if extending else 'reused_')+Path(item['file']).name
        torch.save(dict(saved, source_artifact=str(artifact_path.resolve())), folder/name)
        chunk = dict(begin=begin,end=end,file=name)
        if extending:
            chunk.update(file='extended_'+Path(item['file']).name, seed_file=name)
        chunks.append(chunk); seen[begin:end] = True
    reused_frames = int(seen.sum())
    assert seen[:reused_frames].all() and not seen[reused_frames:].any()
    return chunks, dict(source=str(previous), source_paths=m, target_paths=len(current['paths']),
        frames=reused_frames, raw_coordinates_equal=True, frame_numeric_sources_equal=not extending,
        shared_geometry_and_observable_sources_equal=True,
        reuse_kind='extend saved frame pools with pocket candidates' if extending else 'complete frame candidate pools',
        target_prefix_selection='recompute')
