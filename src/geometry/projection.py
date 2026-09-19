"""Select the declared output algorithm; solving and scoring have separate modules."""
from prefix_projection import project_prefix
from terminal_geometry import project_frames


def project_paths(paths, record, config, budget_state=None):
    if config['method'] == 'prefix_candidate_selection':
        return project_prefix(paths, record, config, budget_state=budget_state)
    assert budget_state is None
    return project_frames(paths, record, config)
