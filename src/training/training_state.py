"""Validate an explicit training-pool expansion at a complete-epoch boundary."""


def validate_cache_expansion(previous, current):
    assert previous['completed'] and current['completed']
    assert previous['input_version'] == current['input_version']
    execution_fields = {'version', 'systems', 'selected_members', 'workers',
                        'source_cache', 'validation_members_source'}
    old = {k:v for k,v in previous['config'].items() if k not in execution_fields}
    new = {k:v for k,v in current['config'].items() if k not in execution_fields}
    assert old == new, 'Training source, calibration and split rule stay fixed'
    groups = []
    for manifest in (previous,current):
        parts = {part:{r['id'] for r in manifest['rows'] if r['partition']==part}
                 for part in ('train','validation')}
        assert parts['train'].isdisjoint(parts['validation'])
        groups.append(parts)
    before,after = groups
    assert before['train'] < after['train'] and before['validation'] == after['validation']
    assert all(len(r['records']) == 3 for m in (previous,current) for r in m['rows'])
    return dict(previous_train_systems=len(before['train']),train_systems=len(after['train']),
        validation_systems=len(after['validation']),validation_members_preserved=True,
        state_policy='preserve weights, AdamW, scheduler, epoch, update, best CFM, shuffle and every rank RNG')
