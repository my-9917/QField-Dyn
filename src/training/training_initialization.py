"""Initialize Aligned model parameters and transfer shared AdamW moments."""
import torch


def initialize_aligned(saved, device, directional_statistics=None):
    """C100 weights plus exactly the declared zero-output global channel."""
    from semiflexible_model import SemiFlexFlow
    spec = dict(saved['spec'], global_position=True)
    if directional_statistics is not None:
        spec['directional_statistics'] = directional_statistics
    model = SemiFlexFlow(**spec).to(device)
    current = model.state_dict()
    added = set(current)-set(saved['model'])
    assert added == {'atomic_masses'} | {name for name in current if '.global_gate.' in name}
    current.update(saved['model'])
    model.load_state_dict(current, strict=True)
    return model


def inherit_adam_state(model, optimizer, saved):
    """Transfer shared C100 moments by name; AdamW creates states for new gates."""
    import copy
    from semiflexible_model import SemiFlexFlow
    parent = SemiFlexFlow(**saved['spec'])
    names = [name for name, p in parent.named_parameters() if p.requires_grad]
    groups = saved['optimizer']['param_groups']
    assert len(groups) == 1 and len(groups[0]['params']) == len(names)
    current = dict(model.named_parameters())
    state = optimizer.state_dict()
    ids = dict(zip((name for name, p in model.named_parameters() if p.requires_grad), state['param_groups'][0]['params']))
    for name, old_id in zip(names, groups[0]['params']):
        assert torch.equal(current[name].detach().cpu(), saved['model'][name]), name
        state['state'][ids[name]] = copy.deepcopy(saved['optimizer']['state'][old_id])
    optimizer.load_state_dict(state)
    return dict(shared_parameters=len(names), fresh_parameters=len(ids)-len(names),
                parent_updates=saved['updates'], mapping='trainable parameter name')
