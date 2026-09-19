"""Read-only module gradient and actual optimizer-step measurements."""
import torch


def module_groups(model):
    named={name:p for name,p in model.named_parameters() if p.requires_grad}
    return dict(encoder={n:p for n,p in named.items() if n.startswith('encoder.')},
        quantum={n:p for n,p in named.items() if n=='history.theta'},
        flow={n:p for n,p in named.items() if n.startswith('flow.')},
        global_branch={n:p for n,p in named.items() if '.global_gate.' in n})


def module_norms(groups, before=None, gradients=False):
    values={}
    with torch.no_grad():
        for group,parameters in groups.items():
            tensors=[p.grad if gradients else p.detach()-before[n] if before is not None else p.detach()
                     for n,p in parameters.items()]
            values[group]=float(sum(t.double().square().sum() for t in tensors).sqrt())
    return values
