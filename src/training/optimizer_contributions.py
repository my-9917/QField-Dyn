"""Measure inherited AdamW increments at the current, shared training state."""
import copy
import torch
import torch.distributed as dist
from training_runtime import prefix_objectives, capture_streams, logical_streams


def adam_increment(model, optimizer_state, gradient, clip):
    parameters = [torch.nn.Parameter(p.detach().clone()) for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    before = [p.detach().clone() for p in parameters]
    offset = 0
    for p in parameters:
        p.grad = gradient[offset:offset+p.numel()].reshape_as(p).to(p)
        offset += p.numel()
    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return torch.cat([(p.detach()-x).flatten().double() for p,x in zip(parameters,before)])


def measure_contributions(probe, model, optimizer, batch, streams, states, config,
                          updates, device, rank, world, calibration, scales):
    """Probe a separate graph; preserve the production graph and all RNG streams."""
    probe.load_state_dict(model.state_dict())
    probe.train()
    local = capture_streams(streams, states)
    ps, pt = logical_streams(config['seed'], 4, rank, world, device, local)
    parameters = [p for p in probe.parameters() if p.requires_grad]
    keys = ('cfm', 'rollout_physics', 'coordinate_es')
    gradients = {key:torch.zeros(sum(p.numel() for p in parameters), device=device, dtype=torch.float64)
                 for key in keys}
    for slot in range(rank, 4, world):
        record = torch.load(batch['files'][slot], weights_only=False, map_location='cpu')
        torch.set_rng_state(pt[slot]['torch'])
        torch.cuda.set_rng_state(pt[slot]['cuda'], device)
        _, terms, info = prefix_objectives(probe, probe, record, ps[slot], config, updates,
                                           device, True, calibration, scales)
        assert info['endpoint_weight'] == 0
        for i,key in enumerate(keys):
            values = torch.autograd.grad(terms[key], parameters, retain_graph=i<2)
            gradients[key] += torch.cat([v.detach().flatten().double() for v in values])/4
        del terms, values
    for vector in gradients.values():
        dist.all_reduce(vector)
        assert torch.isfinite(vector).all()
    result = None
    if rank == 0:
        state = optimizer.state_dict()
        baseline = adam_increment(model, state, gradients['cfm'], config['gradient_clip'])
        assert baseline.norm()>0
        ratios = {}
        for key,weight in calibration['weights'].items():
            delta = adam_increment(model, state, gradients['cfm']+weight*gradients[key], config['gradient_clip'])
            ratios[key] = float((delta-baseline).norm()/baseline.norm())
        joint = gradients['cfm']+sum(calibration['weights'][k]*gradients[k] for k in calibration['weights'])
        result = dict(ratios=ratios, cfm_update_norm=float(baseline.norm()),
                      joint_update_norm=float(adam_increment(model,state,joint,config['gradient_clip']).norm()),
                      gradient_norms={k:float(v.norm()) for k,v in gradients.items()})
    dist.barrier()
    return result
