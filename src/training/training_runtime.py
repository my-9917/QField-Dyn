"""Logical random streams and complete-prefix gradient accumulation."""
from contextlib import nullcontext
import torch
from aligned_losses import feature_context
from semiflexible_model import device_inputs


def logical_streams(seed, logical_size, rank, world, device, saved_states=None):
    streams, states = {}, {}
    for slot in range(rank, logical_size, world):
        streams[slot] = {name:torch.Generator(device=device).manual_seed(seed+slot*100+i)
                         for i,name in enumerate(('noise','flow_time','rollout'))}
        if saved_states is not None:
            states[slot] = saved_states[slot]
            for name,generator in streams[slot].items():
                generator.set_state(states[slot][name])
        else:
            states[slot] = dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device).cpu())
    return streams, states


def capture_streams(streams, states):
    return {slot:dict(states[slot],**{name:g.get_state().cpu() for name,g in values.items()})
            for slot,values in streams.items()}


def prefix_objectives(model, wrapped, record, streams, config, updates, device,
                      rollout, calibration, scales):
    data = device_inputs(record,device)
    target = record['X_future'].to(device=device,dtype=torch.float32)
    noise = model.source_noise(data,1,len(target),streams['noise'])
    flow_time = torch.rand(1,generator=streams['flow_time'],device=device)
    aligned = config.get('aligned')
    weight = config['geometry_weight']
    if aligned:
        weight *= 1-min(1.,updates/aligned['endpoint_transition_updates'])
    use_geometry = (updates+1)%config['geometry_every_updates']==0 and weight>0
    payload = None
    if rollout:
        noise_paths = torch.cat([model.source_noise(data,1,len(target),streams['rollout']) for _ in range(2)])
        payload = dict(noise=noise_paths,steps=aligned['rollout_steps'],geometry=record['geometry'],
                       features=feature_context(record,device),scales=scales,environment_epsilon=0.)
    terms = wrapped(data,target,noise,flow_time,record['geometry'] if use_geometry else None,aligned=payload)
    endpoint = sum(v for k,v in terms.items() if k not in ('cfm','rollout_physics','feature_es','coordinate_es'))
    loss = terms['cfm']+weight*endpoint
    if rollout:
        loss = loss+sum(value*terms[key] for key,value in calibration['weights'].items())
    return loss,terms,dict(endpoint_weight=weight,geometry_update=use_geometry,flow_time=float(flow_time[0]))


def accumulated_update(model, wrapped, batch, streams, states, config, updates,
                       device, rank, world, calibration, scales, optimizer, diagnostics=None):
    """Every active logical slot contributes once to the same global mean loss."""
    active = sum(file is not None for file in batch['files'])
    slots = list(range(rank,len(batch['files']),world))
    optimizer.zero_grad(set_to_none=True)
    if diagnostics is not None:
        from training_diagnostics import module_groups,module_norms
        groups=module_groups(model)
        before={n:p.detach().clone() for parameters in groups.values() for n,p in parameters.items()}
        diagnostics['parameter_norm_before']=module_norms(groups)
    rows = []
    for index,slot in enumerate(slots):
        real = batch['files'][slot] is not None
        file = batch['files'][slot] if real else next(f for f in batch['files'] if f is not None)
        record = torch.load(file,weights_only=False,map_location='cpu')
        assert record['meta']['partition']=='train'
        if not real:
            generators = {k:torch.Generator(device=device).set_state(g.get_state()) for k,g in streams[slot].items()}
        else:
            generators = streams[slot]
        torch.set_rng_state(states[slot]['torch'])
        torch.cuda.set_rng_state(states[slot]['cuda'],device)
        sync = wrapped.no_sync() if world>1 and index<len(slots)-1 else nullcontext()
        with sync:
            loss,terms,info = prefix_objectives(model,wrapped,record,generators,config,updates,
                device,batch['rollout'],calibration,scales)
            assert torch.isfinite(loss)
            (loss*(world/active if real else 0.)).backward()
        if real:
            states[slot].update(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device).cpu())
            rows.append(dict(file=file,id=record['meta']['id'],tier=record['meta']['tier'],logical_rank=slot,
                loss=float(loss.detach()),**info,**{k:float(v.detach()) for k,v in terms.items()}))
    if diagnostics is not None:diagnostics['gradient_norm_before_clip']=module_norms(groups,gradients=True)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        config['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    if diagnostics is not None:
        diagnostics['update_norm']=module_norms(groups,before=before)
        diagnostics['relative_update_norm']={k:v/diagnostics['parameter_norm_before'][k]
            if diagnostics['parameter_norm_before'][k]>0 else None for k,v in diagnostics['update_norm'].items()}
        diagnostics['clip_coefficient']=min(1.,config['gradient_clip']/(float(norm)+1e-6))
        diagnostics['global_branch_scope']='subset of flow parameters'
    return rows,float(norm)
