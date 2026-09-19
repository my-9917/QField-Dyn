"""Single-path adapter inference and T4 feedback using decoded delivery coordinates."""
import numpy as np
import torch
import time
from trajectory_adapter import adapter_context
from covalent_reconstruction import reconstruction_context
from verified_sampling import sample_adapted_path


@torch.no_grad()
def generate_adapted(base,adapter,data,record,seed,chemistry=None,progress=None,feedback_calibration=None):
    meta=record['meta'];horizon=meta['n_pred'];device=data['X_obs'].device
    generator=torch.Generator(device=device).manual_seed(seed)
    started=time.perf_counter();original,quantum=base.condition(data)
    geometry=reconstruction_context(record,device,chemistry)
    if data['X_obs'].is_cuda:torch.cuda.synchronize(device)
    conditioning_seconds=time.perf_counter()-started
    if meta['tier']!='T4':
        noise=base.source_noise(data,1,horizon,generator)
        context=adapter_context(record,dict(ligand_features=original['ligand_features'],quantum_condition=quantum),device,geometry)
        result=sample_adapted_path(base,adapter,noise,original,quantum,context,record,progress=progress)
        return result['coordinates'],[],dict(raw_ligand=result['raw'].cpu(),chemical_restrictions_available=chemistry is not None,
            integration_steps=result['steps'],resolution_checks=result['resolution_checks'],
            timing=dict(conditioning_seconds=conditioning_seconds,sampling=result['timing']))
    block=int(6400//meta['dt_ps']);assert block>=2
    if feedback_calibration is not None:
        from t4_geometry_feedback import feedback_geometry
        feedback_context=feedback_geometry(geometry,feedback_calibration)
    observed=data['X_obs'];current=observed[-1:];parts=[];raw_parts=[];adapter_parts=[];rows=[];rng_states=[]
    for begin in range(0,horizon,block):
        started=time.perf_counter();count=min(block,horizon-begin);local=dict(data,X_obs=current)
        static=base.recurrent_static(local,original)
        rng_states.append(generator.get_state().cpu())
        noise=base.source_noise(local,1,count,generator)
        local_record=dict(record,meta=dict(meta,n_pred=count),inputs=dict(record['inputs'],X_obs=observed))
        context=adapter_context(local_record,dict(ligand_features=static['ligand_features'],quantum_condition=quantum),device,geometry)
        if data['X_obs'].is_cuda:torch.cuda.synchronize(device)
        recurrent_seconds=time.perf_counter()-started
        feedback=(feedback_context,feedback_calibration) if feedback_calibration is not None else None
        result=sample_adapted_path(base,adapter,noise,static,quantum,context,record,feedback=feedback,progress=progress,
                                  feedback_start=current[-1].cpu().numpy())
        corrected=result['coordinates'];delivered=torch.as_tensor(result['encoded'],device=device)
        adapter_parts.append(result['adapter'].cpu());repair_detail=result['repair']
        row=dict(start_frame=begin,frames=count,absolute_lead_times_ps=(np.arange(begin+1,begin+count+1)*meta['dt_ps']).tolist(),
            input_last_frame=current.cpu().tolist(),output_last_frame=delivered[-1].cpu().tolist(),
            maximum_step_angstrom=float(torch.diff(torch.cat((current.double(),delivered)),dim=0).norm(dim=-1).max()),
            feedback='full_atom_XTC_decoded_then_float32',history='original QMem state; rolling adapter observations',
            geometry_bounds='initial observed geometry; retained across blocks')
        row['covalent_feedback']=repair_detail
        row.update(integration_steps=result['steps'],resolution_checks=result['resolution_checks'])
        row['timing']=dict(recurrent_conditioning_seconds=recurrent_seconds,sampling=result['timing'])
        rows.append(row);parts.append(corrected);raw_parts.append(result['raw'].cpu())
        current=delivered[-1:].to(data['X_obs'].dtype)
        observed=torch.cat((observed,delivered.to(observed.dtype)))[-meta['n_obs']:]
        if progress:progress(dict(stage='T4_adapter_block',**row))
    return torch.cat(parts),rows,dict(raw_ligand=torch.cat(raw_parts),adapter_ligand=torch.cat(adapter_parts),block_rng_states=rng_states,
        chemical_restrictions_available=chemistry is not None,
        timing=dict(conditioning_seconds=conditioning_seconds,blocks=[r['timing'] for r in rows]),
        scope='long-time numerical stress test; 80/160/320 ps adapter lags are undefined at 1 ns sampling')
