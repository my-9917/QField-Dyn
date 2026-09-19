"""Four scheduled adaptation epochs; frozen E3 proposals, final-coordinate losses."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from trajectory_adapter import TrajectoryAdapter
from adapter_data import load_adapter_case,ready_proposals
from adapter_losses import adapter_objectives


def save_checkpoint(path,model,optimizer,**state):
    payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),**state)
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    for key in ('root','output','statistics'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--smoke-updates',type=int,default=0)
    p.add_argument('--engineering-review',type=Path,required=True)
    p.add_argument('--chain-review',type=Path,required=True)
    p.add_argument('--smoke-review',type=Path)
    a=p.parse_args();gate=json.loads(a.engineering_review.read_text());assert gate['passed']
    assert json.loads(a.chain_review.read_text())['passed']
    if not a.smoke_updates:assert a.smoke_review and json.loads(a.smoke_review.read_text())['passed']
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(202609191437);torch.cuda.manual_seed_all(202609191437)
    manifest=json.loads((a.root/'inputs/manifest.json').read_text());cfg=manifest['config']
    scales=torch.as_tensor(json.loads(a.statistics.read_text())['scale'],device='cuda',dtype=torch.float64)
    first=manifest['probe'][0];first_cache=torch.load(a.root/'proposals'/(Path(first['file']).stem+'.pt'),weights_only=False,map_location='cpu')
    spec=dict(feature_dim=first_cache['ligand_features'].shape[-1],hidden=64,layers=2,reconstruction_steps=8,structured_initialization=True,initialization_steps=32,final_damping=1.)
    model=TrajectoryAdapter(**spec).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    identity=first_cache['checkpoint_sha256'];params=list(model.parameters());start_epoch=0;start_offset=0;updates=0
    norms={k:[] for k in ('coordinate','dynamics','physics')};calibration=[]
    saved=torch.load(a.resume,weights_only=False,map_location='cpu') if a.resume else None
    for row in ([] if saved else manifest['probe'][:4]):
        record,cache,context=load_adapter_case(row['file'],a.root/'proposals'/(Path(row['file']).stem+'.pt'),a.root/'chemistry_combined_r3')
        assert cache['checkpoint_sha256']==identity and record['meta']['partition']=='train'
        paths,diagnostic=model(cache['paths'][:2].cuda(),context)
        losses=adapter_objectives(paths,record['X_future'].cuda().double(),diagnostic,record,context,scales)
        measured={}
        for key in norms:
            grads=torch.autograd.grad(losses[key],params,retain_graph=True)
            norm=torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
            assert torch.isfinite(norm)
            norms[key].append(float(norm));measured[key]=float(norm)
        calibration.append(dict(prefix=Path(row['file']).stem,gradient_norms=measured))
        del paths,diagnostic,losses,context,record,cache,grads
    weights={'coordinate':1.,'regularizer':.01}
    for key in (() if saved else ('dynamics','physics')):
        assert np.mean(norms[key])>0
        weights[key]=.25*np.mean(norms['coordinate'])/np.mean(norms[key])
    config=dict(spec=spec,weights=weights,calibration=calibration,base_checkpoint_sha256=identity,
        epochs=4,global_batch=4,learning_rate=1e-4,training_seed=202609191437,
        model_version='covalent_adapter_0919_v3',deterministic_algorithms=True,
        gradient_safety_max=1e5,
        input_preparation='frozen E3 to structured coordinates; final reconstruction remains differentiable',
        reconstruction=dict(steps=8,damping=1.,initialization_steps=32,initialization_damping=.1,max_atom_step_angstrom=.25,
            smooth_interval_width=.002,angle_residual='cosine',locked_distance_margin_angstrom=.02,
            planar_volume_margin=.02,stable_planarity_threshold=.05,chiral_min_volume=.1),
        chemical_uncertainty='explicit partial coverage; charges excluded from energy claims',
        formal_baselines=['static','linear_last5'],reconstruction_backend='cuda_float64',training_order='manifest order E1; seeded shuffle E2-E4')
    if saved:
        assert saved['spec']==spec and saved['config']['base_checkpoint_sha256']==identity
        config=saved['config'];weights=config['weights']
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2));common=dict(spec=spec,config=config,base_frozen=True)
    if a.resume:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        start_epoch=saved['next_epoch'];start_offset=saved['next_offset'];updates=saved['updates']
    else:
        save_checkpoint(a.output/'initial.pt',model,optimizer,next_epoch=0,next_offset=0,updates=0,**common)
    files=[file for member in manifest['train'] for file in member['records']];assert len(files)==384
    if a.smoke_updates:
        assert 0<a.smoke_updates<=16
        files=[r['file'] for r in manifest['probe']]
    pairs=[(0,1),(2,3),(0,2),(1,3)];started=time.perf_counter();elapsed_updates=[]
    for epoch in range(start_epoch,4):
        order=list(range(len(files))) if epoch==0 else torch.randperm(len(files),generator=torch.Generator().manual_seed(202609191437+epoch)).tolist()
        offset=start_offset if epoch==start_epoch else 0
        for batch_start in range(offset,len(files),4):
            names=[Path(files[i]).stem for i in order[batch_start:batch_start+4]]
            while not set(names).issubset(ready_proposals(a.root/'proposals')):
                missing=sorted(set(names)-ready_proposals(a.root/'proposals'))
                (a.output/'status.json').write_text(json.dumps(dict(stage='waiting_for_frozen_cache',epoch=epoch+1,updates=updates,missing=missing)))
                time.sleep(5)
            optimizer.zero_grad(set_to_none=True);batch_begin=time.perf_counter();values={k:0. for k in weights};chemical=[]
            for index in order[batch_start:batch_start+4]:
                record,cache,context=load_adapter_case(files[index],a.root/'proposals'/(Path(files[index]).stem+'.pt'),a.root/'chemistry_combined_r3')
                assert record['meta']['partition']=='train' and cache['checkpoint_sha256']==identity
                sample=cache['paths'][list(pairs[epoch])].cuda()
                predicted,diagnostic=model(sample,context)
                losses=adapter_objectives(predicted,record['X_future'].cuda().double(),diagnostic,record,context,scales)
                loss=sum(weights[k]*v for k,v in losses.items())/4
                assert torch.isfinite(loss) and torch.isfinite(predicted).all()
                loss.backward()
                for key,value in losses.items():values[key]+=float(value.detach())/4
                chemical.append(context['geometry']['chemical_restrictions_available'])
                del record,cache,context,sample,predicted,diagnostic,losses,loss
            grad_norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if float(grad_norm)>=config['gradient_safety_max']:
                failure=dict(stage='gradient_safety',epoch=epoch+1,next_update=updates+1,prefixes=names,
                    gradient_norm=float(grad_norm),losses=values,last_complete_checkpoint=str(a.output/'latest.pt'))
                (a.output/'failure.json').write_text(json.dumps(failure,indent=2))
                raise RuntimeError('Final-coordinate gradient amplification requires repair before this update')
            optimizer.step();updates+=1;torch.cuda.synchronize();seconds=time.perf_counter()-batch_begin;elapsed_updates.append(seconds)
            next_offset=batch_start+4;next_epoch=epoch
            if next_offset==len(files):next_epoch=epoch+1;next_offset=0
            save_checkpoint(a.output/'latest.pt',model,optimizer,next_epoch=next_epoch,next_offset=next_offset,updates=updates,**common)
            row=dict(epoch=epoch+1,update=updates,prefixes=names,losses=values,total=sum(weights[k]*v for k,v in values.items()),
                gradient_norm=float(grad_norm),seconds=seconds,chemical_restrictions_available=chemical,finite=True)
            with (a.output/'updates.jsonl').open('a') as file:file.write(json.dumps(row)+'\n')
            (a.output/'status.json').write_text(json.dumps(dict(stage='training',epoch=epoch+1,updates=updates,last_update=row)))
            print(json.dumps(row),flush=True)
            if a.smoke_updates and updates>=a.smoke_updates:
                (a.output/'smoke_review.json').write_text(json.dumps(dict(passed=True,updates=updates,
                    mean_update_seconds=float(np.mean(elapsed_updates)),max_update_seconds=max(elapsed_updates),
                    finite=True,reset_required_before_production=True,trained_parameter_count=sum(p.numel() for p in params)),indent=2));return
        save_checkpoint(a.output/f'epoch_{epoch+1:02d}.pt',model,optimizer,next_epoch=epoch+1,next_offset=0,updates=updates,**common)
    (a.output/'completion.json').write_text(json.dumps(dict(completed=True,epochs=4,updates=updates,prefixes_per_epoch=len(files),
        stop_reason='four scheduled adaptation epochs',converged=False,elapsed_seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
