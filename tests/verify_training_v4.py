"""Two-GPU/four-stream update replay, loss equivalence and stopping transitions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from semiflexible_model import SemiFlexFlow,device_inputs
from training_runtime import logical_streams,capture_streams,accumulated_update
from training_schedule import initial_stopping,update_stopping
from aligned_losses import rollout_objectives,feature_context


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-loss',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==2
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    saved=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
    config=copy.deepcopy(saved['config']);config['aligned']['rollout_steps']=2
    plan=json.loads(a.plan.read_text())
    # Real smallest T1 prefixes, with the full spatial model and actual integration.
    selected=sorted((r for r in plan['selected'] if r['tier']=='T1'),key=lambda r:r['atoms'])[:4]
    batch=dict(files=[r['file'] for r in selected],rollout=True)
    calibration=json.loads(Path(config['aligned']['calibration']).read_text())
    scales=torch.tensor(json.loads(Path(config['aligned']['feature_statistics']).read_text())['scale'],device=device)
    model=SemiFlexFlow(**saved['spec']).to(device);model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
    optimizer.load_state_dict(saved['optimizer'])
    streams,states=logical_streams(config['seed'],4,rank,2,device,saved['rank_states'])
    wrapped=DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False)
    rows,norm=accumulated_update(model,wrapped,batch,streams,states,config,saved['updates'],device,
        rank,2,calibration,scales,optimizer)
    gathered=[None]*2;dist.all_gather_object(gathered,capture_streams(streams,states))
    if rank==0:
        reference=SemiFlexFlow(**saved['spec']).to(device);reference.load_state_dict(saved['model'])
        ref_opt=torch.optim.AdamW([p for p in reference.parameters() if p.requires_grad],lr=config['learning_rate'])
        ref_opt.load_state_dict(saved['optimizer'])
        rs,rt=logical_streams(config['seed'],4,0,1,device,saved['rank_states'])
        _,ref_norm=accumulated_update(reference,reference,batch,rs,rt,config,saved['updates'],device,
            0,1,calibration,scales,ref_opt)
        errors={n:float((p-reference.state_dict()[n]).abs().max()) for n,p in model.state_dict().items()}
        assert max(errors.values())<2e-6,errors
        for part in gathered:
            for slot,st in part.items():
                for key in ('noise','flow_time','rollout'):
                    assert torch.equal(st[key],capture_streams(rs,rt)[slot][key]),(slot,key)
        record=torch.load(batch['files'][0],weights_only=False,map_location='cpu')
        data=device_inputs(record,device);target=record['X_future'].to(device)
        generator=torch.Generator(device=device).manual_seed(419)
        static,history=reference.condition(data)
        noise=torch.cat([reference.source_noise(data,1,len(target),generator) for _ in range(2)])
        paths=reference.flow.integrate(noise,static,history,2)['X_gen']
        spec=importlib.util.spec_from_file_location('frozen_aligned_losses',a.old_loss)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        context=feature_context(record,device)
        kwargs=dict(coordinate_scale=reference.directional_statistics['coordinate_scale'])
        before=old.rollout_objectives(paths,target,record['geometry'],context,scales,**kwargs)
        after=rollout_objectives(paths,target,record['geometry'],context,scales,**kwargs)
        assert all(torch.equal(before[k],after[k]) for k in before)
        parameters=[p for p in reference.parameters() if p.requires_grad]
        lb=sum(w*before[k] for k,w in calibration['weights'].items())
        la=sum(w*after[k] for k,w in calibration['weights'].items())
        gb=torch.autograd.grad(lb,parameters,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(la,parameters,allow_unused=True)
        assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(gb,ga))
        assert not after['feature_es'].requires_grad
        state=initial_stopping()
        states_review=[]
        for i,score in enumerate([1.,.995,.994,.993,.992,.991,.990,.989]):
            state=update_stopping(state,score,True,str(i));states_review.append(dict(state))
        assert states_review[1]['best_score']==.995 and states_review[1]['reference_score']==1.
        assert states_review[3]['action']=='reduce_lr'
        assert states_review[-1]['action']=='eight_epoch_review'
        failures=initial_stopping()
        for i in range(3): failures=update_stopping(failures,2.,False,str(i))
        assert failures['action']=='method_review'
        report=dict(passed=True,physical_world_size=2,logical_world_size=4,
            maximum_parameter_error=max(errors.values()),global_gradient_norm=norm,serial_gradient_norm=ref_norm,
            all_four_random_streams_equal=True,diagnostic_feature_es_values_and_training_gradients_equal=True,
            test_rollout_steps=2,training_rollout_steps=32,stopping_transitions=states_review,
            production_32_64_derivatives='reused verified architecture review',files=batch['files'])
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='stopping_transitions'}),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
