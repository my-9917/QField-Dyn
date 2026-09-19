"""Compare real 32-step training through an optimizer-update save/resume boundary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from training_checkpoint import state_difference
from training_schedule import initial_stopping


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','plan','cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    original=json.loads(a.plan.read_text())
    chosen=sorted((r for r in original['selected'] if r['tier']=='T1'),key=lambda r:r['atoms'])[:16]
    used={r['file'] for r in chosen}
    rows=chosen+[r for r in original['selected'] if r['file'] not in used]
    plan=dict(original,batches=[dict(files=[r['file'] for r in rows[i:i+4]],rollout=True) for i in range(0,256,4)])
    plan['sampling_state']=dict(prefix_order=[r['file'] for r in rows],
        expensive_update_positions=list(range(1,65)),rollout_subset=sorted({r['id'] for r in rows}),next_rotation_seed=202609170510)
    plan['stopping_controller']=initial_stopping()
    (a.output/'plan.json').write_text(json.dumps(plan))
    saved=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
    (a.output/'config.json').write_text(json.dumps(saved['config']))
    base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(source/'train_semiflexible.py'),'--config',str(a.output/'config.json'),
        '--training-plan',str(a.output/'plan.json'),'--cache',str(a.cache)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for name,end,checkpoint in [('continuous',4,a.checkpoint),('first',2,a.checkpoint),('resumed',4,None)]:
        if name=='resumed':
            rows=[json.loads(x) for x in (a.output/'first/snapshots.jsonl').read_text().splitlines()]
            checkpoint=Path(rows[-1]['path'])
        with (a.output/(name+'.log')).open('x') as log:
            result=subprocess.run([*base,'--resume',str(checkpoint),'--output',str(a.output/name),
                '--replay-through',str(end),'--monitor-updates','16'],env=env,stdout=log,stderr=subprocess.STDOUT)
        (a.output/(name+'.exit_code.txt')).write_text(str(result.returncode))
        assert result.returncode==0,name
    states=[]
    for name in ('continuous','resumed'):
        rows=[json.loads(x) for x in (a.output/name/'snapshots.jsonl').read_text().splitlines()]
        states.append(torch.load(rows[-1]['path'],weights_only=False,map_location='cpu'))
    left,right=states
    errors={key:state_difference(left[key],right[key]) for key in
        ('model','optimizer','scheduler','rank_states','shuffle_state','rollout_sampling_state','stopping_controller','scientific_configuration')}
    assert errors['model']<2e-6 and errors['optimizer']<2e-6,errors
    assert errors['rank_states']==errors['shuffle_state']==0
    assert left['updates']==right['updates']==saved['updates']+4
    for state in states:
        assert state['training_progress']['next_batch']==4
        assert state['training_progress']['processed_prefixes']==16
        assert state['exposures']['cfm']==16 and state['exposures']['rollout']==16
        assert state['exposures']['rollout_by_tier']==dict(T1=16,T2=0,T3=0)
    report=dict(passed=True,physical_gpus=[2,3],rollout_steps=32,paths=2,
        continuous_updates=4,resume_after=2,maximum_errors=errors,
        snapshot_contains=['model','optimizer','scheduler','rank_states','shuffle_state','training_progress','training_plan',
            'history_calibration','rollout_sampling_state','stopping_controller','scientific_configuration'])
    (a.output/'review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
