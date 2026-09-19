"""One observation-conditioned trajectory and full labelled metrics per fixed case."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from adapter_inference import generate_adapted
from trajectory_adapter import TrajectoryAdapter
from semiflexible_model import SemiFlexFlow,device_inputs
from semiflexible_scores import score_paths
from physical_quality import score_quality
from projection_encoding import encode_coordinates
from evaluate_naive_baselines import predict,motion_scores
from sampling import case_seed


def main():
    p=argparse.ArgumentParser()
    for name in ('root','manifest','statistics','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--case-key');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads(a.manifest.read_text());assert spec['completed'] and spec['paths_per_case']==1
    assert all(not ids for ids in spec['training_exclusion']['intersections'].values())
    setup=time.perf_counter();bs=torch.load(a.root/'E3.pt',map_location='cpu',weights_only=False)
    base=SemiFlexFlow(**bs['spec']).cuda();base.load_state_dict(bs['model']);base.eval().requires_grad_(False)
    af=a.root/spec['model']['adapter_filename'];saved=torch.load(af,map_location='cpu',weights_only=False)
    adapter=TrajectoryAdapter(**saved['spec']).cuda();adapter.load_state_dict(saved['model']);adapter.eval().requires_grad_(False)
    assert hashlib.sha256((a.root/'E3.pt').read_bytes()).hexdigest()==spec['model']['base_sha256']
    assert hashlib.sha256(af.read_bytes()).hexdigest()==spec['model']['adapter_sha256']
    assert base.solver_steps==spec['solver_steps']==64
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup
    statistics=json.loads(a.statistics.read_text());assert statistics['partition']=='train'
    scale=np.asarray(statistics['scale']);calibration=json.loads((a.root/'phys_calibration.json').read_text())
    tasks=spec['tasks'][a.shard::a.shards]
    if a.case_key:tasks=[t for t in spec['tasks'] if t['key']==a.case_key];assert len(tasks)==1
    rows=[]
    for task in tasks:
        destination=a.output/(task['key']+'.pt')
        if destination.exists():
            old=torch.load(destination,map_location='cpu',weights_only=False)
            assert old['protocol']==spec and old['row']['exact_replay_passed'];rows.append(old['row']);continue
        started=time.perf_counter();loaded=torch.load(task['record'],map_location='cpu',weights_only=False)
        assert loaded['meta']['id']==task['id'] and loaded['meta']['tier']==task['tier']
        observed={k:loaded[k] for k in ('meta','inputs','transform','geometry','encoder_inputs')}
        data=device_inputs(observed,'cuda');seed=case_seed(spec['seed'],observed['meta'])
        torch.cuda.synchronize();preparation_seconds=time.perf_counter()-started
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with torch.no_grad():path,_,detail=generate_adapted(base,adapter,data,observed,seed)
        torch.cuda.synchronize();generation_seconds=time.perf_counter()-begin
        prediction=path.cpu().numpy()[None];assert np.isfinite(prediction).all()
        begin=time.perf_counter()
        with torch.no_grad():replay,_,replay_detail=generate_adapted(base,adapter,data,observed,seed)
        assert torch.equal(path,replay) and detail['resolution_checks']==replay_detail['resolution_checks']
        torch.cuda.synchronize();replay_seconds=time.perf_counter()-begin
        encoded,er=encode_coordinates(prediction,loaded);actual,_=encode_coordinates(np.asarray(loaded['X_future'])[None],loaded)
        metrics,details=score_paths(encoded,er,scale);metrics['Motion']=motion_scores(encoded,er,details)
        phys=score_quality(encoded,er,calibration,reference=actual[0]);reference_phys=score_quality(actual,er,calibration)
        baselines={};baseline_phys={}
        for method,coordinates in predict(observed['inputs']['X_obs'],80,task['n_pred'])[0].items():
            if method!='Linear':continue
            value,_=encode_coordinates(coordinates[None],loaded)
            score,scores=score_paths(value,er,scale);score['Motion']=motion_scores(value,er,scores)
            baselines[method]=score;baseline_phys[method]=score_quality(value,er,calibration,reference=actual[0])
        row=dict(id=task['id'],tier=task['tier'],evaluation_case=task['evaluation_case'],paths=1,
                 metrics=metrics,phys_v2=phys,reference_phys_v2=reference_phys,baselines=baselines,baseline_phys_v2=baseline_phys,
                 finite=True,exact_replay_passed=True,numerical_resolution_passed=detail['resolution_checks'][-1]['passed'],
                 source_record=task['record'],model_status=spec['model']['status'],
                 timing=dict(model_setup_seconds=setup_seconds,input_preparation_seconds=preparation_seconds,
                             model_generation_seconds=generation_seconds,replay_seconds=replay_seconds,
                             cold_generation_seconds=setup_seconds+preparation_seconds+generation_seconds,
                             peak_torch_allocated_memory_bytes=torch.cuda.max_memory_allocated(),
                             device=torch.cuda.get_device_name(),gpu_exclusivity=False,detail=detail['timing']),
                 seconds=time.perf_counter()-started)
        artifact=dict(row=row,paths=prediction,encoded_paths=encoded,seeds=[seed],sampling=detail,protocol=spec,source_record=task['record'])
        temporary=destination.with_suffix('.tmp');torch.save(artifact,temporary);temporary.replace(destination)
        rows.append(row)
        with (a.output/f'rows_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(prefix=task['key'],completed=len(rows),seconds=row['seconds'],feature_es=metrics['Probability']['feature_energy_score'])),flush=True)
    name='smoke_review.json' if a.case_key else f'review_{a.shard}.json'
    (a.output/name).write_text(json.dumps(dict(completed=True,rows=rows,shard=a.shard,shards=a.shards,protocol=spec),indent=2,allow_nan=False))


if __name__=='__main__':main()
