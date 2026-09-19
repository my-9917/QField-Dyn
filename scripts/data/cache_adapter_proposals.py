"""Cache frozen E3 draws and conditions for the structural adapter only once."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from semiflexible_model import SemiFlexFlow,device_inputs
from sampling import case_seed


def main():
    p=argparse.ArgumentParser()
    for key in ('manifest','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scope',choices=('probe','train'),required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads(a.manifest.read_text());cfg=m['config']
    files=[v['file'] for v in m['probe']] if a.scope=='probe' else [f for r in m['train'] for f in r['records']]
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=SemiFlexFlow(**saved['spec']).cuda();model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    assert model.solver_steps==64
    identity=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();rows=[]
    completed=set()
    for log in a.output.glob('progress_*.jsonl'):
        for line in log.read_text().splitlines():
            if line.endswith('}'):completed.add(json.loads(line)['id'])
    for file in files[a.shard::a.shards]:
        start=time.monotonic();record=torch.load(file,map_location='cpu',weights_only=False)
        assert record['meta']['partition']=='train'
        destination=a.output/(Path(file).stem+'.pt')
        seeds=[case_seed(cfg['generation_seed']+i,record['meta']) for i in range(cfg['paths_per_training_prefix'])]
        if Path(file).stem in completed:
            cache=torch.load(destination,map_location='cpu',weights_only=False)
            assert cache['checkpoint_sha256']==identity and cache['seeds']==seeds
            rows.append(dict(id=Path(file).stem,reused=True));continue
        data=device_inputs(record,'cuda')
        with torch.no_grad():
            static,q=model.condition(data);paths=[]
            for seed in seeds:
                noise=model.source_noise(data,1,record['meta']['n_pred'],torch.Generator(device='cuda').manual_seed(seed))
                paths.append(model.flow.sample(noise,static,q,model.solver_steps)['X_gen'].cpu())
        paths=torch.cat(paths);assert torch.isfinite(paths).all()
        payload=dict(meta=record['meta'],paths=paths,source_record=file,seeds=seeds,
            checkpoint_sha256=identity,source_version='structured_adapter_0919_v1',
            atom_names=list(record['inputs']['ligand_graph']['atom_names']),
            ligand_features=static['ligand_features'].detach().cpu(),quantum_condition=q.detach().cpu(),
            coordinate_version=record['input_version'],frozen_base=True)
        temporary=destination.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(destination)
        row=dict(id=Path(file).stem,seconds=time.monotonic()-start,frames=paths.shape[0]*paths.shape[1],finite=True)
        rows.append(row)
        with (a.output/f'progress_{a.scope}_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    (a.output/f'review_{a.scope}_{a.shard}.json').write_text(json.dumps(dict(completed=True,rows=rows,
        checkpoint_sha256=identity,scope=a.scope,shard=a.shard,shards=a.shards),indent=2))


if __name__=='__main__':main()
