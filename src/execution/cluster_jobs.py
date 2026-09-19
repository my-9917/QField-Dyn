"""Register evaluation groups and their canonical two-shard output layout."""
import json
from pathlib import Path
import shutil
from compare_generation import collect
from semiflexible_scores import VERSION


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def register(central,name,checkpoint,cache,paths,priority,role):
    central=Path(central);definition=central/'manifests'/(name+'.json')
    if definition.exists():
        spec=json.loads(definition.read_text())
        assert spec['paths']==paths and spec['cache']==str(cache)
    else:
        raw=central/'raw'/name;raw.mkdir()
        for shard in range(2):
            folder=raw/f'shard_{shard}';folder.mkdir()
            shutil.copy2(checkpoint,folder/'checkpoint.pt')
        spec=dict(name=name,raw=str(raw),checkpoint=str(checkpoint),cache=str(cache),paths=paths,shards=2,
            priority=priority,role=role,active=True)
    spec.update(priority=priority,role=role,active=True)
    write_json(definition,spec)
    registry=central/'manifests/evaluation_jobs.json'
    jobs=json.loads(registry.read_text()) if registry.exists() else dict(jobs=[],closed=False)
    if name not in jobs['jobs']:jobs['jobs'].append(name)
    write_json(registry,jobs)
    return spec


def prefix_tasks(spec):
    members=[r for r in json.loads((Path(spec['cache'])/'manifest.json').read_text())['rows'] if r['partition']=='validation']
    return [dict(key=row['id']+'_'+tier,record=record,shard=i%spec['shards'],tier=tier)
        for i,row in enumerate(members) for tier,record in zip(('T1','T2','T3'),row['records'])]


def finalize_raw(spec,statistics,selection,solver_steps):
    raw=Path(spec['raw']);manifest=json.loads((Path(spec['cache'])/'manifest.json').read_text())
    tasks=prefix_tasks(spec)
    for shard in range(spec['shards']):
        folder=raw/f'shard_{shard}'
        rows=[json.loads(line) for line in (folder/'rows.jsonl').read_text().splitlines()]
        assert {r['id']+'_'+r['tier'] for r in rows}=={t['key'] for t in tasks if t['shard']==shard}
        rows.sort(key=lambda r:(r['id'],r['tier']))
        write_json(folder/'review.json',dict(completed=True,rows=rows,expected_prefixes=len(rows),
            checkpoint=str(folder/'checkpoint.pt'),statistics=statistics,seed=selection['evaluation_seed'],
            shard=shard,shards=spec['shards'],paths_per_prefix=spec['paths'],solver_steps=solver_steps,
            evaluator_version=VERSION,path_batch_size=1))
    write_json(raw/'exit_codes.json',[0]*spec['shards'])
    result=collect(raw,manifest,spec['shards'],selection,spec['paths'])
    write_json(raw/'review.json',dict(completed=True,**result))
