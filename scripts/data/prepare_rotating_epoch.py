"""Freeze the next 2048-CFM / 128-rollout epoch after its measured input gates."""
import argparse
import json
import random
from pathlib import Path
import torch
from history_calibration import file_digest
from training_schedule import pack_batches,initial_stopping,balanced_rollout_batches,interleave_rollout,rotating_systems


def epoch_plan(definition, population, manifest, stopping):
    epoch=stopping['epochs']+1
    if epoch>1:assert stopping['action']=='continue',stopping['action']
    members=[r for r in manifest['rows'] if r['partition']=='train']
    assert len(members)==2048 and definition['completed'] and population['completed']
    assert {r['id'] for r in members}==set(definition['selected_systems'])=={r['id'] for r in population['rows']}
    subsets=definition['rollout_subsets'];assert len(set(sum(subsets,[])))==384
    used=stopping.get('rollout_exposure_counts',{})
    seed=definition['rotation_seeds'][0]+epoch-1
    if epoch<=3:
        identifiers=set(subsets[epoch-1])
        counts=dict(used)
        for identifier in identifiers:counts[identifier]=counts.get(identifier,0)+1
    else:
        selected,counts=rotating_systems(population['rows'],used,128,seed)
        identifiers={row['id'] for row in selected}
    assert len(identifiers)==128
    lookup={(r['id'],t):f for r in members for t,f in zip(('T1','T2','T3'),r['records'])}
    rows=[dict(p,file=lookup[row['id'],p['tier']]) for row in population['rows'] for p in row['prefixes']]
    rng=random.Random(seed)
    costly=balanced_rollout_batches([r for r in rows if r['id'] in identifiers],rng)
    cheap=pack_batches([r for r in rows if r['id'] not in identifiers],False,rng)
    batches=interleave_rollout(cheap,costly)
    files=[f for b in batches for f in b['files']]
    positions=[i for i,b in enumerate(batches) if b['rollout']]
    assert len(files)==len(set(files))==6144 and len(batches)==1536
    assert len(positions)==96 and {b-a for a,b in zip(positions,positions[1:])}=={16}
    sampling=dict(epoch=epoch,prefix_order=files,rollout_subset=sorted(identifiers),
        expensive_update_positions=[i+1 for i in positions],
        next_rotation_seed=seed+1,exposure_counts=counts)
    return dict(version='aligned_training_plan_0918',kind='complete_epoch',logical_world_size=4,
        continuation_epoch=epoch,cfm_systems=2048,cfm_prefixes=6144,rollout_systems=128,rollout_prefixes=384,
        sampling_state=sampling,selected_systems=sorted(identifiers),
        learning_rate=1e-4 if epoch<=2 else 3e-5,stopping_controller=stopping,batches=batches,
        supervision='each CFM prefix once; every 16th update adds 32-step double-path rollout',
        training_provenance=definition['training_provenance'],independent_set='sealed')


def main():
    p=argparse.ArgumentParser()
    for key in ('core-definition','checkpoint','loss-calibration','throughput-review','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stopping-state',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    definition=json.loads(a.core_definition.read_text())
    population=json.loads(Path(definition['population']).read_text())
    manifest=json.loads((Path(definition['cache'])/'manifest.json').read_text())
    calibration=json.loads(a.loss_calibration.read_text())
    assert calibration['completed'] and calibration['partition']=='train' and calibration['passed']
    assert calibration['method']=='per_update_adamw_contribution_0918'
    assert calibration['population_systems']==2048 and calibration['rollout_systems']==128
    throughput=json.loads(a.throughput_review.read_text())
    assert throughput['completed'] and throughput['projected_epoch_hours']<=8
    assert throughput['cfm_prefixes']==6144 and throughput['rollout_prefixes']==384
    saved=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
    assert saved['checkpoint_kind'] in ('complete_epoch','stage_boundary')
    stopping=json.loads(a.stopping_state.read_text()) if a.stopping_state else initial_stopping()
    if stopping['epochs']==0:
        assert calibration['checkpoint_sha256']==file_digest(a.checkpoint)
        assert throughput['checkpoint_sha256']==calibration['checkpoint_sha256']
    else:
        assert saved['training_plan']['continuation_epoch']==stopping['epochs']
        assert stopping['reviewed_checkpoint']==str(a.checkpoint)
    report=epoch_plan(definition,population,manifest,stopping)
    report.update(loss_calibration=str(a.loss_calibration),core_definition=str(a.core_definition),
        throughput_review=str(a.throughput_review),loss_calibration_sha256=file_digest(a.loss_calibration))
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('batches','sampling_state','selected_systems')}))


if __name__=='__main__':main()
