"""Freeze 2048 continuation systems and three disjoint 128-system rollout sets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from training_schedule import rotating_systems,fixed_rollout_subsets
from training_state import validate_cache_expansion


def pocket_descriptor(row):
    torch.set_num_threads(1)
    record=torch.load(row['prefixes'][0]['file'],weights_only=False,map_location='cpu')
    assert record['meta']['tier']=='T1' and record['meta']['partition']=='train'
    inputs=record['inputs'];top=inputs['protein_topology'];z=np.asarray(top['atomic_numbers'])
    protein=np.asarray(inputs['P0']);residue=np.asarray(top['residue_indices'])
    ligand=np.asarray(inputs['X_obs'])[:,np.asarray(inputs['ligand_graph']['atomic_numbers'])>1].reshape(-1,3)
    pocket_residues=np.unique(residue[(z>1)&(cKDTree(ligand).query(protein)[0]<=6.)])
    mask=(z>1)&np.isin(residue,pocket_residues);xyz=protein[mask];numbers=z[mask]
    return dict(id=row['id'],pocket_residues=len(pocket_residues),pocket_atoms=len(xyz),
        pocket_radius=float(np.sqrt(np.square(xyz-xyz.mean(0)).sum(-1).mean())) if len(xyz) else 0.,
        pocket_polar_fraction=float(np.isin(numbers,[7,8,16]).mean()) if len(xyz) else 0.)


def main():
    p=argparse.ArgumentParser()
    for name in ('population','previous-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    population=json.loads(a.population.read_text());assert population['systems']==9846
    manifest=json.loads((Path(population['cache'])/'manifest.json').read_text())
    previous=json.loads((a.previous_cache/'manifest.json').read_text())
    prior={r['id'] for r in previous['rows'] if r['partition']=='train'};assert len(prior)==512
    rows=population['rows'];assert prior<={r['id'] for r in rows}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,description in enumerate(pool.map(pocket_descriptor,rows,chunksize=8)):
            assert description['id']==rows[i]['id'];rows[i].update(description)
            if (i+1)%512==0:print(json.dumps(dict(pocket_metadata_complete=i+1)),flush=True)
    cuts={key:np.quantile([r[key] for r in rows],[1/3,2/3]).tolist()
          for key in ('pocket_residues','pocket_radius','pocket_polar_fraction')}
    for row in rows:
        row['pocket_group']=':'.join(str(np.searchsorted(values,row[key])) for key,values in cuts.items())
    core,_=rotating_systems(rows,{},2048,202609170501,initial=[r for r in rows if r['id'] in prior])
    selected={r['id'] for r in core};assert len(selected)==2048 and prior<=selected
    validation=[r for r in manifest['rows'] if r['partition']=='validation']
    assert selected.isdisjoint(r['id'] for r in validation)
    config=dict(manifest['config'],version='core2048_aligned_0917v5',
        systems=dict(train=2048,validation=len(validation)),selected_members=sorted(selected),source_cache=population['cache'])
    cache=a.output/'cache';cache.mkdir()
    core_manifest=dict(manifest,config=config,rows=[r for r in manifest['rows'] if r['id'] in selected or r['partition']=='validation'])
    transition=validate_cache_expansion(previous,core_manifest)
    (cache/'manifest.json').write_text(json.dumps(core_manifest,indent=2))
    subsets=fixed_rollout_subsets(core,128,[202609170510+i for i in range(3)])
    assert len(set(sum(subsets,[])))==384
    payload=dict(population,systems=2048,prefixes=6144,rows=core,cache=str(cache),pocket_thresholds=cuts)
    (a.output/'population.json').write_text(json.dumps(payload))
    report=dict(completed=True,version='core_training_0917v5',cache=str(cache),population=str(a.output/'population.json'),
        reservoir_systems=9846,core_systems=2048,core_prefixes=6144,selected_systems=sorted(selected),
        mandatory_prior_aligned_systems=sorted(prior),rollout_systems_per_epoch=128,rollout_subsets=subsets,
        rotation_seeds=[202609170510+i for i in range(3)],next_rotation_seed=202609170513,
        core_strata=dict(Counter(r['stratum'] for r in core)),
        rollout_strata=[dict(Counter(r['stratum'] for r in core if r['id'] in ids)) for ids in subsets],
        sequence_identity_groups=len({r['protein_group'] for r in core}),scaffolds=len({r['scaffold'] for r in core}),
        pocket_groups=len({r['pocket_group'] for r in core}),cache_transition=transition,
        training_provenance='9846-system CFM pretraining, prior 512-system aligned training, then fixed 2048-system continuation',
        independence='three correlated task prefixes per system; sequence/pocket/scaffold balancing supports coverage',
        protocol=dict(cfm_prefixes_per_epoch=6144,rollout_prefixes_per_epoch=384,updates_per_epoch=1536,
                      rollout_updates_per_epoch=96,rollout_update_spacing=16,learning_rates=[1e-4,1e-4,3e-5]))
    (a.output/'definition.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('selected_systems','mandatory_prior_aligned_systems','rollout_subsets')}),flush=True)


if __name__=='__main__':main()
