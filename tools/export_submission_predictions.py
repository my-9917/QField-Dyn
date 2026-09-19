"""Export replay-verified predictions in the official tier/case file layout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from competition_io import public_cases, read_observation
from trajectory_delivery import read_xtc, coordinate_encoding_error_bound, XTC_PRECISION


def main():
    parser=argparse.ArgumentParser()
    for name in ('public-root','output','review'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predictions',type=Path,action='append',required=True)
    args=parser.parse_args();rows=[];identities=set()
    for row,meta in public_cases(args.public_root):
        name=meta['id']
        candidates=[directory for root in args.predictions for directory in (root,root/name)
                    if (directory/(name+'.pt')).exists()]
        assert len(candidates)==1,(name,'expected exactly one verified artifact',candidates)
        folder=candidates[0];artifact=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)
        assert artifact['exact_replay_passed'] and artifact['xtc_review']['passed'],name
        assert artifact['meta']==meta,name
        config=artifact['config'];assert config['resolution_tolerance_angstrom']==0.01
        identities.add((config['base_sha256'],config['adapter_sha256'],config['base_seed'],config['t4_feedback_solver']))
        blocks=artifact['blocks']
        checks=[b['resolution_checks'][-1] for b in blocks] if blocks else [artifact['resolution_checks'][-1]]
        assert all(c['passed'] and max(c['raw_rms_angstrom'],c['delivered_rms_angstrom'])<=0.01 for c in checks),name
        path=folder/(name+'_pred.xtc');data=read_xtc(path);xyz=data['coordinates_angstrom']
        assert xyz.shape==(meta['n_pred'],meta['n_atoms'],3) and np.isfinite(xyz).all(),name
        times=np.arange(meta['n_obs'],meta['n_obs']+meta['n_pred'])*meta['dt_ps']
        np.testing.assert_allclose(data['times_ps'],times,rtol=0,atol=.01)
        assert np.all(data['precisions']==10.**XTC_PRECISION),name
        np.testing.assert_array_equal(data['steps'],np.arange(meta['n_obs'],meta['n_obs']+meta['n_pred']))
        observation=read_observation(args.public_root,row,meta)
        fixed=np.concatenate([observation['protein_indices'],observation['ion_indices']])
        reference=observation['coordinates_angstrom'][-1,fixed]
        fixed_error=float(np.abs(xyz[:,fixed]-reference).max())
        encoding_bound=float(coordinate_encoding_error_bound(np.abs(reference).max()))
        assert fixed_error<=encoding_bound,(name,fixed_error,encoding_bound)
        target=args.output/meta['tier']/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        rows.append(dict(id=name,tier=meta['tier'],frames=meta['n_pred'],atoms=meta['n_atoms'],
                         first_time_ps=float(times[0]),last_time_ps=float(times[-1]),
                         exact_replay=True,maximum_fixed_atom_rounding_angstrom=fixed_error,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    assert len(identities)==1,identities
    args.review.parent.mkdir(parents=True,exist_ok=True)
    args.review.write_text(json.dumps(dict(completed=True,cases=len(rows),
        cases_by_tier={t:sum(r['tier']==t for r in rows) for t in ('T1','T2','T3','T4')},
        model_identity=list(identities)[0],rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(completed=True,cases=len(rows))))


if __name__=='__main__':main()
