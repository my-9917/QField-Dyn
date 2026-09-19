"""Export all 90 observed and predicted trajectories with truth and atom identities."""
import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
import torch
from trajectory_delivery import write_xtc,read_xtc,review_xtc


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case-key');a=p.parse_args()
    manifest=json.loads((a.root/'manifest.json').read_text());rows=[]
    tasks=[t for t in manifest['tasks'] if t['key']==a.case_key] if a.case_key else manifest['tasks']
    for task in tasks:
        saved=torch.load(a.root/'qmem'/(task['key']+'.pt'),map_location='cpu',weights_only=False)
        record=torch.load(task['record'],map_location='cpu',weights_only=False);inputs=record['inputs']
        graph,protein,ions=[inputs[k] for k in ('ligand_graph','protein_topology','ion_topology')]
        li,pi,ii=[np.asarray(t['source_atom_indices'],dtype=int) for t in (graph,protein,ions)]
        n=len(li)+len(pi)+len(ii);origin=np.asarray(record['transform']['origin'])
        template=np.empty((n,3));template[pi]=np.asarray(inputs['P0'])+origin;template[ii]=np.asarray(inputs['I0'])+origin
        template[li]=np.asarray(inputs['X_obs'])[-1]+origin
        destination=a.root/'trajectories'/(task['evaluation_case']+'_'+task['id']);destination.mkdir(parents=True,exist_ok=True)
        meta=dict(record['meta'],n_atoms=n)
        for name,ligand,start in [('obs',np.asarray(inputs['X_obs']),0),
                ('pred',saved['paths'][0],meta['n_obs']),('truth',np.asarray(record['X_future']),meta['n_obs'])]:
            coordinates=np.broadcast_to(template,(len(ligand),n,3)).copy();coordinates[:,li]=ligand+origin
            steps=np.arange(start,start+len(ligand));filename=destination/(name+'.xtc')
            write_xtc(filename,coordinates,steps*meta['dt_ps'],steps,np.zeros((3,3)))
            if name=='pred':
                review=review_xtc(filename,coordinates,meta,np.zeros((3,3)))
                decoded=read_xtc(filename)['coordinates_angstrom'][:,li]-origin
                np.testing.assert_array_equal(decoded,saved['encoded_paths'][0])
        names=np.empty(n,dtype=object);resnames=np.empty(n,dtype=object);resids=np.empty(n,dtype=int);numbers=np.empty(n,dtype=int)
        names[pi]=protein['atom_names'];resnames[pi]=protein['residue_names'];resids[pi]=np.asarray(protein['residue_indices'])+1
        names[li]=graph['atom_names'];resnames[li]='MOL';resids[li]=resids[pi].max()+1
        for indices,topology in [(pi,protein),(li,graph),(ii,ions)]:numbers[indices]=np.asarray(topology['atomic_numbers'])
        table=Chem.GetPeriodicTable()
        for j,i in enumerate(ii):names[i]=table.GetElementSymbol(int(numbers[i]));resnames[i]=str(names[i]).upper();resids[i]=resids[li][0]+j+1
        template[li]=np.asarray(inputs['X_obs'])[0]+origin
        protein_set=set(pi);lines=[]
        for i,(name,residue,resid,z,xyz) in enumerate(zip(names,resnames,resids,numbers,template)):
            kind='ATOM' if i in protein_set else 'HETATM';chain='A' if i in protein_set else 'B'
            element=table.GetElementSymbol(int(z));x,y,zcoord=xyz
            lines.append(f'{kind:<6}{i+1:5d} {str(name):>4s} {str(residue):>3s} {chain}{resid:4d}    {x:8.3f}{y:8.3f}{zcoord:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}')
        for first,second in np.asarray(graph['bonds'],dtype=int):lines.append(f'CONECT{li[first]+1:5d}{li[second]+1:5d}')
        (destination/'top.pdb').write_text('\n'.join(lines+['END'])+'\n')
        metadata=dict(task=task,model=manifest['model'],seed=saved['seeds'][0],ligand_resname='MOL',
            coordinate_frame='protein-aligned observed reference; fixed protein environment',units='Angstrom; ps',
            truth_scope='native ligand future after per-frame protein alignment',numerical=saved['sampling']['resolution_checks'],
            exact_replay_passed=saved['row']['exact_replay_passed'],file_review=review)
        (destination/'meta.json').write_text(json.dumps(metadata,indent=2));rows.append(dict(id=task['id'],tier=task['tier'],directory=destination.name,**review))
    review_name='smoke_review.json' if a.case_key else 'review.json'
    (a.root/'trajectories'/review_name).write_text(json.dumps(dict(completed=True,cases=len(rows),rows=rows),indent=2))
    print(json.dumps(dict(completed=True,files=len(rows),T4='awaiting qualified truth')))


if __name__=='__main__':main()
