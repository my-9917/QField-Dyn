"""Uploaded observations use the same frozen CLI as the research delivery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import MDAnalysis as mda

TIERS={'T1':(10,10,80),'T2':(80,20,80),'T3':(20,80,80),'T4':(10,490,1000)}


def execute(job, config):
    state=json.loads((job/'status.json').read_text()); started=time.time()
    def save(**changes):
        state.update(changes); temporary=job/'status.tmp'
        temporary.write_text(json.dumps(state,ensure_ascii=False));temporary.replace(job/'status.json')
    save(status='validating',started_unix=started)
    try:
        tier=state['tier'];k,h,dt=TIERS[tier];identifier='upload-'+job.name[:8]
        case_dir=job/'input'/tier/identifier;case_dir.mkdir(parents=True)
        for old,new in [('topology.pdb','top.pdb'),('observations.xtc','obs.xtc')]:
            (job/old).replace(case_dir/new)
        observed=mda.Universe(str(case_dir/'top.pdb'),str(case_dir/'obs.xtc'))
        assert len(observed.trajectory)==k,f'{tier} requires {k} observed frames'
        assert np.isfinite(observed.atoms.positions).all(),'Coordinates must be finite'
        spec=dict(n_obs=k,n_pred=h,dt_ps=dt,ligand_resname=state['ligand'])
        meta=dict(spec,id=identifier,tier=tier,n_atoms=len(observed.atoms),obs_index_0based=[0,k],pred_index_0based=[k,k+h])
        (case_dir/'meta.json').write_text(json.dumps(meta))
        row=dict(spec,id=identifier,top=f'{tier}/{identifier}/top.pdb',obs=f'{tier}/{identifier}/obs.xtc')
        (job/'input'/tier/'manifest.jsonl').write_text(json.dumps(row)+'\n')
        (job/'input/protocol.json').write_text(json.dumps(dict(tiers={tier:dict(spec,n_systems=1)})))
        output=job/'output';output.mkdir()
        cmd=[sys.executable,str(Path(config['runtime'])/'predict_trajectory_adapter.py'),
             '--base',config['base'],'--adapter',config['adapter'],'--public-root',str(job/'input'),
             '--geometry-calibration',config['geometry_calibration'],'--t4-geometry-calibration',config['phys_calibration'],
             '--output',str(output),'--case',identifier]
        save(status='running',model=config['model_label'],model_status=config['model_status'])
        with (job/'inference.log').open('w') as stream:
            subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(config['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        review=json.loads((output/'review.json').read_text())
        assert review['completed'] and review['rows'][0]['passed']
        predicted=mda.Universe(str(case_dir/'top.pdb'),str(output/(identifier+'_pred.xtc')))
        ligand=predicted.atoms[predicted.atoms.resnames==state['ligand']]
        assert len(ligand)>0
        xyz=np.asarray([ligand.positions.copy() for _ in predicted.trajectory],dtype=float)
        observed.trajectory[-1];original=observed.atoms[ligand.indices].positions.copy()
        ca=observed.select_atoms('protein and name CA')
        reverse={int(v):i for i,v in enumerate(ligand.indices)}
        bonds=[[reverse[int(a)],reverse[int(b)]] for a,b in observed.atoms.bonds.indices if int(a) in reverse and int(b) in reverse]
        centres=xyz.mean(1);drift=np.linalg.norm(centres-original.mean(0),axis=-1)
        rg=np.sqrt(np.square(xyz-centres[:,None]).sum(-1).mean(-1))
        payload=dict(id=identifier,tier=tier,frames=h,dt_ps=dt,first_time_ps=k*dt,
            coordinates=xyz.tolist(),observed_last=original.tolist(),elements=ligand.elements.tolist(),bonds=bonds,
            protein_ca=ca.positions.tolist(),protein_residue_indices=ca.resindices.tolist(),
            centre_displacement_angstrom=drift.tolist(),radius_of_gyration_angstrom=rg.tolist(),
            model=config['model_label'],model_status=config['model_status'],
            inference_seconds=review['rows'][0]['timing']['cold_start_inference_seconds'],
            future_ground_truth_available=False)
        (job/'result.json').write_text(json.dumps(payload,separators=(',',':')))
        save(status='completed',finished_unix=time.time(),elapsed_seconds=time.time()-started,output_filename=identifier+'_pred.xtc')
    except Exception as error:
        save(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        import traceback
        (job/'error.txt').write_text(traceback.format_exc())
