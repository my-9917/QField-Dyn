"""Measure complete T4 delivery and verify its recorded resolution decisions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from competition_io import public_cases, read_observation
from semiflexible_inputs import observation_inputs
from ligand_geometry import build_geometry
from physical_quality import score_quality
from trajectory_delivery import read_xtc
from projection_encoding import encode_coordinates


def main():
    p=argparse.ArgumentParser()
    for name in ('root','prediction','public-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--resolution',action='store_true')
    p.add_argument('--allow-unreplayed',action='store_true')
    p.add_argument('--adapter',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    saved=torch.load(a.prediction/(a.case+'.pt'),map_location='cpu',weights_only=False)
    row,meta=next((r,m) for r,m in public_cases(a.public_root) if m['id']==a.case)
    assert meta['tier']=='T4' and meta['n_pred']==490 and saved['meta']==meta
    case=read_observation(a.public_root,row,meta);inputs,encoded,transform=observation_inputs(case)
    calibration=json.loads((a.root/'source/configs/ligand_geometry_calibration_v1.json').read_text())
    record=dict(meta=meta,inputs=inputs,transform=transform,geometry=build_geometry(inputs,calibration))
    path=np.asarray(saved['ligand']);assert path.shape[0]==490 and np.isfinite(path).all()
    xtc=read_xtc(a.prediction/(a.case+'_pred.xtc'))
    actual=xtc['coordinates_angstrom'][:,case['ligand_indices']]-transform['origin']
    np.testing.assert_array_equal(path,actual)
    feedback=saved['config']['t4_feedback_calibration'];assert feedback is not None
    assert saved['config']['numerical_sampling']=='per_path_final_coordinate_r1'
    encoded_again,encoded_record=encode_coordinates(saved['pre_encoding_ligand'].numpy()[None],record)
    np.testing.assert_array_equal(path,encoded_again[0])
    quality=score_quality(path[None],encoded_record,feedback)
    heavy=np.asarray(record['geometry']['heavy']);xyz=path[:,heavy]
    observed=np.asarray(inputs['X_obs'])[-1,heavy]
    centre=xyz.mean(1);start=observed.mean(0)
    step=np.linalg.norm(np.diff(np.concatenate((observed[None],xyz)),axis=0),axis=-1)
    nearest=cKDTree(np.asarray(encoded_record['geometry']['environment'])).query(xyz.reshape(-1,3))[0].reshape(490,-1)
    windows={}
    for label,indices in zip(('early','middle','late'),np.array_split(np.arange(490),3)):
        local=xyz[indices]
        windows[label]=dict(frames=len(indices),mean_atom_rmsf_angstrom=float(np.sqrt(np.square(local-local.mean(0)).sum(-1).mean(0)).mean()),
            centre_displacement_mean_angstrom=float(np.linalg.norm(centre[indices]-start,axis=-1).mean()),
            maximum_atom_step_angstrom=float(step[indices].max()),contact_frame_fraction_6A=float((nearest[indices].min(-1)<6).mean()),
            valid_frame_fraction=quality['windows'][label]['valid_frame_fraction'])
    report=dict(id=a.case,completed=True,frames=490,finite=True,dt_ps=meta['dt_ps'],
        physical_lead_time_ps=[1000,490000],Geo_accuracy=None,Dyn_reference_accuracy=None,
        accuracy_scope='future truth unavailable; long-time generation stress test',Phys=quality,Stab=windows,
        overall_mean_atom_rmsf_angstrom=float(np.sqrt(np.square(xyz-xyz.mean(0)).sum(-1).mean(0)).mean()),
        final_centre_displacement_angstrom=float(np.linalg.norm(centre[-1]-start)),
        maximum_atom_step_angstrom=float(step.max()),contact_frame_fraction_6A=float((nearest.min(-1)<6).mean()),
        xtc_review=saved['xtc_review'],exact_replay_passed=bool(saved['exact_replay_passed']),
        model_definition=saved['config'],inference_timing=saved['inference_timing'],
        feedback_all_blocks_passed=all(b['covalent_feedback']['passed'] for b in saved['blocks']))
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if not a.resolution:return
    import hashlib
    assert a.adapter is not None
    assert hashlib.sha256(a.adapter.read_bytes()).hexdigest()==saved['config']['adapter_sha256']
    tolerance=json.loads((a.root/'source/configs/selection_v1.json').read_text())['numerical_solver_rms_tolerance_angstrom']
    assert saved['exact_replay_passed'] or a.allow_unreplayed
    assert tolerance==saved['config']['resolution_tolerance_angstrom']
    rows=[]
    end=0
    for index,block in enumerate(saved['blocks']):
        assert block['start_frame']==end
        checks=block['resolution_checks'];last=checks[-1]
        assert all(not c['passed'] for c in checks[:-1])
        assert block['integration_steps']==last['coarse_steps']
        assert all(c['fine_steps']==2*c['coarse_steps'] for c in checks)
        assert checks[0]['coarse_steps']==64
        assert [c['coarse_steps'] for c in checks[1:]]==[c['fine_steps'] for c in checks[:-1]]
        assert last['passed']==(max(last['raw_rms_angstrom'],last['delivered_rms_angstrom'])<=tolerance)
        if end:
            np.testing.assert_array_equal(np.asarray(block['input_last_frame'])[0],path[end-1].astype(np.float32))
        end+=block['frames']
        rows.append(dict(block=index,start_frame=block['start_frame'],frames=block['frames'],
            accepted_steps=block['integration_steps'],raw_rms_difference_angstrom=last['raw_rms_angstrom'],
            final_rms_difference_angstrom=last['delivered_rms_angstrom'],passed=last['passed'],
            feedback_passed=block['covalent_feedback']['passed']))
    assert sum(r['frames'] for r in rows)==490
    (a.output/'resolution.json').write_text(json.dumps(dict(completed=True,id=a.case,rows=rows,
        tolerance_angstrom=tolerance,passed=all(r['passed'] for r in rows),
        exact_full_replay=bool(saved['exact_replay_passed']),
        scope='recorded block-local resolution and encoded feedback verified; full replay status reported separately; predictive accuracy requires truth'),indent=2))


if __name__=='__main__':main()
