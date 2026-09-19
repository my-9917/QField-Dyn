"""Write anonymous T1-T4 files from one frozen E3 plus adapter definition."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json
import time
from pathlib import Path
import numpy as np
import torch
from trajectory_adapter import TrajectoryAdapter
from adapter_inference import generate_adapted
from semiflexible_model import SemiFlexFlow
from semiflexible_inputs import observation_inputs,assemble_prediction
from competition_io import public_cases,read_observation,write_prediction
from ligand_geometry import build_geometry
from sampling import case_seed
from trajectory_delivery import read_xtc,review_xtc
from projection_encoding import encode_coordinates
from local_covalent_solve import VERSION as FEEDBACK_SOLVER_VERSION
from t4_geometry_feedback import FeedbackSolveFailure


def main():
    p=argparse.ArgumentParser()
    for name in ('base','adapter','public-root','geometry-calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',action='append');p.add_argument('--seed',type=int,default=2026091101)
    p.add_argument('--verification-frames',type=int,default=0);p.add_argument('--verify-replay',action='store_true')
    p.add_argument('--solver-steps',type=int)
    p.add_argument('--t4-geometry-calibration',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    setup_started=time.perf_counter()
    bs=torch.load(a.base,map_location='cpu',weights_only=False);state=torch.load(a.adapter,map_location='cpu',weights_only=False)
    identity=hashlib.sha256(a.base.read_bytes()).hexdigest();assert identity==state['config']['base_checkpoint_sha256']
    base=SemiFlexFlow(**bs['spec']).cuda();base.load_state_dict(bs['model']);base.eval().requires_grad_(False)
    if a.solver_steps is not None: base.solver_steps=a.solver_steps
    adapter=TrajectoryAdapter(**state['spec']).cuda();adapter.load_state_dict(state['model']);adapter.eval().requires_grad_(False)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup_started
    config=dict(base_sha256=identity,adapter_sha256=hashlib.sha256(a.adapter.read_bytes()).hexdigest(),adapter_config=state['config'],
        solver_steps=base.solver_steps,base_seed=a.seed,processing_paths=1,output_member=0,
        numerical_sampling='per_path_final_coordinate_r1',resolution_tolerance_angstrom=0.01,
        verification_frames=a.verification_frames,chemical_definition='anonymous observed connectivity and geometry',
        scope='partial-frame numerical verification' if a.verification_frames else 'observation-conditioned trajectory generation')
    config['runtime']=dict(gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,model_setup_seconds=setup_seconds,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'),
        timing_conditions='wall-clock latency under the recorded experiment schedule; GPU exclusivity not assumed')
    feedback=json.loads(a.t4_geometry_calibration.read_text()) if a.t4_geometry_calibration else None
    config['t4_feedback_calibration']=feedback
    config['t4_feedback_solver']=FEEDBACK_SOLVER_VERSION if feedback is not None else None
    (a.output/'run_config.json').write_text(json.dumps(config,indent=2));rows=[]
    calibration=json.loads(a.geometry_calibration.read_text())
    for row,meta in public_cases(a.public_root):
        if a.case and meta['id'] not in a.case:continue
        case_started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        case=read_observation(a.public_root,row,meta)
        if a.verification_frames:
            assert meta['tier']=='T4' and 0<a.verification_frames<=meta['n_pred']
            meta=dict(meta,n_pred=a.verification_frames);case=dict(case,meta=meta)
        inputs,encoded,transform=observation_inputs(case);data={k:v.cuda() for k,v in encoded.items()}
        record=dict(meta=meta,inputs=inputs,transform=transform,geometry=build_geometry(inputs,calibration))
        seed=case_seed(a.seed,meta)
        torch.cuda.synchronize();preparation_seconds=time.perf_counter()-case_started
        def progress(x):
            print(json.dumps({k:v for k,v in x.items() if k not in ('input_last_frame','output_last_frame')}),flush=True)
        begin=time.perf_counter()
        try:
            ligand,blocks,details=generate_adapted(base,adapter,data,record,seed,progress=progress,feedback_calibration=feedback)
        except FeedbackSolveFailure as failure:
            torch.save(dict(**failure.payload,config=config,seed=seed),a.output/(meta['id']+'.failure.pt'))
            (a.output/'failure.json').write_text(json.dumps(dict(id=meta['id'],detail=failure.payload['detail'],config=config),indent=2))
            raise
        torch.cuda.synchronize();generation_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        xyz=assemble_prediction(case,ligand.cpu().numpy(),transform['origin']);path=write_prediction(case,xyz,a.output)
        serialization_seconds=time.perf_counter()-begin;begin=time.perf_counter()
        box=case['trajectory'].trajectory.ts.triclinic_dimensions;review=review_xtc(path,xyz,meta,np.zeros((3,3)) if box is None else box)
        delivered=read_xtc(path)['coordinates_angstrom'][:,case['ligand_indices']]-transform['origin']
        expected,_=encode_coordinates(ligand.cpu().numpy()[None],record);np.testing.assert_array_equal(delivered,expected[0])
        for block in blocks:np.testing.assert_array_equal(delivered[block['start_frame']+block['frames']-1],block['output_last_frame'])
        timing=dict(input_preparation_seconds=preparation_seconds,model_generation_seconds=generation_seconds,
            file_write_seconds=serialization_seconds,single_case_inference_seconds=preparation_seconds+generation_seconds+serialization_seconds,
            cold_start_inference_seconds=setup_seconds+preparation_seconds+generation_seconds+serialization_seconds,
            exact_replay_verification_seconds=0.,output_validation_seconds=time.perf_counter()-begin,
            peak_torch_allocated_memory_bytes=torch.cuda.max_memory_allocated(),
            processing_paths=1,frames=meta['n_pred'],frame_count_per_second=meta['n_pred']/(preparation_seconds+generation_seconds+serialization_seconds),
            model_setup_seconds=setup_seconds,scope='complete uncached generation; replay and scoring excluded from inference total')
        stages=details['timing']
        samples=[b['timing']['sampling'] for b in blocks] if blocks else [stages['sampling']]
        timing['conditioning_seconds']=stages['conditioning_seconds']+sum(b['timing']['recurrent_conditioning_seconds'] for b in blocks)
        for name in ('flow_seconds','adapter_seconds','feedback_with_encoding_seconds','encoding_seconds'):
            timing[name]=sum(s[name] for s in samples)
        timing['maximum_accepted_integration_steps']=max(b['integration_steps'] for b in blocks) if blocks else details['integration_steps']
        artifact=dict(meta=meta,seed=seed,ligand=delivered,pre_encoding_ligand=ligand.cpu(),blocks=blocks,config=config,
            exact_replay_passed=False,
            inference_timing=timing,
            xtc_review=review,transform=transform,**details)
        torch.save(artifact,a.output/(meta['id']+'.generated.pt'))
        print(json.dumps(dict(id=meta['id'],stage='complete_generation_saved',timing=timing)),flush=True)
        if a.verify_replay:
            begin=time.perf_counter()
            replay,replay_blocks,replay_details=generate_adapted(base,adapter,data,record,seed,feedback_calibration=feedback)
            assert torch.equal(ligand,replay)
            assert torch.equal(details['raw_ligand'],replay_details['raw_ligand'])
            if blocks:
                assert [(b['integration_steps'],b['resolution_checks']) for b in blocks]==[(b['integration_steps'],b['resolution_checks']) for b in replay_blocks]
            else:
                assert details['integration_steps']==replay_details['integration_steps'] and details['resolution_checks']==replay_details['resolution_checks']
            torch.cuda.synchronize();timing['exact_replay_verification_seconds']=time.perf_counter()-begin
            artifact['exact_replay_passed']=True
        torch.save(artifact,a.output/(meta['id']+'.pt'))
        rows.append(dict(id=meta['id'],tier=meta['tier'],timing=timing,**review));print(json.dumps(rows[-1]),flush=True)
    assert rows
    (a.output/'review.json').write_text(json.dumps(dict(completed=True,rows=rows,config=config,numerical_resolution_review='recorded_per_path_and_block'),indent=2))


if __name__=='__main__':main()
