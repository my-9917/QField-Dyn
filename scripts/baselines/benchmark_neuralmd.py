"""Author NeuralMD cold/warm inference accounting on actual public observations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from neuralmd_author import load_author_model,predict
from competition_io import public_cases,read_observation
from semiflexible_inputs import observation_inputs
from trajectory_delivery import write_xtc,read_xtc


def main():
    p=argparse.ArgumentParser();p.add_argument('--public-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();model=load_author_model('cuda');torch.cuda.synchronize();setup=time.perf_counter()-start
    rows=[];seen=set()
    for row,meta in public_cases(a.public_root):
        tier=meta['tier']
        if tier=='T4' or tier in seen:continue
        seen.add(tier);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        case=read_observation(a.public_root,row,meta);inputs,_,transform=observation_inputs(case)
        prepared=time.perf_counter();coordinates,heavy=predict(model,inputs,meta['n_pred'],'cuda');torch.cuda.synchronize();generated=time.perf_counter()
        coordinates+=transform['origin'];assert np.isfinite(coordinates).all()
        frames=np.arange(meta['n_obs'],meta['n_obs']+meta['n_pred']);filename=a.output/(meta['id']+'_heavy.xtc')
        box=case['trajectory'].trajectory.ts.triclinic_dimensions
        write_xtc(filename,coordinates,frames*meta['dt_ps'],frames,np.zeros((3,3)) if box is None else box)
        written=time.perf_counter();decoded=read_xtc(filename)
        np.testing.assert_allclose(decoded['coordinates_angstrom'],coordinates,rtol=0,atol=.001)
        rows.append(dict(id=meta['id'],tier=tier,method='NeuralMD author ODE pretrained transfer',frames=meta['n_pred'],
            predicted_atoms=int(heavy.sum()),atom_scope='ligand heavy atoms only',device=torch.cuda.get_device_name(),
            input_preparation_seconds=prepared-start,model_generation_seconds=generated-prepared,
            file_write_seconds=written-generated,model_setup_seconds=setup,
            single_case_inference_seconds=written-start,cold_start_inference_seconds=setup+written-start,
            peak_torch_allocated_memory_bytes=torch.cuda.max_memory_allocated(),gpu_exclusivity=False,
            processing_paths=1,numerical_protocol='author Euler step 0.1; no adaptive precision or geometric postprocessor',
            residue_normalization='HIS maps to author HIE when observed HE2 is bonded to NE2 and HD1 is absent'))
        print(json.dumps(rows[-1]),flush=True)
    (a.output/'review.json').write_text(json.dumps(dict(completed=True,rows=rows,parameters=sum(p.numel() for p in model.parameters()),
        comparison_scope='same A800 model and actual observations; output atom scope and numerical algorithms differ'),indent=2))


if __name__=='__main__':main()
