"""Check repaired T4 feedback precision, exact replay and geometric symmetries."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from local_covalent_solve import solve_local
from covalent_output_constraints import output_constraints
from covalent_output_solver import solve_output_path
from projection_encoding import encode_coordinates


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    saved=torch.load(a.trace/'context.pt',map_location='cpu',weights_only=False)
    record=saved['record'];context=saved['geometry'];results={};reviews=[]
    solver_context={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in context.items()}
    for steps in (128,256):
        old=torch.load(a.trace/f'pre_feedback_{steps}.pt',map_location='cpu',weights_only=False)
        path=old['stages']['adapter_final'];result,detail=solve_output_path(path,record,solver_context)
        assert detail['passed'],detail
        encoded,_=encode_coordinates(result[None],record);results[steps]=encoded
        torch.save(dict(path=result,encoded=encoded,detail=detail),a.output/f'steps_{steps}.pt')
        reviews.append(dict(steps=steps,feedback=detail))
        if steps==128:
            repeated,replay=solve_output_path(path,record,solver_context)
            np.testing.assert_array_equal(result,repeated)
            centres=(path*context['weights'].numpy()[:,None]).sum(-2)
            g=output_constraints(context,record,centres[np.abs(centres).max(-1).argmax()])
            frame=3;warm=result[frame-1];target=path[frame];previous=result[frame-1]-path[frame-1]
            reference,check=solve_local(warm,target,previous,g);assert check['passed']
            rotation,_=np.linalg.qr(np.random.default_rng(20260920).normal(size=(3,3)))
            rotation[:,0]*=np.linalg.det(rotation);shift=np.array([1.7,-2.9,.6])
            changed,check=solve_local(warm@rotation+shift,target@rotation+shift,previous@rotation,g)
            assert check['passed'];rigid_error=float(np.abs(changed-(reference@rotation+shift)).max())
            assert rigid_error<1e-4,rigid_error
            permutation=np.random.default_rng(20260921).permutation(len(warm));inverse=np.argsort(permutation)
            g_perm=dict(g,weights=g['weights'][permutation])
            for name in ('bonds','angles','locked_pairs','tetrahedra','planar'):
                g_perm[name]=torch.as_tensor(inverse)[g[name]]
            changed,check=solve_local(warm[permutation],target[permutation],previous[permutation],g_perm)
            assert check['passed'];permutation_error=float(np.abs(changed[inverse]-reference).max())
            assert permutation_error<1e-4,permutation_error
    difference=float(np.sqrt(np.square(results[128]-results[256]).mean()))
    assert difference<=.01,difference
    report=dict(completed=True,passed=True,exact_replay=True,encoded_rms_difference_angstrom=difference,
        rigid_transform_max_error_angstrom=rigid_error,permutation_max_error_angstrom=permutation_error,
        tolerance_angstrom=.01,rows=reviews)
    (a.output/'review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
