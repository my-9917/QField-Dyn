"""Real-prefix closure, representation, symmetry and finite-difference checks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from covalent_reconstruction import reconstruction_context,reconstruct_covalent,angle_rows,interval_residual
from ligand_geometry import geometry_terms


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((a.root/'inputs/manifest.json').read_text());rows=[];torch.manual_seed(917)
    for name in ('5W6I_T1','5WB3_T2','2WGI_T1','6B4U_T3'):
        row=next(v for v in manifest['probe'] if v['id']+'_'+v['tier']==name)
        record=torch.load(row['file'],map_location='cpu',weights_only=False)
        cp=a.root/'chemistry_combined_r3'/(row['id']+'.json');chem=json.loads(cp.read_text()) if cp.exists() else None
        g=reconstruction_context(record,'cuda',chem)
        truth=record['X_future'].to('cuda',dtype=torch.float64)
        start=time.perf_counter();recovered=reconstruct_covalent(truth,g,steps=8);torch.cuda.synchronize()
        heavy=g['heavy'];rmsd=(recovered[:,heavy]-truth[:,heavy]).square().sum(-1).mean(-1).sqrt()
        raw=torch.load(a.root/'proposals'/(name+'.pt'),map_location='cpu',weights_only=False)['paths'][:1,:2].cuda().double()
        raw.requires_grad_();start_grad=time.perf_counter();out=reconstruct_covalent(raw,g,steps=8)
        direction=torch.randn_like(out);loss=(out*direction).sum();gradient=torch.autograd.grad(loss,raw)[0]
        tangent=torch.randn_like(raw);tangent=tangent/tangent.norm();eps=1e-6
        plus=reconstruct_covalent(raw.detach()+eps*tangent,g,steps=8);minus=reconstruct_covalent(raw.detach()-eps*tangent,g,steps=8)
        numeric=float(((plus-minus)*direction).sum()/(2*eps));analytic=float((gradient*tangent).sum())
        geometry=geometry_terms(out,record['geometry']);angle_error,angle_grad,indices=angle_rows(raw.flatten(0,1),g['angles'],g['angle_min'],g['angle_max'])
        # Check analytic angle derivatives against autograd independently.
        from ligand_geometry import angles
        probe=raw.detach().clone().requires_grad_();angle_total=interval_residual(torch.cos(angles(probe,g['angles'])),torch.cos(g['angle_max']),torch.cos(g['angle_min']))[0].sum()
        automatic=torch.autograd.grad(angle_total,probe)[0].flatten(0,1)
        accumulated=torch.zeros_like(automatic)
        for slot in range(3):accumulated=accumulated.index_add(1,indices[:,slot],angle_grad[:,:,slot])
        result=dict(prefix=name,mean_truth_reconstruction_rmsd=float(rmsd.mean()),max_truth_rmsd=float(rmsd.max()),
            truth_seconds=time.perf_counter()-start,gradient_seconds=time.perf_counter()-start_grad,
            finite=bool(torch.isfinite(out).all() and torch.isfinite(gradient).all()),
            finite_difference_relative_error=abs(numeric-analytic)/max(1.,abs(numeric),abs(analytic)),
            angle_jacobian_max_error=float((accumulated-automatic).abs().max()),
            raw_geometry={k:float(v.mean()) for k,v in geometry_terms(raw,record['geometry']).items()},
            reconstructed_geometry={k:float(v.mean()) for k,v in geometry.items()},
            chemical_restrictions_available=chem is not None)
        from ligand_kinematics import rotation_increment
        rotation=rotation_increment(raw.new_tensor([.17,-.21,.13]));shift=raw.new_tensor([1.1,-.7,.3])
        transformed=reconstruct_covalent(raw.detach()@rotation.T+shift,g,steps=8)
        result['SE3_max_error']=float((transformed-(out.detach()@rotation.T+shift)).abs().max())
        permutation=torch.randperm(raw.shape[-2],device='cuda');inverse=torch.argsort(permutation)
        shuffled=dict(g);shuffled['weights']=g['weights'][permutation]
        for key in ('bonds','angles','tetrahedra','locked_pairs','planar'):shuffled[key]=inverse[g[key]]
        reordered=reconstruct_covalent(raw.detach()[...,permutation,:],shuffled,steps=8)
        result['permutation_max_error']=float((reordered[...,inverse,:]-out.detach()).abs().max())
        result['engineering_pass']=result['finite'] and result['finite_difference_relative_error']<1e-4 and result['SE3_max_error']<1e-6 and result['permutation_max_error']<1e-6
        rows.append(result);print(json.dumps(result),flush=True)
    a.output.write_text(json.dumps(dict(passed=all(r['engineering_pass'] for r in rows),rows=rows,scope='engineering and training truth representation; model remains untrained'),indent=2))
    assert all(r['engineering_pass'] for r in rows)


if __name__=='__main__':main()
