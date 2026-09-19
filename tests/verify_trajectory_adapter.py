"""Check final-coordinate gradients, geometry symmetries and path independence."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json
from pathlib import Path
import torch
from trajectory_adapter import TrajectoryAdapter
from adapter_data import load_adapter_case
from ligand_kinematics import rotation_increment


def main():
    p=argparse.ArgumentParser()
    for name in ('root','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(919)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=TrajectoryAdapter(**saved['spec']).cuda();model.load_state_dict(saved['model']);model.eval()
    manifest=json.loads((a.root/'inputs/manifest.json').read_text());results=[]
    for name in ('6B4U_T3','5W6I_T1'):
        row=next(r for r in manifest['probe'] if r['id']+'_'+r['tier']==name)
        record,cache,c=load_adapter_case(row['file'],a.root/'proposals'/(name+'.pt'),a.root/'chemistry_combined_r3')
        raw=cache['paths'][:2,:2].cuda();c['future_times']=c['future_times'][:2]
        c['quantum'].requires_grad_();c['features'].requires_grad_()
        out,_=model(raw,c);direction=torch.randn_like(out)/out.numel()**.5
        loss=(out*direction).sum();parameter=model.vector_coefficients.weight
        pg,qg,fg=torch.autograd.grad(loss,(parameter,c['quantum'],c['features']))
        eps=1e-4
        with torch.no_grad():
            original=parameter[0,0].clone();parameter[0,0]=original+eps;plus=model(raw,c)[0]
            parameter[0,0]=original-eps;minus=model(raw,c)[0];parameter[0,0]=original
            fd=float(((plus-minus)*direction).sum()/(2*eps));analytic=float(pg[0,0])
            solo=torch.cat([model(x[None],c)[0] for x in raw])
            rot=rotation_increment(raw.new_tensor([.17,-.21,.13]));shift=raw.new_tensor([1.1,-.7,.3]);moved=dict(c)
            for key in ('observed','last','environment'):moved[key]=c[key]@rot.T+shift
            moved['velocity']=c['velocity']@rot.T
            rotated=model(raw@rot.T+shift,moved)[0]
            n=raw.shape[-2];perm=torch.randperm(n,device='cuda');inv=torch.argsort(perm);shuffled=dict(c)
            shuffled['last']=c['last'][perm];shuffled['velocity']=c['velocity'][perm];shuffled['observed']=c['observed'][:,perm]
            shuffled['features']=c['features'][perm];shuffled['edges']=inv[c['edges']]
            g=dict(c['geometry']);g['weights']=g['weights'][perm]
            for key in ('bonds','angles','tetrahedra','locked_pairs','planar'):g[key]=inv[g[key]]
            shuffled['geometry']=g;reordered=model(raw[...,perm,:],shuffled)[0][...,inv,:]
        result=dict(prefix=name,finite=bool(torch.isfinite(out).all() and torch.isfinite(pg).all()),
            parameter_fd_relative_error=abs(fd-analytic)/max(1.,abs(fd),abs(analytic)),
            quantum_gradient_norm=float(qg.norm()),feature_gradient_norm=float(fg.norm()),
            path_independence_max_error=float((out.detach()-solo).abs().max()),
            SE3_max_error=float((rotated-(out.detach()@rot.double().T+shift)).abs().max()),
            permutation_max_error=float((reordered-out.detach()).abs().max()))
        result['passed']=result['finite'] and result['parameter_fd_relative_error']<.02 and result['quantum_gradient_norm']>0 and result['feature_gradient_norm']>0 and result['path_independence_max_error']<1e-8 and result['SE3_max_error']<1e-3 and result['permutation_max_error']<1e-3
        results.append(result);print(json.dumps(result),flush=True)
    a.output.write_text(json.dumps(dict(passed=all(r['passed'] for r in results),rows=results),indent=2));assert all(r['passed'] for r in results)


if __name__=='__main__':main()
