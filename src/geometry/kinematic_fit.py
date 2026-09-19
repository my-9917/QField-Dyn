"""Fit one frozen reference manifold to target coordinates without environment truth."""
import time
import torch
from ligand_kinematics import decode_kinematics, proper_fit, axis_rotate, rotation_increment


@torch.no_grad()
def initial_state(template,target):
    ref=template['reference'];root=template['root_atoms']
    rotation=proper_fit(ref[root],target[...,root,:])
    x=torch.einsum('...ij,nj->...ni',rotation,ref-ref[root].mean(0))+target[...,root,:].mean(-2,keepdim=True)
    angles=[]
    for k,(i,j) in enumerate(template['axes'].tolist()):
        axis=x[...,j,:]-x[...,i,:];axis=axis/torch.linalg.vector_norm(axis,dim=-1,keepdim=True)
        mask=template['masks'][k]
        current=x[...,mask,:]-x[...,j,None,:]
        desired=target[...,mask,:]-target[...,j,None,:]
        current=current-(current*axis[...,None,:]).sum(-1,keepdim=True)*axis[...,None,:]
        desired=desired-(desired*axis[...,None,:]).sum(-1,keepdim=True)*axis[...,None,:]
        sine=(torch.cross(current,desired,dim=-1)*axis[...,None,:]).sum((-1,-2))
        cosine=(current*desired).sum((-1,-2))
        angle=torch.atan2(sine,cosine);angles.append(angle)
        rotated=axis_rotate(x,x[...,j,:],axis,angle)
        x=torch.where(mask[:,None],rotated,x)
    angles=torch.stack(angles,-1) if angles else target.new_empty((*target.shape[:-2],0))
    centre=(target*template['weights'][:,None]).sum(-2)
    return dict(centre=centre,rotation=rotation,torsions=angles)


def fit_kinematics(template,target,max_iter=200,extra_loss=None):
    """Joint centre/pose/torsion least squares on the decoder's actual state space.

    max_iter is a numerical solver bound. Achieved errors and termination state
    are reported; a local fit is an upper bound on the attainable fit error.
    """
    assert target.ndim==3, 'fit one trajectory at a time; each member has an independent solve'
    start=time.monotonic();state=initial_state(template,target)
    q=target.new_zeros((*target.shape[:-2],6+len(template['axes'])),requires_grad=True)
    optimizer=torch.optim.LBFGS([q],lr=1.,max_iter=max_iter,tolerance_grad=1e-7,
        tolerance_change=1e-10,history_size=20,line_search_fn='strong_wolfe')
    calls=0
    def state_at():
        return dict(centre=state['centre']+q[...,:3],rotation=rotation_increment(q[...,3:6])@state['rotation'],
                    torsions=state['torsions']+q[...,6:])
    def closure():
        nonlocal calls
        optimizer.zero_grad();x=decode_kinematics(template,**state_at())
        error=(x[...,template['heavy'],:]-target[...,template['heavy'],:])
        loss=error.square().sum()/int(template['heavy'].sum())
        if extra_loss is not None:loss=loss+extra_loss(x)
        loss.backward();calls+=1
        assert torch.isfinite(loss) and torch.isfinite(q.grad).all()
        return loss
    optimizer.step(closure);closure()
    final={k:v.detach() for k,v in state_at().items()}
    final['torsions']=torch.atan2(final['torsions'].sin(),final['torsions'].cos())
    with torch.no_grad():
        x=decode_kinematics(template,**final)
        rmsd=(x[...,template['heavy'],:]-target[...,template['heavy'],:]).square().sum(-1).mean(-1).sqrt()
    detail=dict(seconds=time.monotonic()-start,closure_calls=calls,iterations=optimizer.state[q]['n_iter'],
        max_gradient=float(q.grad.abs().max()),mean_rmsd=float(rmsd.mean()),max_rmsd=float(rmsd.max()),
        hit_iteration_bound=optimizer.state[q]['n_iter']>=max_iter,
        interpretation='achieved local reconstruction; target truth is used only in coverage diagnostics')
    return final,x.detach(),detail
