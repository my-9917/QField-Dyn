"""Converged constrained correction from the previous molecular configuration."""
import time
import numpy as np
from scipy.optimize import minimize
import torch
from covalent_output_constraints import values_jacobian

VERSION='uniform_covalent_objective_0920_r4'


@torch.no_grad()
def solve_local(warm, original, previous_correction, context):
    """Minimize the existing correction objective with explicit geometry and COM."""
    g={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in context.items()}
    reference=np.asarray(original,dtype=np.float64);warm=np.asarray(warm,dtype=np.float64)
    mass=g['weights'].numpy();centre=(reference*mass[:,None]).sum(0)
    source=warm-(warm*mass[:,None]).sum(0)
    u,_,vt=np.linalg.svd((source*mass[:,None]).T@(reference-centre))
    orientation=np.eye(3);orientation[-1,-1]=np.linalg.det(u@vt)
    initial=(source@(u@orientation@vt)+centre).ravel()
    target=(reference+.5*np.asarray(previous_correction,dtype=np.float64)).ravel()
    _,lo,hi,_,_=values_jacobian(torch.from_numpy(initial.reshape(1,-1,3)),g)
    low=lo.numpy();high=hi.numpy();finite=np.isfinite(high)
    last_x=None;cached=None
    def evaluate(flat):
        nonlocal last_x,cached
        if last_x is None or not np.array_equal(flat,last_x):
            value,_,_,jac,_=values_jacobian(torch.from_numpy(flat.reshape(1,-1,3)),g)
            v=value.numpy()[0];j=jac.numpy()[0]
            cached=(np.concatenate((v-low,high[finite]-v[finite])),np.concatenate((j,-j[finite])))
            last_x=flat.copy()
        return cached
    initial_residual=float(np.maximum(-evaluate(initial)[0],0).max())
    # Initial feasibility is measured; SLSQP enforces the final constraints.
    com_jac=np.einsum('n,ij->inj',mass,np.eye(3)).reshape(3,-1)
    constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[0],jac=lambda x:evaluate(x)[1]),
        dict(type='eq',fun=lambda x:com_jac@x-centre,jac=lambda x:com_jac)]
    begin=time.perf_counter()
    # Per-atom scaling preserves the minimizer and conditions the line search
    # across molecular sizes and large raw-coordinate corrections.
    objective_scale=float(len(reference))
    result=minimize(lambda x:np.square(x-target).sum()/objective_scale,initial,jac=lambda x:2*(x-target)/objective_scale,
        method='SLSQP',constraints=constraints,options=dict(ftol=1e-10,maxiter=1000))
    residual=float(np.maximum(-evaluate(result.x)[0],0).max())
    com_error=float(np.abs(com_jac@result.x-centre).max())
    passed=bool(result.success and residual<=1e-9 and com_error<=1e-9)
    return result.x.reshape(-1,3),dict(passed=passed,movable_atoms=len(reference),total_atoms=len(reference),
        iterations=int(result.nit),stop_reason=str(result.message),solver_success=bool(result.success),
        solver_status=int(result.status),initial_objective=float(np.square(initial-target).sum()),
        final_objective=float(result.fun*objective_scale),objective_scale=objective_scale,
        initial_residual=initial_residual,maximum_residual=residual,centre_error_angstrom=com_error,
        seconds=time.perf_counter()-begin,method='SLSQP_constrained_correction',global_optimum_claimed=False)
