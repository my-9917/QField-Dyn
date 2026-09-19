"""Trajectory dynamics and time-window diagnostics in explicit physical units."""
import numpy as np
from scipy.stats import wasserstein_distance


def trajectory_metrics(predicted,truth,last,dt_ps):
    x,y,last=(np.asarray(a,dtype=float) for a in (predicted,truth,last))
    assert x.shape==y.shape and x.ndim==3 and len(x)>=3
    xf=np.sqrt(np.mean(np.sum((x-x.mean(0))**2,-1),0))
    yf=np.sqrt(np.mean(np.sum((y-y.mean(0))**2,-1),0))
    dx=np.diff(np.concatenate((last[None],x)),axis=0)
    dy=np.diff(np.concatenate((last[None],y)),axis=0)
    sx,sy=(np.linalg.norm(a,axis=-1) for a in (dx,dy))
    rx,ry=(np.sqrt(np.mean(np.sum((a-a.mean(1,keepdims=True))**2,-1),1)) for a in (x,y))
    error=np.sqrt(np.mean(np.sum((x-y)**2,-1),1))
    result={'rmsf_mae_angstrom':float(np.abs(xf-yf).mean()),
            'predicted_mean_atom_rmsf_angstrom':float(xf.mean()),'true_mean_atom_rmsf_angstrom':float(yf.mean()),
            'step_distribution_w1_angstrom':float(np.mean([wasserstein_distance(sx[:,i],sy[:,i]) for i in range(x.shape[1])])),
            'radius_of_gyration_w1_angstrom':float(wasserstein_distance(rx,ry)),
            'rmsd_final_angstrom':float(error[-1]),'rmsd_growth_angstrom':float(error[-1]-error[0]),
            'rmsd_time_average_angstrom':float(np.trapz(error,dx=dt_ps)/((len(x)-1)*dt_ps)),
            'correlations':[],'windows':[]}
    for lag in (1,2,4,8):
        if lag>=len(dx):continue
        values=[]
        for delta in (dx,dy):
            a,b=delta[:-lag],delta[lag:]
            den=np.sqrt(np.sum(a*a)*np.sum(b*b))
            values.append(float(np.sum(a*b)/den) if den>0 else None)
        result['correlations'].append({'lag_ps':lag*dt_ps,'predicted':values[0],'truth':values[1],
            'absolute_error':abs(values[0]-values[1]) if all(v is not None for v in values) else None})
    for name,indices in zip(('early','middle','late'),np.array_split(np.arange(len(x)),3)):
        result['windows'].append({'window':name,'start_ps':float((indices[0]+1)*dt_ps),
            'end_ps':float((indices[-1]+1)*dt_ps),'rmsd_angstrom':float(error[indices].mean()),
            'predicted_step_rms_angstrom':float(np.sqrt(np.mean(sx[indices]**2))),
            'true_step_rms_angstrom':float(np.sqrt(np.mean(sy[indices]**2)))})
    return result
