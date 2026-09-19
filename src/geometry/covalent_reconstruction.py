"""Differentiable covalent reconstruction with every ring-closing edge retained.

Coordinates are redundant internal variables. Bond/angle intervals and signed
local volumes constrain them jointly; each unrolled step retains its graph.
Environment forces belong to the learned adapter and its final-coordinate loss.
"""
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from rdkit import Chem


def reconstruction_context(record,device,chemistry=None):
    g={k:v.to(device=device,dtype=torch.float64) if v.is_floating_point() else v.to(device)
       for k,v in record['geometry'].items() if isinstance(v,torch.Tensor)}
    observed=torch.as_tensor(record['inputs']['X_obs'],device=device,dtype=torch.float64)
    z=np.asarray(record['inputs']['ligand_graph']['atomic_numbers'])
    masses=torch.tensor([Chem.GetPeriodicTable().GetAtomicWeight(int(n)) for n in z],device=device,dtype=torch.float64)
    g['weights']=masses/masses.sum();n=len(z);adj=[[] for _ in z]
    for i,j in g['bonds'].tolist():adj[i].append(j);adj[j].append(i)
    locked=[]
    if chemistry is not None:
        assert list(chemistry['names'])==list(record['inputs']['ligand_graph']['atom_names'])
        fixed={tuple(sorted(b)) for b in chemistry['restricted_links']}
        fixed.update(tuple(sorted(b)) for b,o in zip(chemistry['bonds'],chemistry['bond_orders']) if o>1)
        for b,c in sorted(fixed):
            locked.extend(tuple(sorted((a,d))) for a in adj[b] if a!=c for d in adj[c] if d!=b and d!=a)
    # These distances preserve the observed restricted dihedral branches.
    locked=sorted(set(locked)-{tuple(sorted(b)) for b in g['bonds'].tolist()})
    pairs=torch.tensor(locked,device=device,dtype=torch.long).reshape(-1,2)
    lengths=torch.linalg.vector_norm(observed[:,pairs[:,0]]-observed[:,pairs[:,1]],dim=-1)
    g['locked_pairs']=pairs
    g['locked_min']=lengths.amin(0)-.02
    g['locked_max']=lengths.amax(0)+.02
    g['chemical_restrictions_available']=chemistry is not None
    # Stable observed trigonal planarity is available even in anonymous inputs.
    triples=[(i,*sorted(near)) for i,near in enumerate(adj) if len(near)==3]
    planar=torch.tensor(triples,device=device,dtype=torch.long).reshape(-1,4)
    centre,a,b,c=planar.T
    u,v,w=(observed[:,j]-observed[:,centre] for j in (a,b,c))
    scale=(u.norm(dim=-1)*v.norm(dim=-1)*w.norm(dim=-1)).mean(0)
    volume=(u*torch.cross(v,w,dim=-1)).sum(-1)/scale
    stable=volume.abs().amax(0)<.05
    g['planar']=planar[stable];g['planar_scale']=scale[stable]
    g['planar_min']=volume[:,stable].amin(0)-.02;g['planar_max']=volume[:,stable].amax(0)+.02
    g['atoms']=n
    return g


def interval_residual(value,low,high,width=.002):
    upper=(value-high)/width;lower=(low-value)/width
    residual=width*(F.softplus(upper)-F.softplus(lower))
    derivative=torch.sigmoid(upper)+torch.sigmoid(lower)
    return residual,derivative


def distance_rows(x,pairs,low,high):
    v=x[:,pairs[:,0]]-x[:,pairs[:,1]]
    distance=v.norm(dim=-1).clamp_min(1e-10)
    error,weight=interval_residual(distance,low,high)
    direction=v/distance[...,None]
    return error,torch.stack((direction,-direction),-2)*weight[...,None,None],pairs


def angle_rows(x,triples,low,high):
    a,b,c=triples.T;u=x[:,a]-x[:,b];v=x[:,c]-x[:,b]
    lu=u.norm(dim=-1).clamp_min(1e-10);lv=v.norm(dim=-1).clamp_min(1e-10)
    cosine=(u*v).sum(-1)/(lu*lv)
    # The cosine interval has exactly the same allowed angles. Its derivatives
    # remain bounded near collinearity, where angle derivatives contain 1/sin(theta).
    du=v/(lu*lv)[...,None]-cosine[...,None]*u/lu[...,None].square()
    dv=u/(lu*lv)[...,None]-cosine[...,None]*v/lv[...,None].square()
    error,weight=interval_residual(cosine,torch.cos(high),torch.cos(low))
    return error,torch.stack((du,-du-dv,dv),-2)*weight[...,None,None],triples


def volume_rows(x,indices,scale,low,high,sign):
    c,a,b,d=indices.T;u=x[:,a]-x[:,c];v=x[:,b]-x[:,c];w=x[:,d]-x[:,c]
    value=(u*torch.cross(v,w,dim=-1)).sum(-1)*sign/scale
    ga=torch.cross(v,w,dim=-1);gb=torch.cross(w,u,dim=-1);gd=torch.cross(u,v,dim=-1)
    grads=torch.stack((-ga-gb-gd,ga,gb,gd),-2)*(sign/scale)[None,:,None,None]
    error,weight=interval_residual(value,low,high)
    return error,grads*weight[...,None,None],indices


def reconstruct_covalent(proposal,context,steps=8,damping=.1,max_step=.25):
    shape=proposal.shape;x=proposal.double().reshape(-1,shape[-2],3)
    g=context;original_com=(x*g['weights'][:,None]).sum(-2,keepdim=True)
    eye=torch.eye(3*shape[-2],device=x.device,dtype=x.dtype)
    for _ in range(steps):
        constraints=[distance_rows(x,g['bonds'],g['bond_min'],g['bond_max']),
            angle_rows(x,g['angles'],g['angle_min'],g['angle_max']),
            distance_rows(x,g['locked_pairs'],g['locked_min'],g['locked_max']),
            volume_rows(x,g['tetrahedra'],g['tetrahedron_scale'],.1,float('inf'),g['tetrahedron_sign']),
            volume_rows(x,g['planar'],g['planar_scale'],g['planar_min'],g['planar_max'],1.)]
        errors=[];jacobians=[]
        for error,gradient,indices in constraints:
            matrix=x.new_zeros((len(x),len(indices),shape[-2],3))
            matrix=matrix.scatter_add(2,indices[None,:,:,None].expand(len(x),-1,-1,3),gradient)
            errors.append(error);jacobians.append(matrix.flatten(-2))
        error=torch.cat(errors,1);jac=torch.cat(jacobians,1)
        jt=jac.transpose(-1,-2)
        delta=torch.linalg.solve(jt@jac+damping*eye,-(jt@error[...,None])).reshape_as(x)
        norm=delta.norm(dim=-1).amax(-1).clamp_min(max_step)
        delta=delta*(max_step/norm)[:,None,None]
        x=x+delta
        x=x-(x*g['weights'][:,None]).sum(-2,keepdim=True)+original_com
    return x.reshape(shape)
