"""Proper path scores and physics losses evaluated on reconstructed coordinates."""
import itertools
import numpy as np
import torch
from aligned_losses import coordinate_energy_score,feature_context,trajectory_features
from ligand_geometry import geometry_terms


def dynamics_context(record,device):
    context=feature_context(record,device);graph=record['inputs']['ligand_graph'];z=np.asarray(graph['atomic_numbers'])
    neighbours=[[] for _ in z]
    for i,j in graph['bonds']:neighbours[i].append(j);neighbours[j].append(i)
    names=list(graph['atom_names']);quad=[]
    for b,c in graph['bonds']:
        if z[b]<=1 or z[c]<=1:continue
        left=sorted((i for i in neighbours[b] if i!=c and z[i]>1),key=lambda i:names[i])
        right=sorted((i for i in neighbours[c] if i!=b and z[i]>1),key=lambda i:names[i])
        candidates=[(a,b,c,d) for a in left for d in right if a!=d]
        if candidates:quad.append(candidates[0])
    index=torch.tensor(quad,dtype=torch.long,device=device).reshape(-1,4)
    observed=torch.as_tensor(record['inputs']['X_obs'],device=device,dtype=torch.float64)
    _,valid=torsion_phase(observed,index)
    context['torsions']=index[valid.all(0)];context['dt']=float(record['meta']['dt_ps'])
    return context


def torsion_phase(x,index):
    a,b,c,d=index.T;axis=x[...,c,:]-x[...,b,:]
    first=torch.cross(x[...,b,:]-x[...,a,:],axis,dim=-1)
    second=torch.cross(axis,x[...,d,:]-x[...,c,:],dim=-1)
    denominator=first.norm(dim=-1)*second.norm(dim=-1)
    real=(first*second).sum(-1)/denominator.clamp_min(1e-10)
    imaginary=(torch.cross(first,second,dim=-1)*axis).sum(-1)/(denominator*axis.norm(dim=-1)).clamp_min(1e-10)
    return torch.stack((real,imaginary),-1),denominator>1e-6


def motion_features(paths,context,scales,weights):
    # Each physical group contributes an RMS-normalized full time series.
    com=(paths*weights[:,None]).sum(-2)
    last=context['reference_com']
    scalar=trajectory_features(paths,context)
    groups=[(com-last)/scales[0],scalar[...,1:2]/scales[1]]
    if scalar.shape[-1]==3:groups.append(scalar[...,2:3]/scales[2])
    if len(context['torsions']):
        phase,_=torsion_phase(paths,context['torsions']);groups.append(phase.flatten(-2))
    for lag_ps in (80.,160.,320.):
        lag=int(round(lag_ps/context['dt']))
        if lag>=1 and abs(lag*context['dt']-lag_ps)<1e-6 and paths.shape[-3]>lag:
            groups.append((com[...,lag:,:]-com[...,:-lag,:])/scales[0])
    flattened=[g.flatten(-2)/(g.shape[-1]*g.shape[-2])**.5 for g in groups]
    return torch.cat(flattened,-1)/(len(groups)**.5)


def fair_vector_es(samples,target):
    m=len(samples);assert m>=2
    first=(samples-target).norm(dim=-1).mean()
    pair=torch.cdist(samples,samples,compute_mode='donot_use_mm_for_euclid_dist').sum()/(2*m*(m-1))
    return first-pair


def adapter_objectives(paths,target,diagnostic,record,context,scales):
    coordinate=coordinate_energy_score(paths,target,context['dynamics']['heavy'],scales[0])
    with torch.no_grad():truth=motion_features(target,context['dynamics'],scales,context['geometry']['weights'])
    predicted=motion_features(paths,context['dynamics'],scales,context['geometry']['weights'])
    dynamics=fair_vector_es(predicted,truth)
    terms=geometry_terms(paths,record['geometry'],split_overlap=True)
    physics=sum(v.mean() for v in terms.values())
    regularizer=(diagnostic['correction'].double()/scales[0]).square().mean()
    return dict(coordinate=coordinate,dynamics=dynamics,physics=physics,regularizer=regularizer)
