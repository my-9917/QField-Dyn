"""Shared equivariant trajectory correction from frozen E3 conditions."""
import math
import numpy as np
import torch
from torch import nn
from covalent_reconstruction import reconstruct_covalent


def adapter_context(record,cache,device,geometry):
    x=torch.as_tensor(record['inputs']['X_obs'],device=device,dtype=torch.float32)
    dt=float(record['meta']['dt_ps'])
    bonds=geometry['bonds'];edges=torch.cat((bonds,bonds.flip(-1)),0)
    environment=geometry['environment'].float()
    return dict(observed=x,last=x[-1],velocity=(x[-1]-x[-2])*(80./dt),dt=dt,
        edges=edges,environment=environment,geometry=geometry,
        features=cache['ligand_features'].to(device,dtype=torch.float32),
        quantum=cache['quantum_condition'].to(device,dtype=torch.float32).reshape(-1),
        future_times=torch.arange(1,record['meta']['n_pred']+1,device=device,dtype=torch.float32)*dt)


class TrajectoryAdapter(nn.Module):
    def __init__(self,feature_dim=128,hidden=64,layers=2,reconstruction_steps=8,structured_initialization=False,initialization_steps=32,final_damping=.1):
        super().__init__();self.steps=reconstruction_steps
        self.structured_initialization=structured_initialization;self.initialization_steps=initialization_steps
        self.final_damping=final_damping
        self.input=nn.Sequential(nn.Linear(feature_dim+18+8,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden+1,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.vector_coefficients=nn.Linear(hidden,6)
        nn.init.zeros_(self.vector_coefficients.weight);nn.init.zeros_(self.vector_coefficients.bias)

    def forward(self,proposal,context):
        # A path always uses the same arithmetic shapes during training, ensemble
        # evaluation and single-path delivery.
        if len(proposal)>1:
            values=[self.forward(path[None],context) for path in proposal]
            return torch.cat([v[0] for v in values]),{key:torch.cat([v[1][key] for v in values]) for key in values[0][1]}
        # E3 is frozen. Build its structured initial state before applying learned
        # changes; gradients pass through every step of the FINAL reconstruction.
        with torch.no_grad():
            x=reconstruct_covalent(proposal,context['geometry'],steps=self.initialization_steps).float() if self.structured_initialization else proposal.float()
        m,t,n,_=x.shape;last=context['last'];dt=context['dt']
        displacement=x-last
        velocity=context['velocity'].expand_as(x)
        temporal=[]
        for lag_ps in (80.,160.,320.):
            lag=int(round(lag_ps/dt))
            if lag<1 or abs(lag*dt-lag_ps)>1e-6:
                temporal.append(torch.zeros_like(x));continue
            observed=context['observed'][-lag:][None].expand(m,-1,n,3)
            before=torch.cat((observed,x),1)[:,:t]
            temporal.append((x-before)*(80./lag_ps))
        env=context['environment']
        distance=torch.cdist(x,env,compute_mode='donot_use_mm_for_euclid_dist')
        near_distance,near=distance.topk(min(8,len(env)),dim=-1,largest=False)
        relative=env[near]-x[...,None,:]
        weights=torch.softmax(-near_distance,dim=-1)
        environment_vector=(relative*weights[...,None]).sum(-2)
        times=context['future_times'][None,:,None].expand(m,t,n)/80.
        scalars=torch.stack((displacement.norm(dim=-1),velocity.norm(dim=-1),near_distance[...,0],
            temporal[0].norm(dim=-1),temporal[1].norm(dim=-1),temporal[2].norm(dim=-1),
            torch.log1p(times),torch.full_like(times,math.log1p(dt/80.))),-1)
        features=context['features'].reshape(n,-1)[None,None].expand(m,t,-1,-1)
        quantum=context['quantum'][None,None,None].expand(m,t,n,-1)
        h=self.input(torch.cat((features,quantum,scalars),-1));edges=context['edges'];i,j=edges.T
        length=(x[...,i,:]-x[...,j,:]).norm(dim=-1,keepdim=True)
        degree=torch.bincount(i,minlength=n).to(x).clamp_min(1)[None,None,:,None]
        for message,update in zip(self.messages,self.updates):
            sent=message(torch.cat((h[...,i,:],h[...,j,:],length),-1))
            received=torch.zeros_like(h).index_add(-2,i,sent)/degree
            h=h+update(torch.cat((h,received),-1))
        basis=torch.stack((displacement,velocity,environment_vector,*temporal),-2)
        # Vector directions come from relative geometry, while coefficients are scalars.
        coefficients=self.vector_coefficients(h)
        correction=(coefficients[...,None]*basis).sum(-2)
        corrected=x+correction
        reconstructed=reconstruct_covalent(corrected,context['geometry'],steps=self.steps,damping=self.final_damping)
        return reconstructed,dict(correction=correction,coefficients=coefficients)
