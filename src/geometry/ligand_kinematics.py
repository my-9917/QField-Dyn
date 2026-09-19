"""Differentiable proper rigid motion and chemically allowed subtree rotations."""
from collections import deque
import numpy as np
import torch
from rdkit import Chem


def build_kinematic_template(reference, chemistry):
    names=chemistry['names'];n=len(names)
    assert len(set(names))==n, 'native atom names identify the permutation-independent tree'
    adjacency=[[] for _ in range(n)]
    for i,j in chemistry['bonds']:adjacency[i].append(j);adjacency[j].append(i)
    cut={tuple(sorted(b)) for b in chemistry['rotors']}
    fragments=[];which=np.full(n,-1)
    for start in sorted(range(n),key=lambda i:names[i]):
        if which[start]>=0:continue
        stack=[start];which[start]=len(fragments);atoms=[]
        while stack:
            i=stack.pop();atoms.append(i)
            for j in adjacency[i]:
                if which[j]<0 and tuple(sorted((i,j))) not in cut:
                    which[j]=which[start];stack.append(j)
        fragments.append(sorted(atoms,key=lambda i:names[i]))
    z=np.asarray(chemistry['numbers']);reference=np.asarray(reference,dtype=float)
    root=max(range(len(fragments)),key=lambda f:(sum(z[fragments[f]]>1),len(fragments[f]),tuple(names[i] for i in fragments[f])))
    edges=[[] for _ in fragments]
    for i,j in chemistry['rotors']:
        assert which[i]!=which[j], 'rotor must be a bridge'
        edges[which[i]].append((int(which[j]),i,j));edges[which[j]].append((int(which[i]),j,i))
    order=[];seen={root};queue=deque([root]);children=[[] for _ in fragments]
    while queue:
        f=queue.popleft()
        for child,i,j in sorted(edges[f],key=lambda e:(names[e[1]],names[e[2]])):
            if child in seen:continue
            seen.add(child);queue.append(child);children[f].append(child);order.append((child,i,j))
    assert len(seen)==len(fragments)
    subtrees={}
    def collect(f):
        atoms=list(fragments[f])
        for child in children[f]:atoms.extend(collect(child))
        subtrees[f]=atoms
        return atoms
    collect(root)
    masks=np.zeros((len(order),n),bool)
    for k,(child,_,_) in enumerate(order):masks[k,subtrees[child]]=True
    masses=np.array([Chem.GetPeriodicTable().GetAtomicWeight(int(v)) for v in z]);masses/=masses.sum()
    return dict(version='observed_reference_kinematics_v1',reference=torch.tensor(reference,dtype=torch.float64),
        weights=torch.tensor(masses,dtype=torch.float64),heavy=torch.tensor(z>1),
        axes=torch.tensor([[i,j] for _,i,j in order],dtype=torch.long).reshape(-1,2),
        masks=torch.tensor(masks),root_atoms=torch.tensor(fragments[root]),
        names=names,chemistry=chemistry)


def template_to(template,device,dtype=torch.float32):
    return {k:(v.to(device=device,dtype=dtype) if v.is_floating_point() else v.to(device))
            if isinstance(v,torch.Tensor) else v for k,v in template.items()}


def axis_rotate(x,origin,axis,angle):
    p=x-origin[...,None,:]
    c=angle.cos()[...,None,None];s=angle.sin()[...,None,None]
    a=axis[...,None,:]
    return origin[...,None,:]+p*c+torch.cross(a.expand_as(p),p,dim=-1)*s+a*(p*a).sum(-1,keepdim=True)*(1-c)


def internal_coordinates(template,torsions):
    x=template['reference'].expand(*torsions.shape[:-1],-1,-1)
    for k,(i,j) in enumerate(template['axes'].tolist()):
        origin=x[...,j,:];axis=x[...,j,:]-x[...,i,:]
        axis=axis/torch.linalg.vector_norm(axis,dim=-1,keepdim=True)
        rotated=axis_rotate(x,origin,axis,torsions[...,k])
        x=torch.where(template['masks'][k,:,None],rotated,x)
    return x-(x*template['weights'][:,None]).sum(-2,keepdim=True)


def decode_kinematics(template,centre,rotation,torsions):
    internal=internal_coordinates(template,torsions)
    return torch.einsum('...ij,...nj->...ni',rotation,internal)+centre[...,None,:]


def rotation_increment(omega):
    """SO(3) exponential, differentiable at zero and throughout finite rotations."""
    x,y,z=omega.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*omega.shape[:-1],3,3)
    return torch.matrix_exp(skew)


def proper_fit(reference,target):
    """Unweighted proper Kabsch fit; returns a column-vector rotation."""
    a=reference-reference.mean(-2,keepdim=True);b=target-target.mean(-2,keepdim=True)
    u,_,vh=torch.linalg.svd(a.transpose(-1,-2)@b)
    signs=torch.ones(*u.shape[:-2],3,device=u.device,dtype=u.dtype)
    signs[...,-1]=torch.linalg.det(u@vh)
    return ((u*signs[...,None,:])@vh).transpose(-1,-2)
