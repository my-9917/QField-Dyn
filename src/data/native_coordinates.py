"""MISATO prefix windows with the same observable inputs as anonymous inference."""
from pathlib import Path
from collections import deque

import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors
import numpy as np
from rdkit import Chem

from amber_topology import read_prmtop



FLAGS={'POINTERS','ATOMIC_NUMBER','ATOM_NAME','AMBER_ATOM_TYPE','RESIDUE_LABEL','RESIDUE_POINTER',
       'BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN','BOX_DIMENSIONS'}


def read_training_topology(path):
    source=read_prmtop(Path(path),FLAGS)
    n=source['POINTERS'][0]
    starts=np.asarray(source['RESIDUE_POINTER'])-1
    residue_ids=np.repeat(np.arange(len(starts)),np.diff(np.append(starts,n)))
    labels=np.asarray(source['RESIDUE_LABEL'])[residue_ids]
    keep=np.flatnonzero(~np.isin(labels,['WAT','Na+','Cl-']))
    reverse=np.full(n,-1,dtype=int);reverse[keep]=np.arange(len(keep))
    bonds=np.asarray(source['BONDS_INC_HYDROGEN']+source['BONDS_WITHOUT_HYDROGEN'],dtype=int).reshape(-1,3)
    bonds=reverse[bonds[:,:2]//3];bonds=bonds[(bonds>=0).all(1)]
    beta,a,b,c=source['BOX_DIMENSIONS']
    return {'numbers':np.asarray(source['ATOMIC_NUMBER'])[keep],'names':np.asarray(source['ATOM_NAME'])[keep],
            'types':np.asarray(source['AMBER_ATOM_TYPE'])[keep],
            'residue_ids':residue_ids[keep],'labels':labels[keep],'bonds':bonds,
            'dimensions':np.array([a,b,c,beta,beta,beta]),'source_atom_indices':keep}


def topology_universe(topology):
    residues,first,inverse=np.unique(topology['residue_ids'],return_index=True,return_inverse=True)
    universe=mda.Universe.empty(len(inverse),n_residues=len(residues),atom_resindex=inverse,trajectory=True)
    for key,value in (('names',topology['names']),('elements',[Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in topology['numbers']]),
                      ('resnames',topology['labels'][first]),('resids',residues+1),('bonds',topology['bonds'])):
        universe.add_TopologyAttr(key,value)
    universe.dimensions=topology['dimensions']
    return universe


def unwrap_source(coordinates,universe,protein_indices):
    """Make source covalent components whole, then track their periodic images causally."""
    coordinates=np.asarray(coordinates,dtype=float)
    bonds=universe.bonds.indices
    adjacency=[[] for _ in universe.atoms]
    for a,b in bonds:adjacency[a].append(int(b));adjacency[b].append(int(a))
    n=coordinates.shape[1]
    parent=np.full(n,-1,dtype=int);component=np.full(n,-1,dtype=int);depth=np.zeros(n,dtype=int)
    count=0
    for root in range(n):
        if component[root]>=0:continue
        component[root]=count;parent[root]=root;count+=1
        queue=deque([root])
        while queue:
            a=queue.popleft()
            for b in adjacency[a]:
                if component[b]<0:
                    parent[b]=a;component[b]=component[a];depth[b]=depth[a]+1;queue.append(b)
    children=np.flatnonzero(parent!=np.arange(n))
    delta=coordinates[:,children]-coordinates[:,parent[children]]
    vectors=np.zeros_like(coordinates)
    vectors[:,children]=minimize_vectors(delta.reshape(-1,3),universe.dimensions).reshape(delta.shape)
    whole=coordinates.copy()
    # The graph traversal is shared by all frames. Each level uses already placed parents.
    for level in range(1,int(depth.max())+1):
        atoms=np.flatnonzero(depth==level)
        whole[:,atoms]=whole[:,parent[atoms]]+vectors[:,atoms]
    receptor=np.bincount(component[protein_indices]).argmax()
    centres=np.stack([whole[:,component==i].mean(1) for i in range(count)],axis=1)
    previous=None
    for index in range(len(whole)):
        reference=np.repeat(centres[index,receptor][None],count,axis=0) if previous is None else previous
        delta=centres[index]-reference
        shifts=minimize_vectors(delta,universe.dimensions)-delta
        whole[index]+=shifts[component]
        previous=centres[index]+shifts
    source_delta=coordinates[:,bonds[:,0]]-coordinates[:,bonds[:,1]]
    source_length=np.linalg.norm(minimize_vectors(source_delta.reshape(-1,3),universe.dimensions).reshape(source_delta.shape),axis=-1)
    whole_length=np.linalg.norm(whole[:,bonds[:,0]]-whole[:,bonds[:,1]],axis=-1)
    np.testing.assert_allclose(whole_length,source_length,rtol=0,atol=1e-5)
    return whole
