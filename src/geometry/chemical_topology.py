"""Recover explicit-H ligand bond orders from native topology and total charge.

Connectivity and atom identity come from the native prmtop. Bond orders are
valence reconstructions, checked against the native GAFF aromatic/sp3 labels;
their provenance is retained rather than described as supplied SDF chemistry.
"""
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from amber_topology import read_prmtop


def native_ligand_chemistry(topology, graph, reference, ccd_cache=None, observed_perception=False):
    f = read_prmtop(Path(topology), {'POINTERS', 'RESIDUE_POINTER', 'RESIDUE_LABEL',
        'ATOMIC_NUMBER', 'ATOM_NAME', 'AMBER_ATOM_TYPE', 'CHARGE',
        'BONDS_INC_HYDROGEN', 'BONDS_WITHOUT_HYDROGEN'})
    index = f['RESIDUE_LABEL'].index('MOL')
    starts = np.r_[np.asarray(f['RESIDUE_POINTER'])-1, f['POINTERS'][0]]
    start, end = starts[index:index+2]
    z = np.asarray(f['ATOMIC_NUMBER'][start:end])
    np.testing.assert_array_equal(z, graph['atomic_numbers'])
    np.testing.assert_array_equal(f['ATOM_NAME'][start:end], graph['atom_names'])
    raw_bonds = np.asarray(f['BONDS_INC_HYDROGEN']+f['BONDS_WITHOUT_HYDROGEN']).reshape(-1,3)[:,:2]//3
    bonds = raw_bonds[((raw_bonds>=start)&(raw_bonds<end)).all(1)]-start
    assert set(map(tuple,np.sort(bonds,axis=1))) == set(map(tuple,np.sort(graph['bonds'],axis=1)))
    charge = sum(f['CHARGE'][start:end])/18.2223
    assert abs(charge-round(charge)) < .1, ('nonintegral_native_charge', charge)
    mol = Chem.RWMol()
    for n in z:
        atom = Chem.Atom(int(n)); atom.SetNoImplicit(True); mol.AddAtom(atom)
    for i,j in graph['bonds']: mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    mol = mol.GetMol()
    conformer = Chem.Conformer(len(z))
    for i,xyz in enumerate(np.asarray(reference)): conformer.SetAtomPosition(i,xyz)
    mol.AddConformer(conformer)
    if observed_perception:
        from observed_chemistry import perceive_observed_chemistry
        mol=perceive_observed_chemistry(graph,reference)
        assignment=dict(source='Open Babel 3D perception; explicit-H closed-shell N4/O1 charges',version='3.1.1.23',qualification='candidate checked against native GAFF')
    elif ccd_cache is None:
        rdDetermineBonds.DetermineBondOrders(mol, charge=int(round(charge)), allowChargedFragments=True,
                                            embedChiral=True, useAtomMap=False, maxIterations=10000)
        assignment=dict(source='explicit-H valence reconstruction from rounded native charge')
    else:
        from chemical_templates import ccd_assignment
        mol,assignment=ccd_assignment(mol,Path(topology).parent.name,ccd_cache)
    Chem.SanitizeMol(mol)
    assert len(Chem.GetMolFrags(mol)) == 1, 'disconnected ligand needs separate component kinematics'
    types = f['AMBER_ATOM_TYPE'][start:end]
    for i,t in enumerate(types):
        if t in ('ca','cp','cq'):
            assert mol.GetAtomWithIdx(i).GetIsAromatic(), ('native_aromatic_mismatch',i,t)
        if t == 'c3':
            assert all(b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetAtomWithIdx(i).GetBonds()), ('native_sp3_mismatch',i)
    # Acyl-heteroatom and amidine/guanidine links retain their observed local geometry.
    restricted = set()
    pattern = Chem.MolFromSmarts('[C,S,P](=[O,S,N])-[N,O,S]')
    for centre, double, hetero in mol.GetSubstructMatches(pattern):
        restricted.add(tuple(sorted((centre,hetero))))
    rotors = []
    for bond in mol.GetBonds():
        i,j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if bond.GetBondType()!=Chem.BondType.SINGLE or bond.IsInRing() or tuple(sorted((i,j))) in restricted: continue
        if z[i]<=1 or z[j]<=1: continue
        if sum(n.GetAtomicNum()>1 for n in mol.GetAtomWithIdx(i).GetNeighbors())<=1: continue
        if sum(n.GetAtomicNum()>1 for n in mol.GetAtomWithIdx(j).GetNeighbors())<=1: continue
        rotors.append([i,j])
    return dict(version='native_chemistry_checked_v2', source=str(topology),
        charge_source=assignment['source'], native_charge=charge,
        formal_charge=Chem.GetFormalCharge(mol), native_charge_difference=float(Chem.GetFormalCharge(mol)-charge),
        chemical_assignment=assignment,numbers=z.tolist(), names=list(graph['atom_names']),
        bonds=np.asarray(graph['bonds']).tolist(),
        bond_orders=[mol.GetBondBetweenAtoms(int(i),int(j)).GetBondTypeAsDouble() for i,j in graph['bonds']],
        aromatic=[a.GetIsAromatic() for a in mol.GetAtoms()],
        formal_charges=[a.GetFormalCharge() for a in mol.GetAtoms()],
        chiral_tags=[str(a.GetChiralTag()) for a in mol.GetAtoms()],
        native_types=types, smiles=Chem.MolToSmiles(mol), rotors=rotors,
        restricted_links=[list(v) for v in sorted(restricted)],
        ring_atoms=[list(r) for r in mol.GetRingInfo().AtomRings()],
        provenance='chemical assignment recorded above; native connectivity identity and GAFF cross-check; stereochemistry from observation')
