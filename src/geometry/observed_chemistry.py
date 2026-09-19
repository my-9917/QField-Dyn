"""Observed connectivity/3D bond perception for chemical-input diagnostics.

This candidate is not a qualified replacement for supplied chemical topology.
Unresolved radical valences and added implicit hydrogens are explicit failures.
"""
from rdkit import Chem
from openbabel import openbabel as ob


def perceive_observed_chemistry(graph,reference):
    mol=ob.OBMol();mol.BeginModify()
    for z,xyz in zip(graph['atomic_numbers'],reference):
        atom=mol.NewAtom();atom.SetAtomicNum(int(z));atom.SetVector(*map(float,xyz))
    for i,j in graph['bonds']:mol.AddBond(int(i)+1,int(j)+1,1)
    mol.EndModify();mol.SetDimension(3);mol.PerceiveBondOrders()
    writer=ob.OBConversion();writer.SetOutFormat('mol')
    rd=Chem.MolFromMolBlock(writer.WriteString(mol),sanitize=False,removeHs=False)
    rd.UpdatePropertyCache(strict=False)
    for atom in rd.GetAtoms():
        if atom.GetAtomicNum()==8 and atom.GetExplicitValence()==1:atom.SetFormalCharge(-1)
        if atom.GetAtomicNum()==7 and atom.GetExplicitValence()==4:atom.SetFormalCharge(1)
    Chem.SanitizeMol(rd);Chem.AssignStereochemistryFrom3D(rd)
    assert all(x.GetNumRadicalElectrons()==0 for x in rd.GetAtoms()), 'unresolved open-shell valence'
    assert all(x.GetNumImplicitHs()==0 for x in rd.GetAtoms()), 'native explicit-H identity changed'
    assert [x.GetAtomicNum() for x in rd.GetAtoms()]==list(graph['atomic_numbers'])
    assert sorted(tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in rd.GetBonds())==sorted(tuple(sorted(map(int,b))) for b in graph['bonds'])
    return rd
