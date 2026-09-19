"""Map public CCD bond chemistry onto the observed explicit-H connectivity."""
import io
import json
from pathlib import Path
import urllib.request
from rdkit import Chem


def fetch(cache,name,url):
    path=Path(cache)/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=30) as response:payload=response.read()
        path.write_bytes(payload)
    return path.read_bytes()


def flat_heavy(molecule):
    m=Chem.RemoveHs(molecule,sanitize=False)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0);atom.SetIsAromatic(False);atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0);atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in m.GetBonds():bond.SetBondType(Chem.BondType.SINGLE);bond.SetIsAromatic(False)
    m.UpdatePropertyCache(strict=False)
    return m


def ccd_assignment(molecule,pdb_id,cache):
    data=json.loads(fetch(cache,pdb_id.lower()+'.json',
        'https://www.ebi.ac.uk/pdbe/api/pdb/entry/ligand_monomers/'+pdb_id.lower()))
    components=sorted({r['chem_comp_id'] for r in data[pdb_id.lower()]})
    target_indices=[a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum()>1]
    target_flat=flat_heavy(molecule);candidates=[]
    for component in components:
        payload=fetch(cache,component+'.sdf','https://files.rcsb.org/ligands/download/'+component+'_ideal.sdf')
        template=next(Chem.ForwardSDMolSupplier(io.BytesIO(payload),removeHs=False))
        assert template is not None, ('invalid_CCD',component)
        th=[a.GetIdx() for a in template.GetAtoms() if a.GetAtomicNum()>1]
        if len(th)!=len(target_indices):continue
        source_flat=flat_heavy(template)
        if source_flat.GetNumBonds()!=target_flat.GetNumBonds():continue
        matches=source_flat.GetSubstructMatches(target_flat,uniquify=False,maxMatches=10000)
        assert len(matches)<10000, ('CCD_atom_mapping_limit',pdb_id,component)
        kekule=Chem.Mol(template);Chem.Kekulize(kekule,clearAromaticFlags=True)
        for match in matches:
            mapped={native:th[match[i]] for i,native in enumerate(target_indices)}
            corrected=Chem.Mol(molecule);changes=[];valid=True
            for native,source in mapped.items():
                na=corrected.GetAtomWithIdx(native);sa=kekule.GetAtomWithIdx(source)
                nh=sum(n.GetAtomicNum()==1 for n in na.GetNeighbors())
                sh=sum(n.GetAtomicNum()==1 for n in sa.GetNeighbors())
                delta=nh-sh
                if delta and na.GetAtomicNum() not in (7,8,15,16):valid=False;break
                na.SetFormalCharge(sa.GetFormalCharge()+delta)
                if delta:changes.append(dict(atom=native,hydrogen_change=delta,formal_charge=na.GetFormalCharge()))
            if not valid:continue
            for bond in corrected.GetBonds():
                i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
                if i in mapped and j in mapped:
                    bond.SetBondType(kekule.GetBondBetweenAtoms(mapped[i],mapped[j]).GetBondType())
            try:
                Chem.SanitizeMol(corrected)
            except (Chem.AtomValenceException,Chem.KekulizeException):continue
            Chem.AssignStereochemistryFrom3D(corrected)
            candidates.append((sum(abs(r['hydrogen_change']) for r in changes),Chem.MolToSmiles(corrected),corrected,
                dict(component=component,heavy_atom_map=mapped,protonation_changes=changes,
                     source='RCSB CCD ideal SDF; explicit observed hydrogens define protonation')))
    assert candidates, ('no_CCD_chemical_mapping',pdb_id,components)
    best=min(c[0] for c in candidates);chosen=[c for c in candidates if c[0]==best]
    assert len({c[1] for c in chosen})==1, ('ambiguous_CCD_chemistry',pdb_id)
    # Equivalent graph automorphisms are ordered by the supplied atomic mapping.
    chosen.sort(key=lambda c:tuple(c[3]['heavy_atom_map'].values()))
    return chosen[0][2],chosen[0][3]
