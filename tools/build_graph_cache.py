import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from .topology import read_prmtop


FLAGS = {
    "POINTERS",
    "ATOMIC_NUMBER",
    "RESIDUE_LABEL",
    "RESIDUE_POINTER",
    "BOND_EQUIL_VALUE",
    "BONDS_INC_HYDROGEN",
    "BONDS_WITHOUT_HYDROGEN",
}

parser = argparse.ArgumentParser()
parser.add_argument("--md", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--topology-root", type=Path, required=True)
parser.add_argument("--topology-name", required=True)
parser.add_argument("--split", choices=("train", "validation"), required=True)
parser.add_argument("--pocket-cutoff", type=float, default=10.0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

if args.output.exists():
    raise FileExistsError(args.output)

with args.manifest.open(encoding="utf-8", newline="") as manifest_file:
    manifest = [
        row for row in csv.DictReader(manifest_file) if row["split"] == args.split
    ]

args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.md, "r") as md, h5py.File(args.output, "w") as cache:
    cache.attrs["split"] = args.split
    cache.attrs["frame_index"] = 0
    cache.attrs["pocket_cutoff_angstrom"] = args.pocket_cutoff
    cache.attrs["protein_heavy_only"] = True

    for system_index, row in enumerate(manifest, start=1):
        complex_id = row["complex_id"]
        md_group = md[row["md_group"]]
        md_atomic_numbers = md_group["atoms_number"][:]
        ligand_start = int(md_group["molecules_begin_atom_index"][-1])

        topology_path = (
            args.topology_root / complex_id.lower() / args.topology_name
        )
        sections = read_prmtop(topology_path, FLAGS)
        pointers = np.asarray(sections["POINTERS"], dtype=np.int64)
        topology_atom_count = int(pointers[0])
        topology_atomic_numbers = np.asarray(
            sections["ATOMIC_NUMBER"], dtype=np.int64
        )
        residue_labels = sections["RESIDUE_LABEL"]
        residue_starts = (
            np.asarray(sections["RESIDUE_POINTER"], dtype=np.int64) - 1
        )
        residue_ends = np.concatenate(
            (residue_starts[1:], [topology_atom_count])
        )
        atom_residue_labels = np.empty(topology_atom_count, dtype=object)
        for label, start, end in zip(
            residue_labels, residue_starts, residue_ends
        ):
            atom_residue_labels[start:end] = label
        retained = np.flatnonzero(
            ~np.isin(atom_residue_labels, ["WAT", "Na+", "Cl-"])
        )
        if not np.array_equal(
            topology_atomic_numbers[retained], md_atomic_numbers
        ):
            raise ValueError(f"{complex_id}: topology/MD atom order mismatch")

        topology_to_md = np.full(topology_atom_count, -1, dtype=np.int64)
        topology_to_md[retained] = np.arange(len(retained))
        bond_records = np.asarray(
            sections["BONDS_INC_HYDROGEN"]
            + sections["BONDS_WITHOUT_HYDROGEN"],
            dtype=np.int64,
        ).reshape(-1, 3)
        bond_atoms = topology_to_md[bond_records[:, :2] // 3]
        ligand_bond_mask = np.all(bond_atoms >= ligand_start, axis=1)
        ligand_bonds = bond_atoms[ligand_bond_mask] - ligand_start
        bond_equilibrium = np.asarray(
            sections["BOND_EQUIL_VALUE"], dtype=np.float32
        )[bond_records[ligand_bond_mask, 2] - 1]

        frame_one = md_group["trajectory_coordinates"][0].astype(np.float32)
        protein_heavy_indices = np.flatnonzero(
            md_atomic_numbers[:ligand_start] != 1
        )
        ligand_heavy = md_atomic_numbers[ligand_start:] != 1
        protein_heavy_coordinates = frame_one[protein_heavy_indices]
        ligand_heavy_coordinates = frame_one[ligand_start:][ligand_heavy]
        squared_distances = np.sum(
            (
                protein_heavy_coordinates[:, None, :]
                - ligand_heavy_coordinates[None, :, :]
            )
            ** 2,
            axis=-1,
        )
        pocket_mask = (
            np.min(squared_distances, axis=1) <= args.pocket_cutoff**2
        )
        pocket_md_indices = protein_heavy_indices[pocket_mask]

        output_group = cache.create_group(complex_id)
        output_group.create_dataset(
            "protein_atomic_numbers",
            data=md_atomic_numbers[pocket_md_indices].astype(np.int16),
        )
        output_group.create_dataset(
            "protein_coordinates",
            data=frame_one[pocket_md_indices],
        )
        output_group.create_dataset(
            "pocket_md_indices", data=pocket_md_indices.astype(np.int32)
        )
        output_group.create_dataset(
            "ligand_atomic_numbers",
            data=md_atomic_numbers[ligand_start:].astype(np.int16),
        )
        output_group.create_dataset(
            "ligand_bonds", data=ligand_bonds.astype(np.int32)
        )
        output_group.create_dataset(
            "bond_equilibrium", data=bond_equilibrium
        )

        if system_index % 100 == 0 or system_index == len(manifest):
            print(f"cached {system_index}/{len(manifest)}", flush=True)
