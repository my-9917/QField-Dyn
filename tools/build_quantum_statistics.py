import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--md", type=Path, required=True)
parser.add_argument("--qm", type=Path, required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--internal-split", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()
for output_path in (args.output, args.summary):
    if output_path.exists():
        raise FileExistsError(output_path)

with args.manifest.open(encoding="utf-8", newline="") as handle:
    manifest = {
        row["complex_id"]: row
        for row in csv.DictReader(handle)
        if row["split"] == "train"
    }
with args.internal_split.open(encoding="utf-8", newline="") as handle:
    split_rows = list(csv.DictReader(handle))
split_by_id = {row["complex_id"]: row["internal_split"] for row in split_rows}
if set(split_by_id) != set(manifest):
    raise ValueError("internal split and competition-train IDs differ")

atom_target_name = "gfn2_charge_(water)"
molecule_target_names = ("Electron_Affinity", "Hardness")
train_atom_values = []
train_molecule_values = []

args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.md, "r") as md, h5py.File(
    args.qm, "r"
) as qm, h5py.File(args.graph_cache, "r") as graphs, h5py.File(
    args.output, "w"
) as output:
    output.attrs["source_split"] = "competition_train_only"
    output.attrs["atom_scope"] = "ligand_heavy_atoms"
    output.attrs["atom_target"] = atom_target_name
    output.attrs["molecule_targets"] = json.dumps(molecule_target_names)
    for index, split_row in enumerate(split_rows, start=1):
        complex_id = split_row["complex_id"]
        row = manifest[complex_id]
        graph = graphs[complex_id]
        md_group = md[row["md_group"]]
        ligand_start = int(md_group["molecules_begin_atom_index"][-1])
        coordinates = md_group["trajectory_coordinates"][0, ligand_start:].astype(
            np.float32
        )
        atomic_numbers = graph["ligand_atomic_numbers"][:].astype(np.int16)
        heavy = atomic_numbers != 1
        heavy_indices = np.flatnonzero(heavy)
        old_to_heavy = np.full(len(atomic_numbers), -1, dtype=np.int32)
        old_to_heavy[heavy_indices] = np.arange(len(heavy_indices), dtype=np.int32)
        bonds = graph["ligand_bonds"][:].astype(np.int32)
        heavy_bonds = old_to_heavy[bonds[np.all(heavy[bonds], axis=1)]]

        qm_group = qm[complex_id]
        qm_atomic_numbers = np.asarray(
            [int(value) for value in qm_group["atom_properties"]["atom_names"][:]],
            dtype=np.int16,
        )
        qm_heavy = qm_atomic_numbers != 1
        if not np.array_equal(atomic_numbers[heavy], qm_atomic_numbers[qm_heavy]):
            raise ValueError(f"{complex_id}: QM and MD heavy-atom orders differ")
        property_names = [
            value.decode("utf-8")
            for value in qm_group["atom_properties"]["atom_properties_names"][:]
        ]
        atom_target = qm_group["atom_properties"]["atom_properties_values"][
            qm_heavy, property_names.index(atom_target_name)
        ].astype(np.float32)
        molecule_target = np.asarray(
            [qm_group["mol_properties"][name][()] for name in molecule_target_names],
            dtype=np.float32,
        )
        if not np.isfinite(atom_target).all() or not np.isfinite(
            molecule_target
        ).all():
            raise FloatingPointError(f"{complex_id}: non-finite quantum target")

        group = output.create_group(complex_id)
        group.attrs["internal_split"] = split_by_id[complex_id]
        group.create_dataset("atomic_numbers", data=atomic_numbers[heavy])
        group.create_dataset("coordinates", data=coordinates[heavy])
        group.create_dataset("bonds", data=heavy_bonds)
        group.create_dataset("atom_target", data=atom_target)
        group.create_dataset("molecule_target", data=molecule_target)
        if split_by_id[complex_id] == "train":
            train_atom_values.append(atom_target)
            train_molecule_values.append(molecule_target)
        if index % 1000 == 0 or index == len(split_rows):
            print(f"cached {index}/{len(split_rows)}", flush=True)

    train_atom_values = np.concatenate(train_atom_values)
    train_molecule_values = np.stack(train_molecule_values)
    output.attrs["atom_mean"] = float(train_atom_values.mean())
    output.attrs["atom_std"] = float(train_atom_values.std())
    output.attrs["molecule_mean"] = train_molecule_values.mean(axis=0)
    output.attrs["molecule_std"] = train_molecule_values.std(axis=0)

summary = {
    "system_count": len(split_rows),
    "internal_train_count": sum(value == "train" for value in split_by_id.values()),
    "internal_validation_count": sum(
        value == "validation" for value in split_by_id.values()
    ),
    "atom_target": atom_target_name,
    "molecule_targets": molecule_target_names,
}
args.summary.parent.mkdir(parents=True, exist_ok=True)
args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(summary)
