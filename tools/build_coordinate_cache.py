import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from .coordinates import amber_cell, corrected_ligand_trajectory


parser = argparse.ArgumentParser()
parser.add_argument("--md", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--topology-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-systems", type=int)
args = parser.parse_args()

if args.output.exists():
    raise FileExistsError(args.output)

with args.manifest.open(encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
if args.max_systems is not None:
    rows = rows[: args.max_systems]

args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.md, "r") as md, h5py.File(args.output, "w") as output:
    output.attrs["split"] = args.split
    output.attrs["system_count"] = len(rows)
    output.attrs["coordinate_method"] = "main receptor periodic-boundary correction"
    output.attrs["source_future"] = args.split
    output.attrs["test_future_used"] = False
    for system_index, row in enumerate(rows, start=1):
        complex_id = row["complex_id"]
        source = md[row["md_group"]]
        trajectory = source["trajectory_coordinates"][:].astype(np.float64)
        ligand_start = int(source["molecules_begin_atom_index"][-1])
        corrected, receptor_index, receptor_rmsd = corrected_ligand_trajectory(
            trajectory,
            source["atoms_number"][:],
            source["molecules_begin_atom_index"][:],
            amber_cell(args.topology_root / complex_id.lower() / "production.top.gz"),
        )
        if corrected.shape != (trajectory.shape[0], trajectory.shape[1] - ligand_start, 3):
            raise ValueError(f"{complex_id}: corrected ligand shape mismatch")
        if not np.isfinite(corrected).all():
            raise FloatingPointError(f"{complex_id}: non-finite corrected coordinates")
        group = output.create_group(complex_id)
        group.create_dataset(
            "ligand_coordinates",
            data=corrected.astype(np.float32),
            compression="lzf",
        )
        group.attrs["receptor_molecule_index"] = receptor_index
        group.attrs["receptor_rmsd_max"] = float(receptor_rmsd.max())
        if system_index % 100 == 0 or system_index == len(rows):
            print(f"cached {system_index}/{len(rows)}", flush=True)

print({"split": args.split, "system_count": len(rows), "output": str(args.output)})
