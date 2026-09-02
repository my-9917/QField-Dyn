import argparse
import csv
from pathlib import Path

import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--md", type=Path, required=True)
parser.add_argument("--train", type=Path, required=True)
parser.add_argument("--validation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

split_files = {
    "train": args.train,
    "validation": args.validation,
}
rows = []

with h5py.File(args.md, "r") as md:
    for split, split_file in split_files.items():
        for complex_id in split_file.read_text(encoding="utf-8").splitlines():
            group = md[complex_id]
            atom_count = group["atoms_number"].shape[0]
            ligand_start = int(group["molecules_begin_atom_index"][-1])
            rows.append(
                {
                    "complex_id": complex_id,
                    "split": split,
                    "md_hdf5": str(args.md),
                    "md_group": complex_id,
                    "atom_count": atom_count,
                    "ligand_atom_count": atom_count - ligand_start,
                    "frame_count": group["trajectory_coordinates"].shape[0],
                }
            )

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
