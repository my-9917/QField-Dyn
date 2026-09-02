import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


OBSERVED_FRAMES = {"T1": 10, "T2": 80, "T3": 20}


def main():
    parser = argparse.ArgumentParser(
        description="Export structure and observed frames from MISATO."
    )
    parser.add_argument("--md", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--graph-cache", required=True, type=Path)
    parser.add_argument("--coordinate-cache", required=True, type=Path)
    parser.add_argument("--complex-id", required=True)
    parser.add_argument("--task", required=True, choices=tuple(OBSERVED_FRAMES))
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    with arguments.manifest.open(encoding="utf-8", newline="") as handle:
        row = next(
            row
            for row in csv.DictReader(handle)
            if row["complex_id"] == arguments.complex_id
        )

    with h5py.File(arguments.md, "r") as md, h5py.File(
        arguments.graph_cache, "r"
    ) as graphs, h5py.File(
        arguments.coordinate_cache, "r"
    ) as coordinates:
        source = md[row["md_group"]]
        ligand_start = int(source["molecules_begin_atom_index"][-1])
        graph = graphs[arguments.complex_id]
        np.savez_compressed(
            arguments.output,
            protein_atomic_numbers=source["atoms_number"][:ligand_start],
            protein_coordinates=source["trajectory_coordinates"][
                0, :ligand_start
            ],
            ligand_atomic_numbers=graph["ligand_atomic_numbers"][:],
            ligand_bonds=graph["ligand_bonds"][:],
            ligand_structure_coordinates=source["trajectory_coordinates"][
                0, ligand_start:
            ],
            observed_ligand_coordinates=coordinates[arguments.complex_id][
                "ligand_coordinates"
            ][: OBSERVED_FRAMES[arguments.task]],
        )


if __name__ == "__main__":
    main()
