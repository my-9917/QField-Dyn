import argparse
import csv
from pathlib import Path

import h5py

from src import QFieldDynPredictor


OBSERVED_FRAMES = {"T1": 10, "T2": 80, "T3": 20}

parser = argparse.ArgumentParser()
parser.add_argument("--project-directory", type=Path, default=Path("."))
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--coordinate-cache", type=Path, required=True)
parser.add_argument("--task", choices=tuple(OBSERVED_FRAMES), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

with args.manifest.open(encoding="utf-8", newline="") as handle:
    ids = [
        row["complex_id"]
        for row in csv.DictReader(handle)
        if row["split"] == args.split
    ]

predictor = QFieldDynPredictor(args.project_directory, args.device)
observed_count = OBSERVED_FRAMES[args.task]
args.output.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.graph_cache, "r") as graphs, h5py.File(
    args.coordinate_cache, "r"
) as coordinates, h5py.File(args.output, "w") as output:
    output.attrs["task"] = args.task
    output.attrs["split"] = args.split
    for index, complex_id in enumerate(ids, start=1):
        graph = graphs[complex_id]
        trajectory = coordinates[complex_id]["ligand_coordinates"]
        prediction, _ = predictor.predict(
            graph["protein_atomic_numbers"][:],
            graph["protein_coordinates"][:],
            graph["ligand_atomic_numbers"][:],
            graph["ligand_bonds"][:],
            trajectory[0],
            trajectory[:observed_count],
            args.task,
        )
        output.create_dataset(complex_id, data=prediction, compression="lzf")
        if index % 100 == 0 or index == len(ids):
            print(f"predicted {index}/{len(ids)}", flush=True)
