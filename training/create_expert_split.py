import argparse
import csv
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

with args.manifest.open(encoding="utf-8", newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
generator = np.random.default_rng(20260815)
permutation = generator.permutation(len(rows))
fit_count = 11759
assignments = np.full(len(rows), "internal_validation", dtype=object)
assignments[permutation[:fit_count]] = "fit"

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["complex_id", "partition"])
    writer.writeheader()
    for row, assignment in zip(rows, assignments):
        writer.writerow({"complex_id": row["complex_id"], "partition": assignment})
print({"fit": int(np.sum(assignments == "fit")), "internal_validation": int(np.sum(assignments != "fit"))})
