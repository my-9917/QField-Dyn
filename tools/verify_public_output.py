import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.formats.libmdaxdr import XTCFile


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()

expected_files = set()
rows = []
for tier in ("T1", "T2", "T3", "T4"):
    ids = (args.data / tier / "ids.txt").read_text(encoding="utf-8").splitlines()
    for complex_id in ids:
        expected = args.predictions / tier / f"{complex_id}_pred.xtc"
        expected_files.add(expected.resolve())
        system = args.data / tier / complex_id
        meta = json.loads((system / "meta.json").read_text(encoding="utf-8"))
        universe = mda.Universe(str(system / f"{complex_id}.pdb"))
        ligand_indices = universe.select_atoms(
            f"resname {meta['ligand_resname']}"
        ).indices
        non_ligand_indices = np.setdiff1d(
            np.arange(meta["n_atoms"]), ligand_indices
        )
        with XTCFile(str(system / f"{complex_id}_obs.xtc")) as trajectory:
            observed = [trajectory.read() for _ in range(len(trajectory))]
        with XTCFile(str(expected)) as trajectory:
            predicted = [trajectory.read() for _ in range(len(trajectory))]

        coordinates = np.stack([frame.x.copy() for frame in predicted])
        times = np.asarray([frame.time for frame in predicted])
        boxes = np.stack([frame.box.copy() for frame in predicted])
        reference = observed[-1].x.copy()
        expected_times = (
            meta["n_obs"] + np.arange(meta["n_pred"])
        ) * meta["dt_ps"]

        assert coordinates.shape == (meta["n_pred"], meta["n_atoms"], 3)
        assert np.isfinite(coordinates).all()
        assert np.max(
            np.abs(
                coordinates[:, non_ligand_indices]
                - reference[non_ligand_indices]
            )
        ) <= 0.001
        assert np.max(np.abs(times - expected_times)) <= 1e-6
        assert np.max(np.abs(boxes - observed[0].box)) <= 0.001
        rows.append({"id": complex_id, "tier": tier})

actual_files = {
    path.resolve() for path in args.predictions.rglob("*_pred.xtc")
}
assert actual_files == expected_files
summary = {
    "system_count": len(rows),
    "tier_counts": {
        tier: sum(row["tier"] == tier for row in rows)
        for tier in ("T1", "T2", "T3", "T4")
    },
    "all_protocol_checks_passed": True,
}
args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
