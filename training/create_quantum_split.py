import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import networkx as nx
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--graph-cache", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20260815)
args = parser.parse_args()
for output_path in (args.output, args.summary):
    if output_path.exists():
        raise FileExistsError(output_path)

with args.manifest.open(encoding="utf-8", newline="") as handle:
    manifest = [row for row in csv.DictReader(handle) if row["split"] == "train"]

node_match = nx.algorithms.isomorphism.categorical_node_match("z", 0)
buckets = defaultdict(list)
group_members = defaultdict(list)
group_by_id = {}
next_group = 0

with h5py.File(args.graph_cache, "r") as graphs:
    for system_index, row in enumerate(manifest, start=1):
        complex_id = row["complex_id"]
        graph_data = graphs[complex_id]
        atomic_numbers = graph_data["ligand_atomic_numbers"][:].astype(np.int64)
        heavy_indices = np.flatnonzero(atomic_numbers != 1)
        heavy_set = set(int(value) for value in heavy_indices)
        graph = nx.Graph()
        for atom_index in heavy_indices:
            graph.add_node(int(atom_index), z=int(atomic_numbers[atom_index]))
        for atom_a, atom_b in graph_data["ligand_bonds"][:]:
            atom_a = int(atom_a)
            atom_b = int(atom_b)
            if atom_a in heavy_set and atom_b in heavy_set:
                graph.add_edge(atom_a, atom_b)
        core = nx.k_core(graph, k=2)
        scaffold = core if core.number_of_nodes() else graph
        signatures = []
        for atom in scaffold:
            signatures.append(
                (
                    scaffold.nodes[atom]["z"],
                    scaffold.degree[atom],
                    tuple(
                        sorted(
                            scaffold.nodes[neighbor]["z"]
                            for neighbor in scaffold.neighbors(atom)
                        )
                    ),
                )
            )
        invariant = (
            scaffold.number_of_nodes(),
            scaffold.number_of_edges(),
            tuple(sorted(signatures)),
        )
        group_key = None
        for candidate_key, representative in buckets[invariant]:
            if nx.is_isomorphic(scaffold, representative, node_match=node_match):
                group_key = candidate_key
                break
        if group_key is None:
            group_key = f"S{next_group:06d}"
            next_group += 1
            buckets[invariant].append((group_key, scaffold.copy()))
        group_by_id[complex_id] = group_key
        group_members[group_key].append(complex_id)
        if system_index % 1000 == 0 or system_index == len(manifest):
            print(f"grouped {system_index}/{len(manifest)}", flush=True)

group_keys = np.asarray(sorted(group_members), dtype=object)
rng = np.random.default_rng(args.seed)
group_keys = group_keys[rng.permutation(len(group_keys))]
validation_target = round(0.20 * len(manifest))
validation_groups = set()
validation_count = 0
for group_key in group_keys:
    if validation_count >= validation_target:
        break
    validation_groups.add(group_key)
    validation_count += len(group_members[group_key])

rows = [
    {
        "complex_id": row["complex_id"],
        "internal_split": (
            "validation"
            if group_by_id[row["complex_id"]] in validation_groups
            else "train"
        ),
        "scaffold_group": group_by_id[row["complex_id"]],
    }
    for row in manifest
]
train_groups = {
    row["scaffold_group"] for row in rows if row["internal_split"] == "train"
}
validation_groups_actual = {
    row["scaffold_group"]
    for row in rows
    if row["internal_split"] == "validation"
}
if train_groups & validation_groups_actual:
    raise ValueError("topology-scaffold leakage")

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "method": "topology scaffold split",
    "seed": args.seed,
    "system_count": len(rows),
    "internal_train_count": sum(row["internal_split"] == "train" for row in rows),
    "internal_validation_count": sum(
        row["internal_split"] == "validation" for row in rows
    ),
    "scaffold_group_count": len(group_members),
    "train_scaffold_count": len(train_groups),
    "validation_scaffold_count": len(validation_groups_actual),
    "scaffold_overlap_count": 0,
    "definition": "labeled heavy-atom 2-core; full heavy graph for acyclic ligands; exact graph isomorphism",
}
args.summary.parent.mkdir(parents=True, exist_ok=True)
args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
