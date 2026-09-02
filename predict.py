import argparse
from pathlib import Path

import numpy as np


from src import QFieldDynPredictor


PROJECT_DIRECTORY = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Predict a ligand trajectory in a fixed protein pocket."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("T1", "T2", "T3"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    with np.load(arguments.input) as data:
        inputs = (
            data["protein_atomic_numbers"],
            data["protein_coordinates"],
            data["ligand_atomic_numbers"],
            data["ligand_bonds"],
            data["ligand_structure_coordinates"],
            data["observed_ligand_coordinates"],
        )

    predictor = QFieldDynPredictor(PROJECT_DIRECTORY, arguments.device)
    coordinates, weights = predictor.predict(*inputs, arguments.task)
    np.savez_compressed(
        arguments.output,
        coordinates=coordinates,
        expert_weights=(
            weights
            if weights is not None
            else np.empty((0, 2), dtype=np.float32)
        ),
    )


if __name__ == "__main__":
    main()
