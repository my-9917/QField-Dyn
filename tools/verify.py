import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
EXPECTED_SHAPES = {"T1": 10, "T2": 20, "T3": 80}


def main():
    parser = argparse.ArgumentParser(description="Verify a frozen prediction.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=tuple(EXPECTED_SHAPES))
    parser.add_argument("--project-directory", type=Path, default=PROJECT_DIRECTORY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.project_directory.resolve()))
    from src import QFieldDynPredictor

    with np.load(arguments.input) as data:
        inputs = (
            data["protein_atomic_numbers"],
            data["protein_coordinates"],
            data["ligand_atomic_numbers"],
            data["ligand_bonds"],
            data["ligand_structure_coordinates"],
            data["observed_ligand_coordinates"],
        )
    with np.load(arguments.reference) as data:
        reference = data["coordinates"]

    predictor = QFieldDynPredictor(arguments.project_directory, arguments.device)
    prediction_a, weights_a = predictor.predict(*inputs, arguments.task)
    prediction_b, weights_b = predictor.predict(*inputs, arguments.task)
    expected_shape = (
        EXPECTED_SHAPES[arguments.task],
        len(inputs[2]),
        3,
    )
    maximum_difference = float(np.max(np.abs(prediction_a - reference)))
    result = {
        "shape": prediction_a.shape,
        "finite": bool(np.isfinite(prediction_a).all()),
        "repeat_exact": bool(np.array_equal(prediction_a, prediction_b)),
        "weights_repeat_exact": bool(
            (weights_a is None and weights_b is None)
            or np.array_equal(weights_a, weights_b)
        ),
        "maximum_reference_difference": maximum_difference,
    }
    print(result)
    if prediction_a.shape != expected_shape:
        raise AssertionError(result)
    if not result["finite"] or not result["repeat_exact"]:
        raise AssertionError(result)
    if not result["weights_repeat_exact"]:
        raise AssertionError(result)
    if maximum_difference > arguments.tolerance:
        raise AssertionError(result)


if __name__ == "__main__":
    main()