import argparse
import csv
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--displacement-metrics", type=Path, required=True)
parser.add_argument("--translation-metrics", type=Path, required=True)
parser.add_argument("--rotation-metrics", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20260816)
parser.add_argument("--max-rms-displacement", type=float)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)


def read_row(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


displacement = read_row(args.displacement_metrics)
translation = read_row(args.translation_metrics)
rotation = read_row(args.rotation_metrics)
parameters = {
    "max_rms_displacement": (
        args.max_rms_displacement
        if args.max_rms_displacement is not None
        else float(displacement["max_rms_displacement"])
    ),
    "position_correlation": float(translation["position_correlation"]),
    "translation_log_intercept": float(translation["log_intercept"]),
    "translation_log_slope": float(translation["log_slope"]),
    "rotation_log_intercept": float(rotation["rotation_log_intercept"]),
    "rotation_log_slope": float(rotation["rotation_log_slope"]),
    "seed": args.seed,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
print(parameters)
