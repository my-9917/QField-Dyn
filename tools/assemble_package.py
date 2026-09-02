import argparse
import shutil
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Assemble trained artifacts into an independent inference package."
)
parser.add_argument("--template", type=Path, default=Path("."))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--quantum-encoder",
    type=Path,
    default=Path("reproduced_artifacts/quantum/quantum_encoder.pt"),
)
parser.add_argument(
    "--quantum-statistics",
    type=Path,
    default=Path("reproduced_data/quantum_statistics.hdf5"),
)
parser.add_argument(
    "--trajectory-model",
    type=Path,
    default=Path("reproduced_artifacts/trajectory_model.pt"),
)
parser.add_argument(
    "--trajectory-model-t1",
    type=Path,
    default=Path("reproduced_artifacts/trajectory_model_t1.pt"),
)
parser.add_argument(
    "--trajectory-model-t2",
    type=Path,
    default=Path("reproduced_artifacts/trajectory_model_t2.pt"),
)
parser.add_argument(
    "--history-trajectory-model-t1",
    type=Path,
    default=Path("reproduced_artifacts/history_t1/history_trajectory_t1_epoch2.pt"),
)
parser.add_argument(
    "--history-trajectory-model-t2",
    type=Path,
    default=Path("reproduced_artifacts/history_trajectory_t2.pt"),
)
parser.add_argument(
    "--velocity-correlation-t1",
    type=Path,
    default=Path("reproduced_artifacts/velocity_correlation_t1.csv"),
)
parser.add_argument(
    "--t1-phase-targets",
    type=Path,
    default=Path("reproduced_artifacts/t1_phase_targets.csv"),
)
parser.add_argument(
    "--velocity-correlation",
    type=Path,
    default=Path("reproduced_artifacts/velocity_correlation.csv"),
)
parser.add_argument(
    "--expert-gate",
    type=Path,
    default=Path("reproduced_artifacts/expert_gate.pt"),
)
parser.add_argument(
    "--geometry-coefficients",
    type=Path,
    default=Path("reproduced_artifacts/geometry_coefficients.csv"),
)
parser.add_argument(
    "--model-config",
    type=Path,
    default=Path("reproduced_artifacts/model.json"),
)
arguments = parser.parse_args()

if arguments.output.exists():
    raise FileExistsError(arguments.output)

arguments.output.mkdir(parents=True)
artifacts = arguments.output / "artifacts"
config = arguments.output / "config"
artifacts.mkdir()
config.mkdir()

shutil.copytree(
    arguments.template / "src",
    arguments.output / "src",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
public_tools = arguments.output / "tools"
public_tools.mkdir()
for filename in ("__init__.py", "predict_public.py", "public_io.py"):
    shutil.copy2(arguments.template / "tools" / filename, public_tools / filename)
for filename in (
    "predict.py",
    "requirements.txt",
    "requirements-public.txt",
    "DATA_NOTICE.md",
):
    shutil.copy2(arguments.template / filename, arguments.output / filename)
shutil.copy2(
    arguments.template / "DEPLOYMENT_README.md",
    arguments.output / "README.md",
)

artifact_sources = {
    "quantum_encoder.pt": arguments.quantum_encoder,
    "quantum_statistics.hdf5": arguments.quantum_statistics,
    "trajectory_model.pt": arguments.trajectory_model,
    "trajectory_model_t1.pt": arguments.trajectory_model_t1,
    "trajectory_model_t2.pt": arguments.trajectory_model_t2,
    "history_trajectory_t1.pt": arguments.history_trajectory_model_t1,
    "history_trajectory_t2.pt": arguments.history_trajectory_model_t2,
    "velocity_correlation_t1.csv": arguments.velocity_correlation_t1,
    "t1_phase_targets.csv": arguments.t1_phase_targets,
    "velocity_correlation.csv": arguments.velocity_correlation,
    "expert_gate.pt": arguments.expert_gate,
    "geometry_coefficients.csv": arguments.geometry_coefficients,
}
for filename, source in artifact_sources.items():
    shutil.copy2(source, artifacts / filename)
shutil.copy2(arguments.model_config, config / "model.json")

print(arguments.output.resolve())
for path in sorted(artifacts.iterdir()):
    print(path.name)
print("config/model.json")
