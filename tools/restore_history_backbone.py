import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--history", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

base = torch.load(args.base, map_location="cpu", weights_only=False)
history = torch.load(args.history, map_location="cpu", weights_only=False)
for name, value in base["model_state_dict"].items():
    history["model_state_dict"][name] = value
history["derivation"] = "trained history layers with frozen T2 backbone restored"
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(history, args.output)
