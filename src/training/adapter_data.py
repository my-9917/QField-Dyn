"""Load observed inputs and frozen proposals under one reconstruction contract."""
import json
from pathlib import Path
import torch
from covalent_reconstruction import reconstruction_context
from trajectory_adapter import adapter_context
from adapter_losses import dynamics_context


def load_adapter_case(record_path,proposal_path,chemistry_dir,device='cuda'):
    record=torch.load(record_path,weights_only=False,map_location='cpu')
    cache=torch.load(proposal_path,weights_only=False,map_location='cpu')
    assert list(cache['atom_names'])==list(record['inputs']['ligand_graph']['atom_names'])
    return record,cache,build_adapter_context(record,cache,chemistry_dir,device)


def build_adapter_context(record,cache,chemistry_dir,device='cuda'):
    chemical=Path(chemistry_dir)/(record['meta']['id']+'.json')
    chemistry=json.loads(chemical.read_text()) if chemical.exists() else None
    geometry=reconstruction_context(record,device,chemistry)
    context=adapter_context(record,cache,device,geometry)
    context['dynamics']=dynamics_context(record,device)
    context['dynamics']['reference_com']=(context['last'].double()*geometry['weights'][:,None]).sum(0)
    return context


def ready_proposals(root):
    # Writers append progress only after a complete torch.save; partially written files stay unread.
    ready=set()
    for scope in ('probe','train'):
        for shard in range(4):
            file=Path(root)/f'progress_{scope}_{shard}.jsonl'
            if file.exists():
                for line in file.read_text().splitlines():
                    if line.endswith('}'):ready.add(json.loads(line)['id'])
    return ready
