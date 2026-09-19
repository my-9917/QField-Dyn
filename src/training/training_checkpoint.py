"""Atomic training snapshots with every logical random stream and rank total."""
import json
import hashlib
from numbers import Real
from pathlib import Path
import torch
import torch.distributed as dist
from training_runtime import capture_streams


def scientific_configuration(spec, config, plan, calibration, source, history_identity):
    """Bind the trained mapping, loss, sampling and final-output definitions."""
    definition=dict(model=spec,training=config,loss_weights=calibration['weights'] if calibration else {},
        source_distribution=dict(sigma=spec['sigma'],motion_statistics=spec.get('motion_statistics')),
        history_calibration=history_identity,
        sampling=plan,projection=json.loads((source/'configs/aligned_projection_0917_v4.json').read_text()),
        evaluator=json.loads((source/'configs/selection_v1.json').read_text()),
        inference_steps=64,precision=dict(deterministic=True,tf32=False))
    definition['numerical_source']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
        ('semiflexible_model.py','shared_encoder.py','quantum_memory.py','path_flow.py','structured_motion.py',
         'aligned_losses.py','ligand_geometry.py','semiflexible_scores.py')}
    encoded=json.dumps(definition,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda tensor:tensor.tolist()).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(),definition=json.loads(encoded))


def state_difference(left, right):
    if isinstance(left,torch.Tensor):
        assert left.shape==right.shape and left.dtype==right.dtype
        dtype=torch.complex128 if left.is_complex() else torch.float64
        return float((left.to(dtype)-right.to(dtype)).abs().max()) if left.numel() else 0.
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        return max((state_difference(left[k],right[k]) for k in left),default=0.)
    if isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        return max((state_difference(x,y) for x,y in zip(left,right)),default=0.)
    if isinstance(left,Real):return float(abs(left-right))
    assert left==right,(left,right)
    return 0.


def gather_training_state(streams, states, total, world):
    local = dict(streams=capture_streams(streams, states), total=total.detach().cpu())
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered[0] = local
    combined = {slot:value for item in gathered for slot,value in item['streams'].items()}
    return [combined[i] for i in range(len(combined))], [item['total'] for item in gathered]


def save_training_snapshot(output, payload, cursor):
    output = Path(output)
    name = f"epoch_{payload['epoch']:03d}_update_{payload['updates']:06d}_snapshot.pt"
    path = output/name
    assert not path.exists(), path
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    temporary.replace(path)
    with (output/'snapshots.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(path=str(path), updates=payload['updates'],
            epoch=payload['epoch'], stage_update=cursor, checkpoint_kind=payload['checkpoint_kind'],
            bytes=path.stat().st_size))+'\n')
    return path
