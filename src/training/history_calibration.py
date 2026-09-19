"""Checkpoint-bound, training-observation-only calibration of history angles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semiflexible_model import SemiFlexFlow, device_inputs


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def apply_history_calibration(model, report, checkpoint):
    assert report['completed'] and report['partition'] == 'train'
    assert report['checkpoint_sha256'] == file_digest(checkpoint)
    values = dict(model.spec['angle_statistics'])
    for name in ('z_mean', 'z_scale'):
        value = torch.tensor(report[name], device=model.angles.z_mean.device, dtype=torch.float32)
        assert torch.isfinite(value).all()
        if name == 'z_scale':
            assert (value > 0).all()
        getattr(model.angles, name).copy_(value)
        values[name] = value.cpu().tolist()
    model.spec['angle_statistics'] = values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--combine', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    checkpoint_digest = file_digest(a.checkpoint)
    manifest = json.loads((a.cache/'manifest.json').read_text())
    members = [r for r in manifest['rows'] if r['partition'] == 'train']
    assert len(members) in (512,2048,9846) and all(len(r['records']) == 3 for r in members)
    if a.combine:
        rows = []
        for i in range(a.shards):
            part = torch.load(a.output/f'shard_{i}.pt', weights_only=False, map_location='cpu')
            assert part['checkpoint_sha256'] == checkpoint_digest
            rows.extend(part['rows'])
        expected = {(r['id'], tier) for r in members for tier in ('T1','T2','T3')}
        assert len(rows) == len(expected) and {(r['id'],r['tier']) for r in rows} == expected
        mean = torch.stack([r['z'].double().mean(0) for r in rows]).mean(0)
        second = torch.stack([r['z'].double().square().mean(0) for r in rows]).mean(0)
        scale = (second-mean.square()).sqrt()
        assert torch.isfinite(scale).all() and (scale > 0).all()
        saved = torch.load(a.checkpoint, weights_only=False, map_location='cpu')
        old_mean, old_scale = (saved['model']['angles.'+k].double() for k in ('z_mean','z_scale'))
        summary = {}
        for tier in ('T1','T2','T3'):
            tier_rows = [r for r in rows if r['tier'] == tier]
            metrics = {}
            for label, centre, spread in [('old',old_mean,old_scale),('new',mean,scale)]:
                normalized = torch.cat([(r['z'].double()-centre)/spread for r in tier_rows])
                metrics[label] = dict(saturation=float((normalized.abs()>3).double().mean()),
                    median_sensitivity=float((1-normalized.tanh().square()).median()))
            metrics['saturation_halved'] = metrics['new']['saturation'] <= .5*metrics['old']['saturation']
            metrics['sensitivity_increased'] = metrics['new']['median_sensitivity'] > metrics['old']['median_sensitivity']
            summary[tier] = metrics
        report = dict(completed=True, version='history_calibration_0917v4', partition='train',
            checkpoint=str(a.checkpoint), checkpoint_sha256=checkpoint_digest,
            cache=str(a.cache), systems=len(members), prefixes=len(rows),
            weighting='equal systems, equal tiers, equal observed frames within prefix',
            z_mean=mean.tolist(), z_scale=scale.tolist(), summary=summary,
            interface_passed=all(v['saturation_halved'] and v['sensitivity_increased'] for v in summary.values()),
            query_statistics='preserved', quality_qualification='paired generation required')
        (a.output/'calibration.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(a.checkpoint, weights_only=False,map_location='cpu')
    model = SemiFlexFlow(**saved['spec']).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    rows = []
    with torch.no_grad():
        for member in members[a.shard::a.shards]:
            for file in member['records']:
                record = torch.load(file, weights_only=False,map_location='cpu')
                assert record['meta']['partition'] == 'train'
                encoded = model.encoder(device_inputs(record,'cuda'))
                rows.append(dict(id=member['id'],tier=record['meta']['tier'],z=encoded['z_raw'].cpu()))
            if len(rows)%48 == 0:
                print(json.dumps(dict(shard=a.shard,prefixes=len(rows))),flush=True)
    torch.save(dict(rows=rows,checkpoint_sha256=checkpoint_digest),a.output/f'shard_{a.shard}.pt')
    print(json.dumps(dict(completed=True,shard=a.shard,prefixes=len(rows))),flush=True)


if __name__ == '__main__':
    main()
