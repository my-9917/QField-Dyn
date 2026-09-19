"""Pinned author NeuralMD transfer on the common heavy-atom development protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from neuralmd_author import ASSETS, load_author_model, predict
from cluster_jobs import prefix_tasks
from ligand_geometry import build_geometry
from projection_encoding import encode_coordinates
from semiflexible_scores import score_paths


def heavy_record(record, heavy, calibration):
    inputs = dict(record['inputs']); graph = inputs['ligand_graph']
    reverse = np.full(len(heavy), -1, dtype=int); reverse[heavy] = np.arange(heavy.sum())
    bonds = np.asarray(graph['bonds']); bonds = reverse[bonds[heavy[bonds].all(1)]]
    inputs['ligand_graph'] = dict(atomic_numbers=np.asarray(graph['atomic_numbers'])[heavy], bonds=bonds)
    inputs['X_obs'] = np.asarray(inputs['X_obs'])[:, heavy]
    return dict(record, inputs=inputs, X_future=np.asarray(record['X_future'])[:, heavy],
                geometry=build_geometry(inputs, calibration))


def main():
    p = argparse.ArgumentParser()
    for key in ('statistics','calibration','output'): p.add_argument('--'+key, type=Path, required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest',type=Path);inputs.add_argument('--case-manifest',type=Path)
    p.add_argument('--paths',type=int,default=32)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    tasks = (json.loads(a.case_manifest.read_text())['tasks'] if a.case_manifest else prefix_tasks(json.loads(a.manifest.read_text())))
    ids = {t['key'].rsplit('_',1)[0].lower() for t in tasks}
    train = {s.strip().lower() for s in (ASSETS/'assets/train_MD.txt').read_text().splitlines()}
    overlap = sorted(ids & train); assert not overlap, overlap
    digest = hashlib.sha256((ASSETS/'assets/model.pth').read_bytes()).hexdigest()
    assert digest == '4f35d8fea9a8f38e4f2fb576cf74279a0359a7b614f926b1d453937d7a548351'
    started=time.perf_counter(); model=load_author_model(a.device); setup=time.perf_counter()-started
    protocol=dict(method='NeuralMD author ODE seed22 pretrained transfer',source='https://github.com/chao1224/NeuralMD',
        source_commit='a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe',checkpoint_sha256=digest,
        parameters=sum(p.numel() for p in model.parameters()),device=a.device,threads=a.threads,
        overlap_with_author_training=overlap,atom_scope='ligand heavy atoms',deterministic_paths=a.paths,
        integration='author Euler step_size 0.1; native time index / 100',model_setup_seconds=setup,
        training='author checkpoint; no retraining on this project',future_ground_truth_used_in_generation=False)
    (a.output/f'protocol_{a.shard}.json').write_text(json.dumps(protocol,indent=2))
    scale=np.asarray(json.loads(a.statistics.read_text())['scale']); calibration=json.loads(a.calibration.read_text())
    rows=[]
    for task in tasks[a.shard::a.shards]:
        dest=a.output/(task['key']+'.json')
        if dest.exists(): rows.append(json.loads(dest.read_text())); continue
        record=torch.load(task['record'],map_location='cpu',weights_only=False)
        started=time.perf_counter(); coordinates,heavy=predict(model,record['inputs'],record['meta']['n_pred'],a.device)
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        assert np.isfinite(coordinates).all()
        # Unscored H placeholders retain the exact full-complex XTC encoding; only heavy atoms enter scores.
        full=np.broadcast_to(record['inputs']['X_obs'][-1],(len(coordinates),len(heavy),3)).copy()
        full[:,heavy]=coordinates; encoded,er=encode_coordinates(full[None],record)
        common=heavy_record(er,heavy,calibration)
        paths=np.repeat(encoded[:,:,heavy],a.paths,axis=0)
        metrics,_=score_paths(paths,common,scale)
        row=dict(id=record['meta']['id'],tier=record['meta']['tier'],finite=True,
            method='NeuralMD',scope='common heavy atom supplementary comparison',metrics=metrics,
            complete_model_inference_seconds=seconds,cold_start_inference_seconds=setup+seconds,
            frames=len(coordinates),parameters=protocol['parameters'])
        torch.save(dict(coordinates=encoded[0][:,heavy],heavy_indices=np.flatnonzero(heavy),source_record=task['record'],row=row),a.output/(task['key']+'.pt'))
        dest.write_text(json.dumps(row,indent=2,allow_nan=False));rows.append(row)
        print(json.dumps(dict(prefix=task['key'],seconds=seconds,completed=len(rows))),flush=True)
    (a.output/f'review_{a.shard}.json').write_text(json.dumps(dict(completed=True,protocol=protocol,rows=rows),indent=2,allow_nan=False))


if __name__=='__main__': main()
