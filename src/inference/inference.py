"""Conditioned trajectory generation and encoded T4 recurrence; no artifact I/O."""
import numpy as np
import torch
from projection import project_paths
from projection_encoding import encode_coordinates


@torch.no_grad()
def generate(model, data, meta, seed, geometry_record=None, projection_config=None, progress=None):
    generator = torch.Generator(device=data['X_obs'].device).manual_seed(seed)
    horizon = meta['n_pred']
    if meta['tier'] != 'T4':
        noise = model.source_noise(data, 1, horizon, generator)
        raw = model.sample(data, noise)['X_gen'][0]
        assert torch.isfinite(raw).all()
        with torch.enable_grad():
            corrected, evidence = project_paths(raw.cpu().numpy()[None], geometry_record, projection_config)
        return torch.from_numpy(corrected[0]).to(raw.device), [], dict(
            generation_version='v2_constraint_decoded', raw_ligand=raw.cpu(),
            projection=evidence, projection_config=projection_config,
            processing_paths=1, output_member=0)
    assert geometry_record is not None and projection_config is not None
    block = int(6400//meta['dt_ps'])
    assert block >= 2
    original_static, history = model.condition(data)
    current = data['X_obs'][-1:]
    parts, raw_parts, rows, projection_rows = [], [], [], []
    budget_state = {}
    for begin in range(0, horizon, block):
        count = min(block, horizon-begin)
        local = dict(data, X_obs=current)
        static = model.recurrent_static(local, original_static)
        noise = model.source_noise(local, 1, count, generator)
        coarse = model.flow.sample(noise, static, history, model.solver_steps)['X_gen'][0]
        assert torch.isfinite(coarse).all()
        # Every recurrent input obeys the same ligand geometry contract as observed training inputs.
        with torch.enable_grad():
            corrected, evidence = project_paths(coarse.cpu().numpy()[None], geometry_record, projection_config,
                                                budget_state=budget_state)
        decoded, _ = encode_coordinates(corrected, geometry_record)
        delivered = torch.from_numpy(decoded[0]).to(coarse.device)
        raw_parts.append(coarse.cpu())
        projection_rows.extend(dict(block_start=begin, **row) for row in evidence)
        rows.append(dict(start_frame=begin, frames=count,
            local_lead_times_ps=(np.arange(1,count+1)*meta['dt_ps']).tolist(),
            absolute_lead_times_ps=(np.arange(begin+1,begin+count+1)*meta['dt_ps']).tolist(),
            feedback_representation='full_atom_xtc_decoded_then_model_float32',
            input_last_frame=current.detach().cpu().tolist(),
            output_last_frame=delivered[-1].detach().cpu().tolist(),
            budget_after_block=dict(budget_state),
            ligand_centroid_shift_angstrom=float((delivered[-1].mean(0)-original_static['X_last'].mean(0)).norm()),
            maximum_step_angstrom=float(torch.diff(torch.cat([current.double(), delivered]), dim=0).norm(dim=-1).max()),
            raw_maximum_step_angstrom=float(torch.diff(torch.cat([current, coarse]), dim=0).norm(dim=-1).max())))
        if progress is not None:
            progress(dict(stage='t4_block', id=meta['id'], **rows[-1],
                projection_success_frames=sum(row['success'] for row in evidence)))
        parts.append(torch.from_numpy(corrected[0]).to(coarse.device))
        current = delivered[-1:].to(dtype=data['X_obs'].dtype)
    return torch.cat(parts), rows, dict(generation_version='t4_encoded_geometry_feedback_v2',
        raw_ligand=torch.cat(raw_parts), projection=projection_rows, projection_config=projection_config,
        processing_paths=1, output_member=0,
        raw_ligand_scope='per-block raw outputs conditioned on the corrected previous block; paired with delivered blocks')
