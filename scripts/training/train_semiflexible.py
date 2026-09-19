"""Distributed quantum ligand training with complete epochs and replayable state."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from semiflexible_model import SemiFlexFlow, device_inputs
from sampling import case_seed
from training_runtime import logical_streams, capture_streams, accumulated_update
from training_checkpoint import gather_training_state, save_training_snapshot,scientific_configuration
from training_schedule import exposure_counts


# Measured large-system updates take about 700 s; faster ranks await their gradients.
COLLECTIVE_TIMEOUT = timedelta(hours=1)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--initial', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    origin.add_argument('--resume', type=Path)
    parser.add_argument('--expand-training-cache', action='store_true')
    parser.add_argument('--anchor', choices=['inherited', 'off', 'centred_shared'])
    parser.add_argument('--motion-statistics', type=Path)
    parser.add_argument('--history-calibration', type=Path)
    parser.add_argument('--training-plan', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--replay-through', type=int, help='Save and finish a diagnostic replay at this existing plan position')
    parser.add_argument('--monitor-updates',type=int,default=0)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text())
    plan = json.loads(args.training_plan.read_text()) if args.training_plan else None
    logical_size = plan['logical_world_size'] if plan else world
    assert logical_size % world == 0
    aligned_config = config.get('aligned')
    torch.manual_seed(config['seed'])
    if world > 1:
        dist.init_process_group('nccl', timeout=COLLECTIVE_TIMEOUT)
    manifest = json.loads((args.cache/'manifest.json').read_text())
    assert manifest['completed'] and manifest['input_version'] == 'semiflexible_protein_aligned_v2'
    records = {partition: [file for row in manifest['rows'] if row['partition'] == partition for file in row['records']]
               for partition in ('train', 'validation')}
    initial = torch.load(args.resume or args.initial, weights_only=False, map_location='cpu')
    spec = dict(initial['spec'])
    if args.anchor is not None:
        spec['anchor_enabled'] = args.anchor != 'off'
        if args.anchor == 'centred_shared': spec['anchor_mode'] = 'centred_shared'
    if args.motion_statistics:
        spec['motion_statistics'] = json.loads(args.motion_statistics.read_text())
    if 'geometry_mode' in config: spec['geometry_mode'] = config['geometry_mode']
    if aligned_config and not spec.get('global_position', False):
        from training_initialization import initialize_aligned
        directional = json.loads(Path(aligned_config['directional_statistics']).read_text()) if 'directional_statistics' in aligned_config else None
        model = initialize_aligned(initial, device, directional)
    else:
        model = SemiFlexFlow(**spec).to(device)
        model.load_state_dict(initial['model'], strict=True)
    spec = model.spec
    history_identity = initial.get('history_calibration')
    if args.history_calibration:
        from history_calibration import apply_history_calibration, file_digest
        history_report = json.loads(args.history_calibration.read_text())
        apply_history_calibration(model,history_report,args.resume or args.initial)
        history_identity = dict(path=str(args.history_calibration),sha256=file_digest(args.history_calibration),
                                encoder_checkpoint_sha256=history_report['checkpoint_sha256'])
        spec = model.spec
    if aligned_config:
        from aligned_losses import feature_context
        calibration = json.loads(Path(aligned_config['calibration']).read_text())
        assert calibration['completed'] and calibration['partition'] == 'train' and calibration['prefixes'] == 32
        assert calibration['steps'] == aligned_config['rollout_steps'] and calibration['paths'] == 2
        assert json.loads(Path(aligned_config['distributed_review']).read_text())['passed']
        feature_scales = torch.tensor(json.loads(Path(aligned_config['feature_statistics']).read_text())['scale'], device=device)
        assert spec['global_position'] and config['geometry_mode'] == 'conditional_endpoint'
        if plan and plan.get('loss_calibration'):
            calibration = json.loads(Path(plan['loss_calibration']).read_text())
            assert calibration['completed'] and calibration['partition']=='train'
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer_transfer = None
    if aligned_config and aligned_config.get('inherit_optimizer') and not args.resume:
        from training_initialization import inherit_adam_state
        optimizer_transfer = inherit_adam_state(model, optimizer, initial)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
        patience=config['plateau_patience'], threshold=config['relative_improvement'], min_lr=config['minimum_learning_rate'])
    shuffle = random.Random(config['seed'])
    epoch, updates, best = 0, 0, float('inf')
    cache_transition = None
    assert not args.expand_training_cache or args.resume is not None
    if args.resume:
        saved = initial
        assert saved.get('checkpoint_kind', 'complete_epoch') in ('complete_epoch','stage_boundary','training_snapshot')
        assert saved.get('logical_world_size',saved['world_size']) == logical_size
        for key in ('anchor_enabled', 'anchor_mode', 'motion_statistics', 'geometry_mode', 'solver_steps', 'global_position', 'directional_statistics'):
            assert saved['spec'].get(key) == spec.get(key), key
        assert saved['config'] == config
        if args.expand_training_cache:
            from training_state import validate_cache_expansion
            prior_run = json.loads((args.resume.parent/'run_config.json').read_text())
            previous = json.loads((Path(prior_run['cache'])/'manifest.json').read_text())
            assert saved['cache_config'] == previous['config']
            cache_transition = validate_cache_expansion(previous, manifest)
        else:
            assert saved['cache_config'] == manifest['config']
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        epoch, updates, best = saved['epoch'], saved['updates'], saved['best_cfm']
        shuffle.setstate(saved['shuffle_state'])
    streams, stream_states = logical_streams(config['seed'],logical_size,rank,world,device,
                                             initial['rank_states'] if args.resume else None)
    if plan:
        assert plan['version'] in ('aligned_training_plan_0917v4','aligned_training_plan_0917v5','aligned_training_plan_0918')
        files = [f for b in plan['batches'] for f in b['files'] if f is not None]
        assert len(files)==len(set(files)) and set(files)<=set(records['train'])
        assert all(len(b['files'])==logical_size for b in plan['batches'])
        if plan['kind']=='complete_epoch':
            assert set(files)==set(records['train'])
        else:
            assert plan['kind']=='paired_adaptation' and len(plan['batches'])==64
        if 'learning_rate' in plan:
            for group in optimizer.param_groups: group['lr']=plan['learning_rate']
    resume_progress = initial.get('training_progress') if initial.get('checkpoint_kind')=='training_snapshot' else None
    if resume_progress:
        assert plan is not None
        assert plan == initial['training_plan'] and not args.expand_training_cache
        assert world == initial['world_size']
    if args.replay_through is not None:
        assert plan['kind']=='paired_adaptation' and 0 < args.replay_through <= len(plan['batches'])
    scientific=scientific_configuration(spec,config,plan,calibration if aligned_config else None,Path(__file__).parent,history_identity)
    if resume_progress and 'scientific_configuration' in initial:
        assert scientific['sha256']==initial['scientific_configuration']['sha256']
    if args.validate_only:
        print(json.dumps(dict(validated=True,rank=rank,physical_world_size=world,logical_world_size=logical_size,
            restored_updates=updates,restored_epoch=epoch,owned_streams=list(streams),history_calibration=history_identity,
            train_prefixes=len(records['train']),validation_prefixes=len(records['validation']))),flush=True)
        if world>1: dist.destroy_process_group()
        return
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.resume:
            shutil.copy2(args.resume.parent/'cfm_candidate.pt', args.output/'cfm_candidate.pt')
        (args.output/'source').mkdir()
        for path in Path(__file__).parent.glob('*.py'): shutil.copy2(path, args.output/'source'/path.name)
        (args.output/'run_config.json').write_text(json.dumps(dict(config=config, cache=str(args.cache.resolve()),
            initial=str(args.initial.resolve()) if args.initial else None, resume=str(args.resume) if args.resume else None,
            training_plan=plan,history_calibration=history_identity,logical_world_size=logical_size,
            scientific_configuration_sha256=scientific['sha256'],monitor_updates=args.monitor_updates,
            optimizer_transfer=optimizer_transfer, cache_transition=cache_transition,
            anchor=spec.get('anchor_mode', 'atomwise') if spec['anchor_enabled'] else 'off', world_size=world,
            model_variant={key: spec.get(key) for key in ('anchor_enabled', 'anchor_mode',
                'motion_statistics', 'geometry_mode', 'solver_steps', 'global_position')},
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            distributed_state_boundary='saved optimizer-update or complete-epoch boundary',
            replay_through=args.replay_through, collective_timeout_seconds=COLLECTIVE_TIMEOUT.total_seconds(),
            cache_config=manifest['config']), indent=2))
        (args.output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(), deterministic=True, tf32=False, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
    if world > 1: dist.barrier()
    train_start = time.perf_counter()
    start_epoch = resume_progress['start_epoch'] if resume_progress else epoch
    previous_exposures = resume_progress['stage_start_exposures'] if resume_progress else initial.get('exposures',
        dict(cfm=0,rollout=0,rollout_by_tier={t:0 for t in ('T1','T2','T3')},
             tracked_since_checkpoint=str(args.resume or args.initial)))

    def state_payload(rank_states, kind):
        return dict(spec=spec, model=model.state_dict(), optimizer=optimizer.state_dict(),
            checkpoint_kind=kind, scheduler=scheduler.state_dict(), epoch=epoch, updates=updates, best_cfm=best,
            rank_states=rank_states, shuffle_state=shuffle.getstate(), world_size=world,logical_world_size=logical_size,
            training_plan=plan,history_calibration=history_identity,
            scientific_configuration=scientific,
            rollout_sampling_state=plan.get('sampling_state') if plan else initial.get('rollout_sampling_state'),
            stopping_controller=plan.get('stopping_controller') if plan else initial.get('stopping_controller'),
            config=config, cache_config=manifest['config'],
            source_initial=initial['source_initial'] if args.resume else str(args.initial.resolve()),
            final_generation_selection_pending=True)

    while True:
        diagnostic = plan is not None and plan['kind']=='paired_adaptation'
        epoch += int(not diagnostic and resume_progress is None)
        # Checkpoints and DDP's bucket-order cache share the same epoch boundary.
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
        if plan:
            batches = plan['batches']
        else:
            order = list(records['train']); shuffle.shuffle(order)
            batches = [dict(files=order[i:i+logical_size]+[None]*max(0,i+logical_size-len(order)),
                rollout=bool(aligned_config) and (updates+i//logical_size+1)%aligned_config['every_updates']==0)
                for i in range(0,len(order),logical_size)]
        planned_files = [f for b in batches for f in b['files'] if f is not None]
        cursor = resume_progress['next_batch'] if resume_progress else 0
        processed = resume_progress['processed_prefixes'] if resume_progress else 0
        start = time.perf_counter()
        checkpoint_updates = {math.ceil(fraction*len(batches)): fraction
            for fraction in ([] if diagnostic else config.get('checkpoint_fractions', []))}
        total = resume_progress['rank_totals'][rank].to(device) if resume_progress else torch.zeros(3, device=device, dtype=torch.float64)
        clip_rows = resume_progress['clipping'] if resume_progress else []
        auxiliary_rows = resume_progress['objectives'] if resume_progress else []
        snapshot_every = 16 if diagnostic else 256

        def snapshot(position):
            rank_states, rank_totals = gather_training_state(streams,stream_states,total,world)
            if rank == 0:
                payload = state_payload(rank_states,'training_snapshot')
                payload['training_progress'] = dict(next_batch=position,processed_prefixes=processed,
                    start_epoch=start_epoch,rank_totals=rank_totals,clipping=clip_rows,objectives=auxiliary_rows,
                    stage_start_exposures=previous_exposures)
                payload['exposures'] = exposure_counts(batches,position,previous_exposures)
                save_training_snapshot(args.output,payload,position)
            if world > 1: dist.barrier()

        if plan: snapshot(cursor)
        model.train()
        for epoch_update in range(cursor+1,len(batches)+1):
            batch = batches[epoch_update-1]
            diagnostics={} if rank==0 and epoch_update<=args.monitor_updates else None
            rows,norm = accumulated_update(model,wrapped,batch,streams,stream_states,config,updates,
                device,rank,world,calibration if aligned_config else None,
                feature_scales if aligned_config else None,optimizer,diagnostics)
            updates += 1
            if diagnostics is not None:
                with (args.output/'module_updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(stage_update=epoch_update,updates=updates,**diagnostics))+'\n')
            processed += sum(f is not None for f in batch['files'])
            use_geometry = any(r['geometry_update'] for r in rows)
            if batch['rollout']:
                with (args.output/f'rollout_members_rank{rank}.jsonl').open('a') as stream:
                    for row in rows:
                        stream.write(json.dumps(dict(epoch=epoch,update=updates,
                            **{k:row[k] for k in ('file','id','tier','logical_rank')}))+'\n')
            for row in rows:
                total += torch.tensor([row['loss'],row['cfm'],1.],device=device)
            if rank == 0:
                keys = ('cfm','rollout_physics','coordinate_es','feature_es','endpoint_weight')
                auxiliary_rows.append(dict(updates=updates,auxiliary=batch['rollout'],
                    logging_scope='physical_rank_0_prefix_mean',
                    **{key:float(np.mean([r[key] for r in rows])) for key in keys if key in rows[0]}))
                with (args.output/'objectives.jsonl').open('a') as stream:
                    stream.write(json.dumps(auxiliary_rows[-1])+'\n')
                coefficient = min(1.,config['gradient_clip']/(norm+1e-6))
                clip_row = dict(epoch=epoch,updates=updates,gradient_norm=norm,coefficient=coefficient,
                    active=coefficient<1.,logical_ranks=logical_size)
                clip_rows.append(clip_row)
                with (args.output/'clipping.jsonl').open('a') as stream:
                    stream.write(json.dumps(clip_row)+'\n')
            if epoch_update in checkpoint_updates:
                fraction = checkpoint_updates[epoch_update]
                if rank == 0:
                    candidate = dict(spec=spec, model=model.state_dict(), epoch=epoch, updates=updates,
                        checkpoint_kind='generation_evaluation', epoch_fraction=fraction,
                        processed_prefixes=processed, world_size=world,logical_world_size=logical_size,
                        config=config, cache_config=manifest['config'],
                        source_initial=initial['source_initial'] if args.resume else str(args.initial.resolve()),
                        final_generation_selection_pending=True)
                    candidate['seen_train_systems'] = len({Path(file).stem.rsplit('_',1)[0]
                                                         for file in planned_files[:processed]})
                    name = f'epoch_{epoch:03d}_candidate_{round(fraction*100):03d}.pt'
                    torch.save(candidate, args.output/(name+'.tmp'))
                    (args.output/(name+'.tmp')).replace(args.output/name)
                    with (args.output/'candidates.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(path=str((args.output/name).resolve()),
                            updates=updates, epoch=epoch, fraction=fraction,
                            seen_train_systems=candidate['seen_train_systems'],
                            processed_prefixes=candidate['processed_prefixes']))+'\n')
                if world > 1: dist.barrier()
            if rank == 0 and (updates % 8 == 0 or epoch_update == len(batches)):
                progress = dict(epoch=epoch, updates=updates, prefixes_done=processed,
                    prefixes_total=len(planned_files), stage='paired_adaptation' if diagnostic else 'complete_epoch',
                    geometry_update=use_geometry, gradient_norm=float(norm),
                    elapsed_seconds=time.perf_counter()-start)
                (args.output/'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
            if plan and (epoch_update%snapshot_every==0 or epoch_update==len(batches) or epoch_update==args.replay_through):
                snapshot(epoch_update)
            if epoch_update==args.replay_through:
                if rank == 0:
                    (args.output/'completion.json').write_text(json.dumps(dict(completed=True,
                        completion_reason='diagnostic_replay_boundary',epoch=epoch,updates=updates,
                        stage_update=epoch_update,converged=False),indent=2))
                if world > 1: dist.destroy_process_group()
                return
        model.eval()
        validation = torch.zeros(2, device=device, dtype=torch.float64)
        with torch.no_grad():
            for file in records['validation'][rank::world]:
                record = torch.load(file, weights_only=False, map_location='cpu')
                data = device_inputs(record, device)
                target = record['X_future'].to(device=device, dtype=torch.float32)
                rng = torch.Generator(device=device).manual_seed(case_seed(config['validation_seed'], record['meta']))
                noise = model.source_noise(data, 1, len(target), rng)
                s = torch.rand(1, generator=rng, device=device)
                loss = model(data, target, noise, s)['cfm']
                assert torch.isfinite(loss)
                validation += torch.tensor([float(loss), 1.], device=device)
        if world > 1:
            dist.all_reduce(total); dist.all_reduce(validation)
        score = float(validation[0]/validation[1])
        if plan is None: scheduler.step(score)
        improved = score < best
        best = min(best, score)
        local_state = capture_streams(streams,stream_states)
        gathered_states = [None]*world
        if world > 1: dist.all_gather_object(gathered_states,local_state)
        else: gathered_states[0]=local_state
        combined_states = {slot:value for group in gathered_states for slot,value in group.items()}
        rank_states = [combined_states[slot] for slot in range(logical_size)]
        if rank == 0:
            measured = dict(epoch=epoch, updates=updates, training_cfm=float(total[1]/total[2]),
                training_total=float(total[0]/total[2]), validation_cfm=score, epoch_seconds=time.perf_counter()-start,
                total_seconds=time.perf_counter()-train_start, learning_rate=optimizer.param_groups[0]['lr'])
            coefficients = np.array([row['coefficient'] for row in clip_rows])
            measured['clipping'] = dict(updates=len(clip_rows), activated_fraction=float((coefficients < 1.).mean()),
                coefficient_minimum=float(coefficients.min()), coefficient_median=float(np.median(coefficients)),
                coefficient_p05=float(np.quantile(coefficients, .05)),
                fraction_below_0_1=float((coefficients < .1).mean()),
                source='global gradient norm after DDP averaging, before clipping; PyTorch norm + 1e-6')
            if aligned_config:
                expected_rollout = [f for b in batches if b['rollout'] for f in b['files'] if f is not None]
                if expected_rollout:
                    seen = [json.loads(line)['file'] for r in range(world)
                        for line in (args.output/f'rollout_members_rank{r}.jsonl').read_text().splitlines()
                        if json.loads(line)['epoch'] == epoch]
                    if resume_progress:
                        seen += [f for b in batches[:cursor] if b['rollout'] for f in b['files'] if f is not None]
                    assert sorted(seen) == sorted(expected_rollout)
                measured['aligned'] = dict(auxiliary_updates=sum(row['auxiliary'] for row in auxiliary_rows),
                    rollout_steps=aligned_config['rollout_steps'], paths=2, weights=calibration['weights'],
                    endpoint_weight_last=auxiliary_rows[-1]['endpoint_weight'],
                    initial_optimizer=('shared C100 moments by parameter name' if optimizer_transfer else
                        'restored complete aligned epoch' if args.resume else 'fresh AdamW'),
                    rollout_prefixes=len(expected_rollout),cfm_prefixes=len(planned_files),
                    complete_cfm_prefix_coverage=set(planned_files)==set(records['train']),
                    complete_rollout_prefix_coverage=set(expected_rollout)==set(records['train']))
            saved = state_payload(rank_states,'stage_boundary' if diagnostic else 'complete_epoch')
            saved['validation'] = measured
            saved['exposures'] = exposure_counts(batches,len(batches),previous_exposures)
            torch.save(saved, args.output/'latest.tmp.pt'); (args.output/'latest.tmp.pt').replace(args.output/'latest.pt')
            if improved: shutil.copy2(args.output/'latest.pt', args.output/'cfm_candidate.pt')
            with (args.output/'training.jsonl').open('a') as file: file.write(json.dumps(measured)+'\n')
            print(json.dumps(measured), flush=True)
        # A complete-epoch measurement stage is separate from scientific convergence.
        measured_complete = config['stage'] == 'throughput_measurement' and epoch >= config['measurement_epochs']
        production_complete = config['stage'] == 'production' and epoch-start_epoch >= config['production_epochs']
        converged = optimizer.param_groups[0]['lr'] <= config['minimum_learning_rate'] and scheduler.num_bad_epochs >= config['plateau_patience']
        del wrapped
        if plan is not None or measured_complete or production_complete or converged: break
        previous_exposures = exposure_counts(batches,len(batches),previous_exposures)
    if rank == 0:
        (args.output/'completion.json').write_text(json.dumps(dict(completed=True, epoch=epoch, updates=updates,
            start_epoch=start_epoch, stage_epochs=epoch-start_epoch,
            converged=converged, stage=config['stage'],
            completion_reason=('complete_paired_adaptation' if diagnostic else 'complete_planned_epoch') if plan else
                ('validation_plateau' if converged else 'complete_planned_epochs'),
            final_generation_selection_pending=True), indent=2))
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
