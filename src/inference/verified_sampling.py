"""Choose integration resolution from the delivered coordinates of one trajectory."""
import numpy as np
import torch
import time
from projection_encoding import encode_coordinates


@torch.no_grad()
def sample_adapted_path(base, adapter, noise, static, quantum, context, record,
                        tolerance=0.01, raw_cache=None, feedback=None, progress=None, feedback_start=None):
    """Accept the first resolution whose raw and delivered paths agree with doubling.

    Noise, observations, weights and output processing stay fixed during refinement.
    A caller may share raw integrals across candidate adapters for this same draw.
    Each adapter accepts its own first qualifying resolution, independently.
    """
    raw_cache={} if raw_cache is None else raw_cache
    timings=dict(flow_seconds=0.,adapter_seconds=0.,feedback_with_encoding_seconds=0.,encoding_seconds=0.)
    started=time.perf_counter();cached=[]
    def evaluate(steps):
        begin=time.perf_counter()
        if steps not in raw_cache:
            raw_cache[steps]=base.flow.sample(noise,static,quantum,steps)['X_gen']
            if noise.is_cuda:torch.cuda.synchronize(noise.device)
            timings['flow_seconds']+=time.perf_counter()-begin
        else:cached.append(steps)
        raw=raw_cache[steps];assert raw.shape[0]==1 and torch.isfinite(raw).all()
        begin=time.perf_counter()
        adapted,_=adapter(raw,context);assert torch.isfinite(adapted).all()
        if noise.is_cuda:torch.cuda.synchronize(noise.device)
        timings['adapter_seconds']+=time.perf_counter()-begin
        begin=time.perf_counter()
        if feedback is None:
            coordinates=adapted[0]
            encoded,_=encode_coordinates(coordinates.cpu().numpy()[None],record)
            encoded=encoded[0];repair=None
            timings['encoding_seconds']+=time.perf_counter()-begin
        else:
            from t4_geometry_feedback import restore_feedback
            coordinates,encoded,repair=restore_feedback(adapted[0],record,*feedback,initial_coordinates=feedback_start)
            timings['feedback_with_encoding_seconds']+=time.perf_counter()-begin
        return dict(coordinates=coordinates,encoded=encoded,raw=raw[0],adapter=adapted[0],repair=repair)
    steps=base.solver_steps;coarse=evaluate(steps);checks=[]
    while True:
        fine=evaluate(steps*2)
        raw_error=float((coarse['raw']-fine['raw']).square().mean().sqrt())
        output_error=float(np.sqrt(np.square(coarse['encoded']-fine['encoded']).mean()))
        check=dict(coarse_steps=steps,fine_steps=steps*2,raw_rms_angstrom=raw_error,
                   delivered_rms_angstrom=output_error,passed=max(raw_error,output_error)<=tolerance)
        checks.append(check)
        if progress:progress(dict(stage='integration_resolution',**check))
        if check['passed']:
            return dict(coarse,steps=steps,resolution_checks=checks,tolerance_angstrom=tolerance,
                timing=dict(timings,total_sampling_seconds=time.perf_counter()-started,cached_raw_resolutions=cached))
        # Successive growth of the raw integration discrepancy indicates loss of
        # numerical convergence. Preserve the trace instead of accepting that path.
        if len(checks)>=4 and all(checks[i]['raw_rms_angstrom']>=checks[i-1]['raw_rms_angstrom'] for i in range(len(checks)-3,len(checks))):
            raise RuntimeError(dict(reason='raw integration refinement stagnated',checks=checks))
        coarse=fine;steps*=2
