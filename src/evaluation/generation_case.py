"""Generate and score one prefix with the frozen independent path sampler."""
import time
from semiflexible_model import device_inputs
from semiflexible_scores import score_paths
from sampling import case_seed, sample_paths


def evaluate_prefix(model, record, source_record, checkpoint, scales, seed, paths):
    data=device_inputs(record,'cuda')
    seeds=[case_seed(seed+i,record['meta']) for i in range(paths)]
    started=time.perf_counter()
    coordinates=sample_paths(model,data,record['meta'],seeds)
    generation_seconds=time.perf_counter()-started
    started=time.perf_counter()
    metrics,detail=score_paths(coordinates,record,scales)
    row=dict(id=record['meta']['id'],tier=record['meta']['tier'],paths=paths,metrics=metrics,
        generation_seconds=generation_seconds,scoring_seconds=time.perf_counter()-started)
    artifact=dict(meta=record['meta'],paths=coordinates,seeds=seeds,metrics=metrics,detail=detail,
        checkpoint=str(checkpoint),source_record=str(source_record))
    return artifact,row
