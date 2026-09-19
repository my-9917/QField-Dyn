"""Frozen complete-prefix batches and generation-quality stopping state."""
import random
from collections import defaultdict
from pathlib import Path


def rollout_cost(row):
    horizon = {'T1':10,'T2':20,'T3':80}[row['tier']]
    return row['atoms']*horizon*(horizon-1)


def balanced_rollout_batches(rows, rng):
    """Mix tier pairs while giving both GPUs comparable accumulation workloads."""
    pairs = {}
    for tier in ('T1','T2','T3'):
        group = [r for r in rows if r['tier']==tier]
        rng.shuffle(group)
        group.sort(key=rollout_cost)
        assert len(group)%2==0
        pairs[tier] = [group[i:i+2] for i in range(0,len(group),2)]
        rng.shuffle(pairs[tier])
    assert len({len(v) for v in pairs.values()})==1
    order = list(pairs); rng.shuffle(order)
    ordered = [pairs[tier][i] for i in range(len(pairs[order[0]])) for tier in order]
    assert len(ordered)%2==0
    batches = []
    for i in range(0,len(ordered),2):
        a,b = ordered[i:i+2]
        # Physical rank 0 consumes slots 0,2; rank 1 consumes slots 1,3.
        block = [a[0],a[1],b[1],b[0]]
        batches.append(dict(files=[r['file'] for r in block],rollout=True))
    rng.shuffle(batches)
    return batches


def interleave_rollout(cheap, expensive):
    """Place rollout updates at evenly spaced positions in the full epoch."""
    total = len(cheap)+len(expensive)
    positions = {(2*i+1)*total//(2*len(expensive)) for i in range(len(expensive))}
    assert len(positions)==len(expensive)
    c,r = iter(cheap),iter(expensive)
    return [next(r) if i in positions else next(c) for i in range(total)]


def exposure_counts(batches, through, previous):
    counts = dict(previous)
    counts['rollout_by_tier'] = dict(previous['rollout_by_tier'])
    for batch in batches[:through]:
        for file in batch['files']:
            if file is None: continue
            counts['cfm']+=1
            if batch['rollout']:
                counts['rollout']+=1
                counts['rollout_by_tier'][Path(file).stem.rsplit('_',1)[1]]+=1
    return counts


def pack_batches(rows, rollout, rng, logical_size=4):
    """Keep costly prefixes together and group comparable time/atom workloads."""
    by_tier = defaultdict(list)
    for row in rows:
        by_tier[row['tier']].append(row)
    batches = []
    tail = []
    for tier in sorted(by_tier):
        values = by_tier[tier]
        rng.shuffle(values)
        values.sort(key=lambda r:r['atoms'])
        end = len(values)-len(values)%logical_size
        for start in range(0,end,logical_size):
            block = values[start:start+logical_size]
            rng.shuffle(block)
            batches.append(dict(files=[r['file'] for r in block],rollout=rollout))
        tail.extend(values[end:])
    rng.shuffle(tail)
    for start in range(0,len(tail),logical_size):
        files = [r['file'] for r in tail[start:start+logical_size]]
        batches.append(dict(files=files+[None]*(logical_size-len(files)),rollout=rollout))
    rng.shuffle(batches)
    return batches


def stratified_prefixes(rows, count, seed):
    """Equal strata sampling for the paired diagnostic; labels are train-only."""
    rng = random.Random(seed)
    cells = defaultdict(list)
    for tier in ('T1','T2','T3'):
        group = [r for r in rows if r['tier']==tier]
        ranked = sorted(group,key=lambda r:(r['rmsf'],r['id']))
        motion = {r['id']:min(3,4*i//len(ranked)) for i,r in enumerate(ranked)}
        ranked = sorted(group,key=lambda r:(r['atoms'],r['id']))
        size = {r['id']:min(2,3*i//len(ranked)) for i,r in enumerate(ranked)}
        for row in group:
            cells[(tier,size[row['id']],motion[row['id']])].append(row)
    for group in cells.values(): rng.shuffle(group)
    selected = []
    while len(selected)<count:
        keys = [k for k,v in cells.items() if v]; rng.shuffle(keys)
        for key in keys:
            selected.append(cells[key].pop())
            if len(selected)==count: break
    return selected


def diversity_labels(row):
    labels=[('stratum',row['stratum']),('protein',row['protein_group']),('scaffold',row['scaffold']),
            ('chemistry',row['chemistry_bin'])]
    labels += [('behaviour',v) for v in row['behaviours']]
    if 'pocket_group' in row: labels.append(('pocket',row['pocket_group']))
    return labels


def rotating_systems(population, used, count, seed, initial=()):
    """Stratified rotation with train-defined behaviour and diversity balancing."""
    rng = random.Random(seed)
    selected, counts = list(initial),defaultdict(int)
    selected_ids={r['id'] for r in selected}
    remaining = [r for r in population if r['id'] not in selected_ids]; rng.shuffle(remaining)
    for row in selected:
        for label in diversity_labels(row):counts[label]+=1
    while len(selected)<count:
        minimum = min(used.get(r['id'],0) for r in remaining)
        eligible = [r for r in remaining if used.get(r['id'],0)==minimum]
        def score(row):
            return sum(counts[label] for label in diversity_labels(row))
        chosen = min(eligible,key=score)
        selected.append(chosen); remaining.remove(chosen)
        for label in diversity_labels(chosen):
            counts[label]+=1
    state = dict(used)
    for row in selected: state[row['id']]=state.get(row['id'],0)+1
    return selected,state


def fixed_rollout_subsets(population, count, seeds):
    """Reserve rare strata for each disjoint round before diversity balancing."""
    used=set();subsets=[]
    for round_index,seed in enumerate(seeds):
        rng=random.Random(seed);cells=defaultdict(list)
        for row in population:
            if row['id'] not in used:cells[row['stratum']].append(row)
        assert set(cells)=={r['stratum'] for r in population}
        reserved=len(seeds)-round_index-1
        pool=[];initial=[]
        for key in sorted(cells):
            rows=cells[key];rng.shuffle(rows)
            assert len(rows)>reserved,(key,len(rows),reserved)
            available=rows[:len(rows)-reserved]
            pool.extend(available);initial.append(available[0])
        selected,_=rotating_systems(pool,{},count,seed,initial=initial)
        identifiers=sorted(r['id'] for r in selected)
        assert used.isdisjoint(identifiers)
        used.update(identifiers);subsets.append(identifiers)
    return subsets


def update_stopping(state, score, qualified, checkpoint, learning_rates=(1e-4,3e-5,1e-5)):
    """Apply the 0917v4 rules after one complete 16x3x8 epoch evaluation."""
    s = dict(state)
    s['epochs']+=1
    s['epochs_at_level']+=1
    s['normal_epochs']+=int(s['lr_level']==0)
    s['low_epochs']+=int(s['lr_level']>0)
    s['unqualified_streak']=0 if qualified else s['unqualified_streak']+1
    if qualified and (s['best_score'] is None or score<s['best_score']):
        s.update(best_score=score,best_checkpoint=checkpoint)
    progress = qualified and (s['reference_score'] is None or
        (s['reference_score']-score)/max(abs(s['reference_score']),1e-12)>=.01)
    if progress:
        s.update(reference_score=score,patience=0)
    else:
        s['patience']+=1
    s['action']='continue'
    if s['unqualified_streak']>=3:
        s['action']='method_review'
    elif s['patience']>=3 and s['normal_epochs']>=2:
        if s['lr_level']<len(learning_rates)-1:
            s.update(lr_level=s['lr_level']+1,epochs_at_level=0,patience=0,action='reduce_lr')
        elif s['low_epochs']>=1:
            s['action']='plateau_complete'
    if s['epochs']>=8 and s['action'] in ('continue','reduce_lr'):
        s['action']='eight_epoch_review'
    s['next_learning_rate']=learning_rates[s['lr_level']]
    return s


def initial_stopping():
    return dict(epochs=0,epochs_at_level=0,normal_epochs=0,low_epochs=0,lr_level=0,
        best_score=None,best_checkpoint=None,reference_score=None,patience=0,unqualified_streak=0)


def update_stopping_v5(state, score, qualified, checkpoint):
    """Two normal-LR epochs, one qualified improving low-LR epoch, then review."""
    s=dict(state);prior=s.get('last_score');prior_qualified=s.get('last_qualified',False)
    s['epochs']+=1
    if qualified and (s['best_score'] is None or score<s['best_score']):
        s.update(best_score=score,best_checkpoint=checkpoint)
    progress=qualified and (s['reference_score'] is None or
        (s['reference_score']-score)/abs(s['reference_score'])>=.01)
    s['patience']=0 if progress else s['patience']+1
    if progress:s['reference_score']=score
    s['last_score']=score;s['last_qualified']=qualified
    s['unqualified_streak']=0 if qualified else s['unqualified_streak']+1
    s['normal_epochs']+=int(s['epochs']<=2);s['low_epochs']+=int(s['epochs']>=3)
    s['epochs_at_level']+=1;s['converged']=False
    s['action']='continue';s['next_learning_rate']=1e-4
    if s['epochs']==2:
        if qualified and prior_qualified and prior is not None and score<prior:
            s.update(action='low_lr_epoch_3',lr_level=1,epochs_at_level=0,next_learning_rate=3e-5)
        else:s['action']='two_epoch_quality_review'
    elif s['epochs']>=3:
        s['action']='fourth_epoch_review' if progress else 'planned_three_epochs_complete'
        s['next_learning_rate']=3e-5
        if s['best_checkpoint'] is None:s['action']='method_review'
    return s


def update_stopping_0918(state, score, output_eligible, checkpoint):
    """Train on the model's own ES trend; retain output eligibility separately."""
    s = dict(state)
    prior_best = s['best_score']
    s['epochs'] += 1
    s['last_score'], s['last_output_eligible'] = score, output_eligible
    s['normal_epochs'] += int(s['epochs'] <= 2)
    s['low_epochs'] += int(s['epochs'] >= 3)
    if prior_best is None or score < prior_best:
        s.update(best_score=score, best_checkpoint=checkpoint)
    prior_output=s.get('best_output_score')
    if output_eligible and (prior_output is None or score<prior_output):
        s.update(best_output_score=score,best_output_checkpoint=checkpoint)
    gain = (prior_best-score)/abs(prior_best) if prior_best is not None else None
    s.update(improvement_from_prior_best=gain, converged=False,
             next_learning_rate=1e-4 if s['epochs'] < 2 else 3e-5,
             lr_level=int(s['epochs'] >= 2), action='continue')
    if s['epochs'] >= 3:
        if gain >= .01:
            s['action'] = 'continue'
        else:
            s['action'] = 'generation_plateau_complete'
    s['plateau_scope'] = 'model projected Feature ES trend; baseline advantage and output eligibility are reported separately'
    return s


def update_stopping_0919(state, score, output_eligible, checkpoint):
    """Track candidate quality under the authorized fixed three-epoch schedule."""
    s = update_stopping_0918(state, score, output_eligible, checkpoint)
    assert s['epochs'] <= 3
    s.update(action='scheduled_training_complete' if s['epochs'] == 3 else 'continue',
             stopping_rule='scheduled_three_epochs_0919', planned_epochs=3, converged=False)
    s.pop('plateau_scope', None)
    s['stopping_reason'] = 'predefined three-epoch schedule' if s['epochs'] == 3 else 'scheduled epoch remaining'
    s.pop('generation_reviews_pending', None)
    return s
