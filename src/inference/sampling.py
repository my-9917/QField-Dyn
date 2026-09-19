"""Stable member seeds and independent raw path sampling; no file I/O."""
import numpy as np
import torch


def case_seed(seed, meta):
    return int(np.random.SeedSequence([seed, *map(ord, meta['id']+':'+meta['tier'])]).generate_state(1, dtype=np.uint64)[0])



@torch.no_grad()
def sample_paths(model, data, meta, seeds, batch_size=1):
    """Production evaluates one path per call; batch_size supports the numerical comparison."""
    static, history = model.condition(data)
    paths = []
    for begin in range(0, len(seeds), batch_size):
        noise = torch.cat([model.source_noise(data, 1, meta['n_pred'],
            torch.Generator(device=data['X_obs'].device).manual_seed(seed))
            for seed in seeds[begin:begin+batch_size]])
        generated = model.flow.sample(noise, static, history, model.solver_steps)['X_gen']
        assert torch.isfinite(generated).all()
        paths.append(generated.cpu().numpy())
    return np.concatenate(paths)
