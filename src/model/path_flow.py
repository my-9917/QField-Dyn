"""Ligand whole-future flow with a fixed environment and physical time in ns."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from shared_encoder import mlp, spatial_edges


class PathLayer(nn.Module):
    def __init__(self, hidden, anchor_enabled=True, update_hidden=True, anchor_mode='atomwise', global_position=False,
                 directional_context=False):
        super().__init__()
        self.anchor_enabled = anchor_enabled
        self.update_hidden = update_hidden
        self.anchor_mode = anchor_mode
        self.history = nn.Linear(18, hidden)
        self.spatial_message = mlp(2*hidden+2, hidden, hidden)
        self.spatial_vector = mlp(hidden, hidden, 1)
        self.temporal_message = mlp(2*hidden+2, hidden, hidden)
        self.temporal_vector = mlp(hidden, hidden, 1)
        self.update = mlp(3*hidden, hidden, hidden)
        self.anchor = mlp(hidden, hidden, 1)
        self.anchor.requires_grad_(anchor_enabled)
        self.norm = nn.LayerNorm(hidden)
        self.update.requires_grad_(update_hidden)
        self.norm.requires_grad_(update_hidden)
        self.global_position = global_position
        self.directional_context = directional_context
        if global_position:
            channels = 3 if directional_context else 2
            self.global_gate = mlp(hidden+channels, hidden, channels)
            nn.init.zeros_(self.global_gate[-1].weight)
            nn.init.zeros_(self.global_gate[-1].bias)

    def temporal_block(self, hidden, positions, targets, dt_ps):
        """Exact all-time messages for one block of receiving frames."""
        paths, horizon, atoms, _ = hidden.shape
        t = targets.repeat_interleave(horizon)
        u = torch.arange(horizon, device=hidden.device).repeat(len(targets))
        selected = t != u
        t, u = t[selected], u[selected]
        vector = positions[:, u]-positions[:, t]
        lag = ((u-t)*dt_ps/1000.).to(hidden.dtype)[None, :, None].expand(paths, -1, atoms)
        features = torch.stack((torch.linalg.vector_norm(vector, dim=-1), lag), -1)
        message = self.temporal_message(torch.cat((hidden[:, t], hidden[:, u], features), -1))
        temporal = message.reshape(paths, len(targets), horizon-1, atoms, -1).mean(2)
        velocity = (self.temporal_vector(message)*vector).reshape(paths, len(targets), horizon-1, atoms, 3).mean(2)
        return temporal, velocity

    def temporal_aggregate(self, hidden, positions, dt_ps, chunk_size=4):
        """Retain every time pair while bounding intermediate memory."""
        parts = []
        for targets in torch.arange(hidden.shape[1], device=hidden.device).split(chunk_size):
            if torch.is_grad_enabled():
                part = checkpoint(self.temporal_block, hidden, positions, targets, dt_ps,
                                  use_reentrant=False, preserve_rng_state=False)
            else:
                part = self.temporal_block(hidden, positions, targets, dt_ps)
            parts.append(part)
        return tuple(torch.cat(values, dim=1) for values in zip(*parts))

    def spatial_block(self, flat, environment, f, target, source, vector, order):
        scalar = torch.stack((torch.linalg.vector_norm(vector, dim=-1), order), -1)
        atoms = flat.shape[1]
        source_hidden = flat[f, source.clamp_max(atoms-1)]
        cross = (source >= atoms).nonzero(as_tuple=True)[0]
        source_hidden = source_hidden.index_copy(0, cross, environment[source[cross]-atoms])
        message = self.spatial_message(torch.cat((flat[f, target], source_hidden, scalar), -1))
        return message, self.spatial_vector(message)*vector

    def forward(self, hidden, positions, static, history, edges):
        paths, horizon, atoms, width = hidden.shape
        hidden = hidden+self.history(history)[None, None, None]
        flat = hidden.reshape(paths*horizon, atoms, width)
        environment = static['environment_features']
        f, target, source = edges['frame'], edges['target'], edges['source']
        count = torch.bincount(edges['target_flat'], minlength=paths*horizon*atoms).clamp_min(1)[:, None]
        spatial = hidden.new_zeros(paths*horizon*atoms, width)
        spatial_v = hidden.new_zeros(paths*horizon*atoms, 3)
        for start in range(0, len(f), 32768):
            sl = slice(start, start+32768)
            args = (flat, environment, f[sl], target[sl], source[sl], edges['vector'][sl], edges['order'][sl])
            if torch.is_grad_enabled():
                message, vector = checkpoint(self.spatial_block, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                message, vector = self.spatial_block(*args)
            spatial = spatial.index_add(0, edges['target_flat'][sl], message)
            spatial_v = spatial_v.index_add(0, edges['target_flat'][sl], vector)
        spatial, spatial_v = spatial/count, spatial_v/count
        temporal, temporal_v = self.temporal_aggregate(hidden, positions, static['dt_ps'])
        updated = self.norm(hidden+self.update(torch.cat((hidden, spatial.reshape_as(hidden), temporal), -1))) if self.update_hidden else hidden
        anchor = 0.
        if self.anchor_enabled:
            delta = static['X_last'][None, None]-positions
            alpha = self.anchor(updated)
            if self.anchor_mode == 'centred_shared':
                delta = delta-delta.mean(-2, keepdim=True)
                alpha = alpha.mean(-2, keepdim=True)
            anchor = alpha*delta
        velocity = spatial_v.reshape_as(positions)+temporal_v+anchor
        if self.global_position:
            centre = (positions*static['ligand_mass_fraction'][None, None, :, None]).sum(-2)
            if self.directional_context:
                vectors = torch.stack((centre-static['ligand_reference_centre'],
                    (centre-static['pocket_centre'])*static['pocket_valid'],
                    static['observed_translation'].expand_as(centre)), -2)
                distance = torch.linalg.vector_norm(vectors, dim=-1)/static['global_length_scale']
                carriers = vectors/torch.sqrt(1+distance.square())[..., None]
                gates = self.global_gate(torch.cat((hidden.mean(-2), torch.log1p(distance)), -1))
                velocity = velocity+(gates[..., None]*carriers).sum(-2)[:, :, None]
            else:
                vectors = torch.stack((centre-static['ligand_reference_centre'], centre-static['pocket_centre']), -2)
                distances = torch.linalg.vector_norm(vectors, dim=-1)
                gates = self.global_gate(torch.cat((hidden, distances[:, :, None].expand(-1, -1, atoms, -1)), -1))
                velocity = velocity+(gates[..., None]*vectors[:, :, None]).sum(-2)
        return updated, velocity


class PathFlow(nn.Module):
    def __init__(self, hidden=128, layers=3, cutoff=6., anchor_enabled=True, anchor_mode='atomwise', global_position=False,
                 directional_context=False):
        super().__init__()
        self.cutoff = cutoff
        self.time = mlp(4, hidden, hidden)
        # Hidden updates feed the next layer or the current layer's anchor velocity.
        self.layers = nn.ModuleList(PathLayer(hidden, anchor_enabled,
            update_hidden=(index < layers-1 or anchor_enabled), anchor_mode=anchor_mode,
            global_position=global_position, directional_context=directional_context) for index in range(layers))

    def velocity(self, displacement, flow_time, static, history):
        paths, horizon, atoms, _ = displacement.shape
        positions = displacement+static['X_last'][None, None]
        s = torch.as_tensor(flow_time, dtype=positions.dtype, device=positions.device).expand(paths)
        dt = static['dt_ps']/1000.
        lead = torch.arange(1, horizon+1, dtype=positions.dtype, device=positions.device)*dt
        time = torch.stack((s[:, None].expand(-1, horizon), (1-s)[:, None].expand(-1, horizon),
            lead[None].expand(paths, -1), torch.ones(paths, horizon, device=positions.device, dtype=positions.dtype)*dt), -1)
        hidden = static['ligand_features'][None, None].expand(paths, horizon, -1, -1)+self.time(time)[:, :, None]
        edges = spatial_edges(positions.reshape(paths*horizon, atoms, 3), static['E0'], static['bonds'], self.cutoff)
        velocity = torch.zeros_like(displacement)
        for layer in self.layers:
            hidden, contribution = layer(hidden, positions, static, history, edges)
            velocity = velocity+contribution/len(self.layers)
        return velocity

    def loss(self, target, noise, flow_time, static, history, sigma):
        """Conditional straight-line path, using s as interpolation time only."""
        endpoint = target-static['X_last'][None]
        s = torch.as_tensor(flow_time, dtype=noise.dtype, device=noise.device)[:, None, None, None]
        interpolated = (1-s)*noise+s*endpoint[None]
        truth = endpoint[None]-noise
        predicted = self.velocity(interpolated, flow_time, static, history)
        return ((predicted-truth)/sigma).square().mean(), predicted, truth

    def integrate(self, noise, static, history, steps):
        """Ligand endpoint integration; caller autograd context controls gradients."""
        displacement = noise.clone()
        step = 1./steps
        for index in range(steps):
            s = index*step
            if torch.is_grad_enabled():
                first = checkpoint(self.velocity, displacement, s, static, history,
                                   use_reentrant=False, preserve_rng_state=False)
                second = checkpoint(self.velocity, displacement+step*first, s+step, static, history,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                first = self.velocity(displacement, s, static, history)
                second = self.velocity(displacement+step*first, s+step, static, history)
            displacement = displacement+.5*step*(first+second)
        return {'X_gen': displacement+static['X_last'][None, None],
            'P_gen': static['P0'][None, None].expand(len(noise), noise.shape[1], -1, -1),
            'I_gen': static['I0'][None, None].expand(len(noise), noise.shape[1], -1, -1)}

    @torch.no_grad()
    def sample(self, noise, static, history, steps):
        """Heun inference for a complete future displacement path."""
        return self.integrate(noise, static, history, steps)
