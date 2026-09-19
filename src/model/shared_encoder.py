"""Shared framewise invariant geometry encoder with separate static projection."""
import torch
from torch import nn


def mlp(source, hidden, target):
    return nn.Sequential(nn.Linear(source, hidden), nn.SiLU(), nn.Linear(hidden, target))


def spatial_edges(positions, environment, bonds, cutoff):
    """Directed ligand edges and static protein/ion-to-ligand contact edges."""
    frames, atoms, _ = positions.shape
    pair_distance = torch.cdist(positions, positions, compute_mode='donot_use_mm_for_euclid_dist')
    active = ((pair_distance < cutoff) | (bonds[None] > 0)) & ~torch.eye(atoms, dtype=torch.bool, device=positions.device)[None]
    f, target, source = active.nonzero(as_tuple=True)
    vector = positions[f, source]-positions[f, target]
    cross_distance = torch.cdist(positions, environment[None], compute_mode='donot_use_mm_for_euclid_dist')
    cf, ct, cs = (cross_distance < cutoff).nonzero(as_tuple=True)
    cross_vector = environment[cs]-positions[cf, ct]
    return {'frame': torch.cat((f, cf)), 'target': torch.cat((target, ct)),
        'source': torch.cat((source, cs+atoms)), 'vector': torch.cat((vector, cross_vector)),
        'order': torch.cat((bonds[target, source], bonds.new_zeros(len(cs)))),
        'target_flat': torch.cat((f*atoms+target, cf*atoms+ct))}


class FrameEncoder(nn.Module):
    def __init__(self, hidden=128, layers=3, cutoff=6.):
        super().__init__()
        self.hidden, self.cutoff = hidden, cutoff
        self.element = nn.Embedding(119, hidden)
        self.role = nn.Embedding(3, hidden)  # ligand, protein, ion
        self.ion_charge = mlp(1, hidden, hidden)
        self.chemistry = mlp(4, hidden, hidden)
        self.condition = mlp(10, hidden, hidden)
        self.displacement = mlp(1, hidden, hidden)
        self.messages = nn.ModuleList(mlp(2*hidden+4, hidden, hidden) for _ in range(layers))
        self.updates = nn.ModuleList(mlp(2*hidden, hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.static_projection = mlp(2*hidden, hidden, 12)
        self.dynamic_projection = mlp(2*hidden, hidden, 12)

    def frame(self, data, current, previous, previous_environment=None):
        environment = torch.cat((data['P0'], data['I0']))
        context = self.condition(data['condition'])
        ligand = self.element(data['ligand_numbers'])+self.role.weight[0]+self.chemistry(data['chemistry'])+context
        ligand = ligand+self.displacement(torch.linalg.vector_norm(current-previous, dim=-1, keepdim=True))
        p = self.element(data['protein_numbers'])+self.role.weight[1]+context
        ions = (self.element(data['ion_numbers']) + self.role.weight[2] + context
            + self.ion_charge(data['ion_charge_e'][:, None]))
        environment_features = torch.cat((p, ions))
        edges = spatial_edges(current[None], environment, data['bonds'], self.cutoff)
        target, source = edges['target'], edges['source']
        before = torch.cat((previous, environment if previous_environment is None else previous_environment))
        previous_vector = before[source]-previous[target]
        feature = torch.stack((torch.linalg.vector_norm(edges['vector'], dim=-1),
            torch.linalg.vector_norm(previous_vector, dim=-1), edges['order'],
            (edges['vector']*previous_vector).sum(-1)/(1+(previous_vector.square().sum(-1)*edges['vector'].square().sum(-1)).sqrt())), -1)
        count = torch.bincount(target, minlength=len(current)).clamp_min(1)[:, None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            all_hidden = torch.cat((ligand, environment_features))
            values = message(torch.cat((ligand[target], all_hidden[source], feature), -1))
            aggregate = torch.zeros_like(ligand).index_add(0, target, values)/count
            ligand = norm(ligand+update(torch.cat((ligand, aggregate), -1)))
        pooled = torch.cat((ligand.mean(0), ligand.max(0).values))
        return ligand, environment_features, pooled

    def static_features(self, data):
        observed = data['X_obs']
        ligand, environment, pooled = self.frame(data, observed[-1], observed[-1])
        c0 = self.static_projection(pooled)
        static = {'X_last': observed[-1], 'P0': data['P0'], 'I0': data['I0'],
            'E0': torch.cat((data['P0'], data['I0'])), 'ligand_features': ligand,
            'environment_features': environment, 'bonds': data['bonds'], 'condition': data['condition'], 'dt_ps': data['dt_ps']}
        return static, c0

    def forward(self, data):
        observed = data['X_obs']
        static, c0 = self.static_features(data)
        z = []
        for index, current in enumerate(observed):
            before=max(0,index-1)
            _, _, dynamic = self.frame(data, current, observed[before])
            z.append(self.dynamic_projection(dynamic))
        return {'static': static, 'c0_raw': c0, 'z_raw': torch.stack(z)}


class DescriptorAngles(nn.Module):
    def __init__(self, z_mean, z_scale, c_mean, c_scale):
        super().__init__()
        for name, value in zip(('z_mean', 'z_scale', 'c_mean', 'c_scale'), (z_mean, z_scale, c_mean, c_scale)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def forward(self, encoded):
        z = torch.pi*torch.tanh((encoded['z_raw']-self.z_mean)/self.z_scale)
        c = torch.pi*torch.tanh((encoded['c0_raw']-self.c_mean)/self.c_scale)
        return z, c
