"""Plan0 six-qubit memory. Inputs are encoded angles; q0 is the MSB."""
import numpy as np
import torch
from torch import nn
from angle_encoding import AffineAngles


EDGES = ((0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0))
READOUT_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))


def rotation(angle, axis):
    angle = angle.to(torch.float64)
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    zero = torch.zeros_like(c)
    if axis == 'X':
        entries = (c, -1j * s, -1j * s, c)
    elif axis == 'Y':
        entries = (c, -s, s, c)
    elif axis == 'Z':
        entries = (c - 1j * s, zero, zero, c + 1j * s)
    else:
        raise ValueError(axis)
    return torch.stack(entries, dim=-1).reshape(*angle.shape, 2, 2).to(torch.complex128)


def product_gate(local):
    result = local[..., 0, :, :]
    for i in range(1, 6):
        result = torch.einsum('...ab,...cd->...acbd', result, local[..., i, :, :])
        result = result.reshape(*local.shape[:-3], 2 ** (i + 1), 2 ** (i + 1))
    return result


class QuantumMemory(nn.Module):
    def __init__(self, gamma=0.1, entangler='CNOT', reservoir_seed=2026090701, train_seed=2026090702,
                 *, history_encoding='fixed', readout_encoding='fixed'):
        super().__init__()
        self.gamma = gamma
        self.entangler = entangler
        if history_encoding not in ('fixed', 'affine') or readout_encoding not in ('fixed', 'affine'):
            raise ValueError('Angle encoding must be fixed or affine')
        self.history_encoding = history_encoding
        self.readout_encoding = readout_encoding
        self.history_angles = AffineAngles((12,)) if history_encoding == 'affine' else nn.Identity()
        self.readout_angles = AffineAngles((3, 12)) if readout_encoding == 'affine' else nn.Identity()
        fixed = np.random.Generator(np.random.PCG64(reservoir_seed)).uniform(-np.pi, np.pi, (2, 6))
        theta = np.random.Generator(np.random.PCG64(train_seed)).normal(0, 0.1, (3, 6, 2))
        self.register_buffer('fixed_angles', torch.tensor(fixed, dtype=torch.float64))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float64))
        rho0 = torch.zeros(64, 64, dtype=torch.complex128)
        rho0[0, 0] = 1
        self.register_buffer('rho0', rho0)
        ring = torch.eye(64, dtype=torch.complex128)
        basis = torch.arange(64)
        for control, target in EDGES:
            if entangler == 'CNOT':
                permutation = basis ^ (((basis >> (5 - control)) & 1) << (5 - target))
                ring = ring[permutation]
            elif entangler == 'CZ':
                phase = 1 - 2 * (((basis >> (5 - control)) & 1) * ((basis >> (5 - target)) & 1))
                ring = phase[:, None] * ring
            else:
                raise ValueError(entangler)
        self.register_buffer('ring', ring)
        z_signs = torch.stack([1 - 2 * ((basis >> (5 - i)) & 1) for i in range(6)]).double()
        self.register_buffer('z_signs', z_signs)
        self.register_buffer('zz_signs', torch.stack([z_signs[i] * z_signs[j] for i, j in READOUT_EDGES]))
        self.register_buffer('x_indices', torch.stack([basis ^ (1 << (5 - i)) for i in range(6)]))
        h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
        self.register_buffer('all_h', product_gate(h.repeat(6, 1, 1)))

    def reservoir_unitary(self, z):
        """One time-input block: RY(z[0:6]), RZ(z[6:12]), fixed RX/RZ, entangler."""
        z = self.history_angles(z)
        local = rotation(self.fixed_angles[1], 'Z') @ rotation(self.fixed_angles[0], 'X')
        local = local @ rotation(z[..., 6:], 'Z') @ rotation(z[..., :6], 'Y')
        return self.ring @ product_gate(local)

    def readout_unitary(self, c):
        """Three uploads of c; each block is data RY/RZ, trained RY/RZ, entangler.

        Gates act right to left in matrix products. Layers share the evolving
        quantum state; each complete data block counts as one upload.
        """
        result = torch.eye(64, dtype=torch.complex128, device=c.device).expand(*c.shape[:-1], 64, 64)
        angles = self.readout_angles(c.unsqueeze(-2).expand(*c.shape[:-1], 3, 12))
        for layer in range(3):
            local = rotation(self.theta[layer, :, 1], 'Z') @ rotation(self.theta[layer, :, 0], 'Y')
            local = local @ rotation(angles[..., layer, 6:], 'Z') @ rotation(angles[..., layer, :6], 'Y')
            result = (self.ring @ product_gate(local)) @ result
        return result

    def reset_probability(self, dt_ps):
        """Gamma is defined over 200 ps; survival composes over physical time."""
        return 1 - (1 - self.gamma) ** (dt_ps / 200.)

    def encode(self, z, initial=None, dephase=False, *, dt_ps):
        rho = self.rho0.expand(z.shape[0], 64, 64) if initial is None else initial
        reset = self.reset_probability(dt_ps)
        for t in range(z.shape[1]):
            rho = (1 - reset) * rho + reset * self.rho0
            unitary = self.reservoir_unitary(z[:, t])
            rho = unitary @ rho @ unitary.mH
            if dephase:
                rho = torch.diag_embed(rho.diagonal(dim1=-2, dim2=-1))
        return rho

    def readout(self, rho, c):
        unitary = self.readout_unitary(c)
        state = unitary @ rho @ unitary.mH
        probabilities = state.diagonal(dim1=-2, dim2=-1).real
        z = probabilities @ self.z_signs.T
        zz = probabilities @ self.zz_signs.T
        indices = torch.arange(64, device=state.device)
        x = torch.stack([state[..., indices, flip].sum(-1).real for flip in self.x_indices], dim=-1)
        return torch.cat((z, x, zz), dim=-1)

    def forward(self, z, c, dt_ps):
        return self.readout(self.encode(z, dt_ps=dt_ps), c)

    @torch.no_grad()
    def sample_readout(self, z, c, shots, seed, dt_ps):
        """Each shot executes a whole-register Bernoulli reset at each step."""
        rng = torch.Generator(device=z.device).manual_seed(seed)
        unitaries = self.reservoir_unitary(z)
        readout = self.readout_unitary(c)
        answers = []
        for context in range(z.shape[0]):
            measured = []
            for basis in ('Z', 'X'):
                state = torch.zeros(shots, 64, dtype=torch.complex128, device=z.device)
                state[:, 0] = 1
                for t in range(z.shape[1]):
                    reset = torch.rand(shots, generator=rng, device=z.device) < self.reset_probability(dt_ps)
                    state[reset] = 0
                    state[reset, 0] = 1
                    state = state @ unitaries[context, t].T
                state = state @ readout[context].T
                if basis == 'X':
                    state = state @ self.all_h.T
                indices = torch.multinomial(state.abs().square(), 1, generator=rng).squeeze(-1)
                measured.append(indices)
            measured_z = self.z_signs[:, measured[0]].mean(-1)
            measured_x = self.z_signs[:, measured[1]].mean(-1)
            measured_zz = self.zz_signs[:, measured[0]].mean(-1)
            answers.append(torch.cat((measured_z, measured_x, measured_zz)))
        return torch.stack(answers)


class QuantumHistory(QuantumMemory):
    """One molecular history in, 18 float32 generator conditions out."""

    def forward(self, z, c, dt_ps):
        return super().forward(z[None].double(), c[None].double(), dt_ps)[0].to(z.dtype)
