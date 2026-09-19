"""Shared training and inference interface for the fixed-protein quantum model."""
import torch
from torch import nn
from shared_encoder import FrameEncoder, DescriptorAngles
from quantum_memory import QuantumHistory
from path_flow import PathFlow
from structured_motion import rigid_source, component_loss


class SemiFlexFlow(nn.Module):
    def __init__(self, model_config, angle_statistics, sigma, anchor_enabled=True, solver_steps=32,
                 motion_statistics=None, anchor_mode='atomwise', geometry_mode='integrated', global_position=False,
                 directional_statistics=None, translation_reference_ps=None):
        super().__init__()
        self.encoder = FrameEncoder(**model_config['encoder'])
        self.angles = DescriptorAngles(**angle_statistics)
        self.history = QuantumHistory(**model_config['history']['quantum'])
        self.flow = PathFlow(**model_config['generator'], anchor_enabled=anchor_enabled,
                             anchor_mode=anchor_mode, global_position=global_position,
                             directional_context=directional_statistics is not None)
        self.global_position = global_position
        self.directional_statistics = directional_statistics
        self.translation_reference_ps = translation_reference_ps
        if directional_statistics is not None:
            assert global_position and directional_statistics['partition'] == 'train'
            assert directional_statistics['global_length_scale'] > 0 and directional_statistics['coordinate_scale'] > 0
        if global_position:
            from rdkit import Chem
            table = Chem.GetPeriodicTable()
            self.register_buffer('atomic_masses', torch.tensor([0.]+[table.GetAtomicWeight(z) for z in range(1, 119)]))
        self.sigma = float(sigma)
        self.solver_steps = solver_steps
        self.motion_statistics = motion_statistics
        self.geometry_mode = geometry_mode
        assert anchor_mode in ('atomwise', 'centred_shared')
        assert geometry_mode in ('integrated', 'conditional_endpoint')
        if motion_statistics is not None:
            assert motion_statistics['completed'] and motion_statistics['partition'] == 'train'
            assert motion_statistics['version'] == 'rigid_source_v2'
        active_config = dict(version='semiflexible_model_v1',
            encoder=model_config['encoder'], history=model_config['history'], generator=model_config['generator'],
            state_blocks=['ligand'], environment='fixed_last_observed', coordinate_version='semiflexible_protein_aligned_v2',
            lead_time_unit='ns', temporal_lag_unit='ns')
        self.spec = dict(model_config=active_config, angle_statistics=angle_statistics,
                         sigma=self.sigma, anchor_enabled=anchor_enabled, solver_steps=solver_steps)
        if motion_statistics is not None or anchor_mode != 'atomwise' or geometry_mode != 'integrated':
            active_config['version'] = 'semiflexible_model_v2'
            self.spec.update(motion_statistics=motion_statistics, anchor_mode=anchor_mode, geometry_mode=geometry_mode)
        if global_position:
            active_config['version'] = 'semiflexible_aligned_v1'
            self.spec['global_position'] = True
        if directional_statistics is not None:
            active_config['version'] = 'semiflexible_aligned_v2'
            self.spec['directional_statistics'] = directional_statistics
        if translation_reference_ps is not None:
            assert directional_statistics is not None and translation_reference_ps > 0
            active_config['version'] = 'semiflexible_aligned_time_v3'
            self.spec['translation_reference_ps'] = float(translation_reference_ps)

    def position_context(self, static, data, reference=None):
        if self.global_position:
            if reference is not None:
                for key in ('ligand_mass_fraction', 'ligand_reference_centre', 'pocket_centre', 'observed_pocket_atoms'):
                    static[key] = reference[key]
                if self.directional_statistics is not None:
                    for key in ('observed_translation', 'global_length_scale', 'pocket_valid'):
                        static[key] = reference[key]
                return static
            masses = self.atomic_masses[data['ligand_numbers']]
            fraction = masses/masses.sum()
            last = static['X_last']
            protein = static['P0'][data['protein_numbers'] > 1]
            query = data['X_obs'][:, data['ligand_numbers'] > 1].reshape(-1, 3) if self.directional_statistics else last[data['ligand_numbers'] > 1]
            distance = protein.new_full((len(protein),), float('inf'))
            for block in query.split(512):
                distance = torch.minimum(distance, torch.cdist(block, protein,
                    compute_mode='donot_use_mm_for_euclid_dist').amin(0))
            pocket = protein[distance <= 6.]
            centre = (last*fraction[:, None]).sum(0)
            static.update(ligand_mass_fraction=fraction,
                ligand_reference_centre=centre,
                pocket_centre=pocket.mean(0) if len(pocket) else (centre if self.directional_statistics else protein.mean(0)),
                observed_pocket_atoms=len(pocket))
            if self.directional_statistics is not None:
                centres = (data['X_obs'][-5:]*fraction[None, :, None]).sum(-2)
                index = torch.arange(len(centres), device=last.device, dtype=last.dtype)
                index = index-index.mean()
                # Legacy weights use A/frame; the versioned input map uses A/reference interval.
                translation = (index[:, None]*(centres-centres.mean(0))).sum(0)/index.square().sum()
                if self.translation_reference_ps is not None:
                    translation = translation*(self.translation_reference_ps/data['dt_ps'])
                static.update(observed_translation=translation, pocket_valid=float(len(pocket) > 0),
                    global_length_scale=self.directional_statistics['global_length_scale'])
        return static

    def recurrent_static(self, data, reference):
        static, _ = self.encoder.static_features(data)
        return self.position_context(static, data, reference)

    def condition(self, data):
        encoded = self.encoder(data)
        self.position_context(encoded['static'], data)
        z, c = self.angles(encoded)
        history = self.history(z, c, dt_ps=data['dt_ps'])
        return encoded['static'], history

    def source_noise(self, data, paths, horizon, generator):
        if self.motion_statistics is not None:
            return rigid_source(data['X_obs'][-1], paths, horizon, generator, self.motion_statistics['scales'])
        return self.sigma*torch.randn(paths, horizon, len(data['X_obs'][0]), 3,
            generator=generator, device=data['X_obs'].device, dtype=data['X_obs'].dtype)

    def forward(self, data, target, noise, flow_time, geometry=None, aligned=None):
        static, history = self.condition(data)
        cfm, predicted, truth = self.flow.loss(target, noise, flow_time, static, history, self.sigma)
        if self.motion_statistics is not None:
            cfm = component_loss(predicted-truth, self.motion_statistics['scales'])
        terms = {'cfm': cfm}
        if geometry is not None:
            from ligand_geometry import geometry_terms
            if self.geometry_mode == 'conditional_endpoint':
                s = torch.as_tensor(flow_time, dtype=noise.dtype, device=noise.device)[:, None, None, None]
                interpolated = noise+s*truth
                generated = static['X_last'][None, None]+interpolated+(1-s)*predicted
            else:
                generated = self.flow.integrate(noise, static, history, self.solver_steps)['X_gen']
            terms.update({key: value.mean() for key, value in geometry_terms(generated, geometry).items()})
        if aligned is not None:
            from aligned_losses import rollout_objectives
            generated = self.flow.integrate(aligned['noise'], static, history, aligned['steps'])['X_gen']
            terms.update(rollout_objectives(generated, target, aligned['geometry'],
                aligned['features'], aligned['scales'], aligned['environment_epsilon'],
                coordinate_scale=self.directional_statistics['coordinate_scale'] if self.directional_statistics else None))
        return terms

    @torch.no_grad()
    def sample(self, data, noise, steps=None):
        static, history = self.condition(data)
        return self.flow.sample(noise, static, history, self.solver_steps if steps is None else steps)


def transfer_joint(checkpoint, device, anchor_enabled=True):
    model = SemiFlexFlow(checkpoint['model_config'], checkpoint['angle_statistics'],
                         checkpoint['flow_scales']['ligand'], anchor_enabled).to(device)
    for name in ('encoder', 'angles', 'history'):
        getattr(model, name).load_state_dict(checkpoint['models'][name], strict=True)
    source = checkpoint['models']['flow']
    keys = model.flow.state_dict()
    assert all(key in source and source[key].shape == value.shape for key, value in keys.items())
    model.flow.load_state_dict({key: source[key] for key in keys}, strict=True)
    report = {'parent_epoch': checkpoint['epoch'], 'reused_flow_keys': list(keys),
              'removed_flow_keys': sorted(set(source)-set(keys)), 'descriptor_heads_removed': True,
              'quantum_trainable_angles': model.history.theta.numel(), 'anchor_enabled': anchor_enabled}
    return model, report


def device_inputs(record, device):
    return {key: value.to(device) for key, value in record['encoder_inputs'].items()}
