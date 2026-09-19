"""Raw-prediction collision and residue-contact constraints for terminal geometry."""
import numpy as np
from scipy.spatial.distance import cdist


class ObservableConstraints:
    def __init__(self, original, inputs, geometry, distance_padding):
        self.atoms = len(original)
        self.heavy = np.asarray(geometry['heavy'])
        self.environment = np.asarray(geometry['environment'])
        self.pairs = np.asarray(geometry['self_pairs'])
        pair_distance = np.linalg.norm(original[self.pairs[:, 0]]-original[self.pairs[:, 1]], axis=-1)
        # Scoring calls a collision when normalized compression exceeds 1e-7.
        self.self_floor = np.minimum(pair_distance, np.asarray(geometry['self_limits'])*(1-1e-7))+distance_padding
        cross = cdist(original[self.heavy], self.environment)
        self.cross_floor = np.minimum(cross, np.asarray(geometry['cross_limits'])*(1-1e-7))+distance_padding
        pt = inputs['protein_topology']
        protein_heavy = np.asarray(pt['atomic_numbers']) > 1
        self.protein_count = int(protein_heavy.sum())
        residues, slots = np.unique(np.asarray(pt['residue_indices'])[protein_heavy], return_inverse=True)
        self.residue_groups = [np.flatnonzero(slots == i) for i in range(len(residues))]
        distances = np.array([cross[:, group].min() for group in self.residue_groups])
        self.contact_sign = np.where(distances < 4.5, -1., 1.)
        self.padding = distance_padding
        self.pair_groups = [np.flatnonzero(self.pairs[:, 0] == i) for i in np.unique(self.pairs[:, 0])]

    def values_jacobian(self, flat):
        x = flat.reshape(self.atoms, 3)
        cross = cdist(x[self.heavy], self.environment)
        residual = cross-self.cross_floor
        partner = residual.argmin(1)
        slots = np.arange(len(self.heavy))
        cross_value = residual[slots, partner]
        cross_jac = np.zeros((len(slots), self.atoms, 3))
        cross_jac[slots, self.heavy] = (x[self.heavy]-self.environment[partner])/cross[slots, partner, None]

        delta = x[self.pairs[:, 0]]-x[self.pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=-1)
        pair_residual = distance-self.self_floor
        active = np.array([group[np.argmin(pair_residual[group])] for group in self.pair_groups], dtype=int)
        self_value = pair_residual[active]
        self_jac = np.zeros((len(active), self.atoms, 3))
        direction = delta[active]/distance[active, None]
        self_jac[np.arange(len(active)), self.pairs[active, 0]] = direction
        self_jac[np.arange(len(active)), self.pairs[active, 1]] = -direction

        # The minimum pair changes with coordinates: a contact is the union of all
        # ligand/protein pair contacts in that residue, with a piecewise gradient.
        protein_cross = cross[:, :self.protein_count]
        closest_ligand = protein_cross.argmin(0)
        protein_distance = protein_cross.min(0)
        selected = np.array([group[np.argmin(protein_distance[group])] for group in self.residue_groups])
        ligand = self.heavy[closest_ligand[selected]]
        contact_value = self.contact_sign*(protein_distance[selected]-4.5)-self.padding
        contact_jac = np.zeros((len(selected), self.atoms, 3))
        contact_jac[np.arange(len(selected)), ligand] = self.contact_sign[:, None]*(x[ligand]-self.environment[selected])/protein_distance[selected, None]
        return np.concatenate((self_value, cross_value, contact_value)), np.concatenate((self_jac, cross_jac, contact_jac)).reshape(-1, self.atoms*3)

    def fun(self, flat):
        return self.values_jacobian(flat)[0]

    def jac(self, flat):
        return self.values_jacobian(flat)[1]

    def pair_gaps(self, x):
        distance = np.linalg.norm(x[self.pairs[:, 0]]-x[self.pairs[:, 1]], axis=-1)
        return distance-self.self_floor, cdist(x[self.heavy], self.environment)-self.cross_floor

    def explicit_values_jacobian(self, x, self_indices, cross_indices, contact_residues):
        """Conjunctive pair rows plus existential contact rows for the given residues."""
        si, cp = self_indices, cross_indices
        pairs = self.pairs[si]
        delta = x[pairs[:, 0]]-x[pairs[:, 1]]
        distance = np.linalg.norm(delta, axis=-1)
        sj = np.zeros((len(si), self.atoms, 3))
        sj[np.arange(len(si)), pairs[:, 0]] = delta/distance[:, None]
        sj[np.arange(len(si)), pairs[:, 1]] = -delta/distance[:, None]
        atoms, partners = self.heavy[cp[:, 0]], cp[:, 1]
        delta = x[atoms]-self.environment[partners]
        cross_distance = np.linalg.norm(delta, axis=-1)
        cj = np.zeros((len(cp), self.atoms, 3))
        cj[np.arange(len(cp)), atoms] = delta/cross_distance[:, None]
        values = [distance-self.self_floor[si], cross_distance-self.cross_floor[cp[:, 0], partners]]
        jac = [sj, cj]
        cross = cdist(x[self.heavy], self.environment)
        for residue in contact_residues:
            group = self.residue_groups[residue]
            i, k = np.unravel_index(cross[:, group].argmin(), (len(self.heavy), len(group)))
            partner, atom = group[k], self.heavy[i]
            row = np.zeros((1, self.atoms, 3))
            row[0, atom] = -(x[atom]-self.environment[partner])/cross[i, partner]
            values.append(np.array([4.5-self.padding-cross[i, partner]])); jac.append(row)
        return np.concatenate(values), np.concatenate(jac).reshape(-1, self.atoms*3)
