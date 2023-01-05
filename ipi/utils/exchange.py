"""Contains all methods to evalaute potential energy and forces for indistinguishable particles.
Used in /engine/normalmodes.py
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.
from ipi.utils import units
from ipi.utils.depend import *

import numpy as np


def kth_diag_indices(a, k):
    """
    Indices to access matrix k-diagonals in numpy.
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class ExchangePotential(dobject):
    def __init__(self, nm):
        assert len(nm.bosons) != 0
        self.bosons = nm.bosons
        self.beads = nm.beads  # TODO: make dependence on positions explicit
        self.natoms = nm.natoms
        self.omegan2 = nm.omegan2
        self.ensemble = nm.ensemble

        self._N = len(self.bosons)
        self._P = nm.nbeads
        self._betaP = 1.0 / (self._P * units.Constants.kb * self.ensemble.temp)

        self._q = self._init_bead_position_array(dstrip(self.beads.q))

        # self._bead_diff_intra[j] = [r^{j+1}_0 - r^{j}_0, ..., r^{j+1}_{N-1} - r^{j}_{N-1}]
        self._bead_diff_intra = np.diff(self._q, axis=0)
        # self._bead_dist_inter_first_last_bead[l][m] = r^0_{l} - r^{P-1}_{m}
        self._bead_diff_inter_first_last_bead = self._q[0, :, np.newaxis, :] - self._q[self._P - 1, np.newaxis, :, :]

        self._Ek_N = self.Evaluate_Ek_N()
        self._V = self.Evaluate_VB()

        self._V_backward = self.Evaluate_V_backward_from_V_forward()

    def _init_bead_position_array(self, qall):
        qall = dstrip(self.beads.q)

        q = np.zeros((self._P, self._N, 3), float)
        # Stores coordinates just for bosons in separate arrays with new indices 1,...,Nbosons
        # q[j,:] stores 3*natoms xyz coordinates of all atoms.
        # Index of bead #(j+1) of atom #(l+1) is [l,3*l]
        for ind, boson in enumerate(self.bosons):
            q[:, ind, :] = qall[:, 3 * boson: (3 * boson + 3)]

        return q


    def V_forward(self, l):
        """
        V_forward(l) is V[l+1]: log the weight of the representative permutations on particles 0,...,l.
        """
        return self._V[l + 1]

    def V_backward(self, s):
        """
        V_backward[s] is the same recursive calculation of V with a different initial condition: V(s) = 0.
        This is log the weight of the representative permutations on particles N-s-1,...,N-1.
        """
        return self._V_backward[s]

    def V_all(self):
        """
        V_(1)^(N), which is V_forward(self._N - 1) == self.V_backward(0)
        """
        return self._V[self._N]

    def get_vspring_and_fspring(self):
        """
        Calculates spring forces and potential for bosons.
        Evaluated using recursion relation from arXiv:1905.090.
        """
        F = self.evaluate_dVB_from_VB()

        return [self._V[-1], F]

    def evaluate_dVB_from_VB(self):
        F = np.zeros((self._P, self.natoms, 3), float)

        # force on intermediate beads

        # for 1 <= j < self._P - 1, 0 <= l < self._N:
        # F[j, l, :] = self._spring_force_prefix() * (-self._bead_diff_intra[j][l] + self._bead_diff_intra[j - 1][l])
        F[1:-1, :, :] = self._spring_force_prefix() * (-self._bead_diff_intra[1:, :] +
                                                       np.roll(self._bead_diff_intra, axis=0, shift=1)[1:, :])

        # force on endpoint beads
        #
        connection_probs = np.zeros((self._N, self._N), float)
        # close cycle probabilities:
        # for u in range(0, self._N):
        #     for l in range(u, self._N):
        #         connection_probs[l][u] = 1 / (l + 1) * \
        #                np.exp(- self._betaP *
        #                        (self.V_forward(u - 1) + self.Ek_N(l + 1 - u, l + 1) + self.V_backward(l + 1)
        #                         - self.V_all()))
        tril_indices = np.tril_indices(self._N, k=0)
        connection_probs[tril_indices] = (  # np.asarray([1 / (l + 1) for l in range(self._N)])[:, np.newaxis] *
                            np.reciprocal(np.arange(1.0, self._N + 1))[:, np.newaxis] *
                            np.exp(- self._betaP * (
                                # np.asarray([self.V_forward(u - 1) for u in range(self._N)])[np.newaxis, :]
                                self._V[np.newaxis, :-1]
                                # + np.asarray([(self.Ek_N(l + 1 - u, l + 1) if l >= u else 0) for l in range(self._N) for u in range(self._N)]).reshape((self._N, self._N))
                                + self._Ek_N.T
                                # + np.asarray([self.V_backward(l + 1) for l in range(self._N)])[:, np.newaxis]
                                + self._V_backward[1:, np.newaxis]
                                - self.V_all()
                            )))[tril_indices]

        # direct link probabilities:
        # for 0 <= l < self._N - 1:
        # connection_probs[l][l+1] = 1 - (np.exp(- self._betaP * (self.V_forward(l) + self.V_backward(l + 1) -
        #                                     self.V_all())))
        superdiagonal_indices = kth_diag_indices(connection_probs, k=1)
        connection_probs[superdiagonal_indices] = 1 - (np.exp(- self._betaP *
                                                        (self._V[1:-1] + self._V_backward[1:-1] - self.V_all())))

        # on the last bead:
        #
        # for 0 <= l < self._N:
        #   for 0 <= next_l <= max(l + 1, self._N - 1):
        #       force_from_neighbor[l][next_l] = self._spring_force_prefix() * \
        #                         (-self._bead_diff_inter_first_last_bead[next_l][l] + self._bead_diff_intra[-1][l])
        # F[-1, l, :] = sum_{next_l}{connection_probs[l][next_l] * force_from_neighbors[next_l]}
        #
        # First vectorization:
        # for 0 <= l < self._N:
        #   force_from_neighbors[l] = self._spring_force_prefix() * \
        #                         (-self._bead_diff_inter_first_last_bead[:, l] + self._bead_diff_intra[-1][l])
        # F[-1, l, :] = np.dot(connection_probs[l], force_from_neighbors)
        force_from_neighbors = self._spring_force_prefix() * \
                               (-np.transpose(self._bead_diff_inter_first_last_bead,
                                              axes=(1,0,2))
                                + self._bead_diff_intra[-1, :, np.newaxis])
        # F[-1, l, k] = sum_{j}{force_from_neighbors[l][j][k] * connection_probs[l,j]}
        F[-1, :, :] = np.einsum('ljk,lj->lk', force_from_neighbors, connection_probs)

        # on the first bead:
        #
        # for 0 <= l < self._N:
        #   for l - 1 <= prev_l < self._N:
        #       force_from_neighbors[l][prev_l] = self._spring_force_prefix() * \
        #                    (-self._bead_diff_intra[0][l] + self._bead_diff_inter_first_last_bead[l][prev_l])
        #    F[0, l, :] = sum_{prev_l} connection_probs[prev_l, l] * force_from_neighbors[l])
        #
        # First vectorization:
        # for 0 <= l < self._N:
        #   force_from_neighbors[l] = self._spring_force_prefix() * \
        #                          (-self._bead_diff_intra[0][l] + self._bead_diff_inter_first_last_bead[l, :])
        #    F[0, l, :] = np.dot(connection_probs[:, l], force_from_neighbors[l])
        force_from_neighbors = self._spring_force_prefix() * \
                                    (self._bead_diff_inter_first_last_bead - self._bead_diff_intra[0, :, np.newaxis])
        # F[0, l, k] = sum_{j}{force_from_neighbors[l][j][k] * connection_probs[j,l]}
        F[0, :, :] = np.einsum('ljk,jl->lk', force_from_neighbors, connection_probs)

        return F.reshape((self._P, 3 * self.natoms))
    
    def _spring_force_prefix(self):
        m = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson
        omegaP_sq = self.omegan2
        return (-1.0) * m * omegaP_sq

    def Ek_N(self, k, m):
        if k > m:
            return 0
        upper = m - 1
        lower = upper - k + 1
        assert 0 <= upper < self._N, upper
        assert 0 <= lower < self._N, lower
        return self._Ek_N[lower][upper]

    def Evaluate_Ek_N(self):
        mass = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson

        omegaP_sq = self.omegan2
        coefficient = 0.5 * mass * omegaP_sq

        Emks = np.zeros((self._N, self._N), float)

        intra_spring_energies = np.sum(self._bead_diff_intra ** 2, axis=(0, -1))
        spring_energy_first_last_bead_array = np.sum(self._bead_diff_inter_first_last_bead ** 2, axis=-1)

        for m in range(self._N):
            Emks[m][m] = coefficient * (intra_spring_energies[m] + spring_energy_first_last_bead_array[m, m])

            for k in range(1, m + 1):
                added_atom_index = m - k
                added_atom_potential = intra_spring_energies[added_atom_index]
                close_chain_to_added_atom = spring_energy_first_last_bead_array[added_atom_index, m]
                connect_added_atom_to_rest = spring_energy_first_last_bead_array[added_atom_index + 1,
                                                                                 added_atom_index]
                break_existing_ring = spring_energy_first_last_bead_array[added_atom_index + 1, m]

                Emks[m - k][m] = Emks[m - k + 1][m] + coefficient * (- break_existing_ring
                                                       + added_atom_potential + connect_added_atom_to_rest
                                                       + close_chain_to_added_atom)

        return Emks

    def Evaluate_VB(self):
        """
        Evaluate VB_m, m = {0,...,N}. VB0 = 0.0 by definition.
        Evaluation of each VB_m is done using Equation 5 of arXiv:1905.0905.
        Returns all VB_m and all E_m^{(k)} which are required for the forces later.
        """
        V = np.zeros(self._N + 1, float)

        for m in range(1, self._N + 1):
            # This is required for numerical stability. See SI of arXiv:1905.0905
            Elong = min(self.Ek_N(m, 1) + V[m-1], self.Ek_N(m, m) + V[0])

            # sig = 0.0
            # for u in range(m):
            #   sig += np.exp(- self._betaP *
            #                (V[u] + self._Ek_N[u, m - 1] - Elong) # V until u-1, then cycle from u to m
            #                 )
            sig = np.sum(np.exp(- self._betaP *
                                (V[:m] + self._Ek_N[:m, m - 1] - Elong)
                                ))
            assert sig != 0.0
            V[m] = Elong - np.log(sig / m) / self._betaP

        return V

    def Evaluate_V_backward_from_V_forward(self):
        """
        Evaluate VB_m, m = {0,...,N}. VB0 = 0.0 by definition.
        Evaluation of each VB_m is done using Equation 5 of arXiv:1905.0905.
        Returns all VB_m and all E_m^{(k)} which are required for the forces later.
        """
        RV = np.zeros(self._N + 1, float)

        for l in range(self._N - 1, 0, -1):
            sig = 0.0
            # For numerical stability. See SI of arXiv:1905.0905
            Elong = min(self.Ek_N(1, l + 1) + RV[l + 1], self.Ek_N(self._N - l, self._N) + RV[self._N])

            # TODO: sum order for reasons that are now obsolete (had to with Elong)
            for p in range(l, self._N):
                # comparing sum of self.separate_cycle_close_probability(l, _) with self.direct_link_probability(l - 1)
                k = p - l + 1
                E_k_p = self.Ek_N(k, p + 1)

                prefactor = 1 / (p + 1)
                sig += prefactor * np.exp(- self._betaP * (E_k_p + RV[p + 1]
                                                           # cancel: + self.V_forward(l - 1) - self.V_forward(l - 1)
                                                           - Elong))

            assert sig != 0.0
            RV[l] = Elong - np.log(sig) / self._betaP

        # V^{(N)}_{(1)}
        RV[0] = self._V[-1]

        return RV
