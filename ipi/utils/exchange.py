"""Contains all methods to evalaute potential energy and forces for indistinguishable particles.
Used in /engine/normalmodes.py
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.
from ipi.utils import units
from ipi.utils.depend import *

import numpy as np


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

    # def full_cycle_probability(self):
    #     return np.exp(- self._betaP * self.Ek_N(self._N, self._N)) / (self._N * self.V_forward(self._N))

    def direct_link_probability(self, l):
        """
        The probability that l,l+1 (l=0...N-1) are joined in the same ring.
        Computed by 1 - the probability that a cycle "cuts" exactly between l,l+1.
        """
        assert 0 <= l < self._N - 1
        prob = 1 - (np.exp(- self._betaP * (self.V_forward(l) + self.V_backward(l + 1) -
                                            self.V_all())))
        return prob

    def separate_cycle_close_probability(self, l1, l2):
        assert l1 <= l2

        prob = 1 / (l2 + 1) * \
               np.exp(- self._betaP *
                       (self.V_forward(l1 - 1) + self.Ek_N(l2 + 1 - l1, l2 + 1) + self.V_backward(l2 + 1)
                        - self.V_all()))
        return prob

    def get_vspring_and_fspring(self):
        """
        Calculates spring forces and potential for bosons.
        Evaluated using recursion relation from arXiv:1905.090.
        """
        F = self.evaluate_dVB_from_VB()

        return [self._V[-1], F]

    def evaluate_dVB_from_VB(self):
        F = np.zeros((self._P, 3 * self.natoms), float)
        for ind, l in enumerate(self.bosons):
            # force on intermediate beads is independent of the permutation
            for j in range(1, self._P - 1):
                F[j, 3 * l: 3 * (l + 1)] = self._force_on_intermediate_bead(l, j)

        for ind, l in enumerate(self.bosons):
            for j in [0, self._P - 1]:
                total_force = 0

                if j == self._P - 1:
                    # TODO: vectorize sum?
                    for peer_boson in range(0, l + 1): # l + 1 to include cycle of l with itself
                        lower = peer_boson
                        higher = l
                        total_force += self.separate_cycle_close_probability(lower, higher) \
                                        * self._force_on_last_bead(l, peer_boson)

                    if l != self._N - 1:
                        total_force += self.direct_link_probability(l) \
                                        * self._force_on_last_bead(l, l + 1)

                if j == 0:
                    for peer_boson in range(l, self._N):
                        lower = l
                        higher = peer_boson
                        total_force += self.separate_cycle_close_probability(lower, higher) \
                                        * self._force_on_first_bead(l, peer_boson)

                    if l != 0:
                        total_force += self.direct_link_probability(l - 1) \
                                        * self._force_on_first_bead(l, l - 1)

                F[j, 3 * l: 3 * (l + 1)] = total_force

        return F
    
    def _spring_force_prefix(self):
        m = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson
        omegaP_sq = self.omegan2
        return (-1.0) * m * omegaP_sq
    def _force_on_intermediate_bead(self, l, j):
        assert 1 <= j <= self._P - 1
        return self._spring_force_prefix() * (-self._bead_diff_intra[j][l] + self._bead_diff_intra[j-1][l])

    def _force_on_first_bead(self, l, prev_l):
        return self._spring_force_prefix() * \
               (-self._bead_diff_intra[0][l] + self._bead_diff_inter_first_last_bead[l][prev_l])

    def _force_on_last_bead(self, l, next_l):
        return self._spring_force_prefix() * \
               (-self._bead_diff_inter_first_last_bead[next_l][l] + self._bead_diff_intra[-1][l])

    def Ek_N(self, k, m):
        end_of_m = m * (m + 1) // 2
        return self._Ek_N[end_of_m - k]

    def Evaluate_Ek_N(self):
        mass = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson

        omegaP_sq = self.omegan2

        save_Ek_N = np.zeros(self._N * (self._N + 1) // 2, float)

        intra_spring_energies = np.sum(self._bead_diff_intra ** 2, axis=(0, -1))
        spring_energy_first_last_bead_array = np.sum(self._bead_diff_inter_first_last_bead ** 2, axis=-1)

        count = 0
        for m in range(1, self._N + 1):
            Emks = np.zeros(m + 1, float)

            for k in range(0, m):
                added_atom_index = m - k - 1
                added_atom_potential = intra_spring_energies[added_atom_index]
                close_chain_to_added_atom = spring_energy_first_last_bead_array[added_atom_index, m - 1]
                if k > 0:
                    connect_added_atom_to_rest = spring_energy_first_last_bead_array[added_atom_index + 1,
                                                                                     added_atom_index]
                    break_existing_ring = spring_energy_first_last_bead_array[added_atom_index + 1, m - 1]
                else:
                    connect_added_atom_to_rest = 0
                    break_existing_ring = 0

                coefficient = 0.5 * mass * omegaP_sq
                Emks[k + 1] = Emks[k] + coefficient * (- break_existing_ring
                                                       + added_atom_potential + connect_added_atom_to_rest
                                                       + close_chain_to_added_atom)

            # Reversed order similar to Evaluate_VB and Evaluate_dVB
            for k in range(m, 0, -1):
                save_Ek_N[count] = Emks[k]
                count += 1

        return save_Ek_N

    def Evaluate_VB(self):
        """
        Evaluate VB_m, m = {0,...,N}. VB0 = 0.0 by definition.
        Evaluation of each VB_m is done using Equation 5 of arXiv:1905.0905.
        Returns all VB_m and all E_m^{(k)} which are required for the forces later.
        """
        V = np.zeros(self._N + 1, float)

        for m in range(1, self._N + 1):
            sig = 0.0
            # This is required for numerical stability. See SI of arXiv:1905.0905
            Elong = min(self.Ek_N(m, 1) + V[m-1], self.Ek_N(m, m) + V[0])

            # TODO: Reversed sum order for reasons that are not obsolete (had to do with Elong)
            for k in range(m, 0, -1):
                E_k_N = self.Ek_N(k, m)
                sig = sig + np.exp(- self._betaP * (E_k_N + V[m - k] - Elong))

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
