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

        self._factorial_cache = FactorialMemoization()

        self._N = len(self.bosons)
        self._P = nm.nbeads
        self._betaP = 1.0 / (self._P * units.Constants.kb * self.ensemble.temp)

        self._Ek_N = self.Evaluate_Ek_N()
        self._V = self.Evaluate_VB()

        self._V_backward = self.Evaluate_V_backward_from_V_forward()

    def _factorial(self, n):
        return self._factorial_cache.factorial(n)

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
                F[j, 3 * l: 3 * (l + 1)] = -1.0 * self.Evaluate_dEkn_on_atom_full_ring(l, j)

        for ind, l in enumerate(self.bosons):
            for j in [0, self._P - 1]:
                total_force = 0

                if j == self._P - 1:
                    for peer_boson in range(0, l + 1): # l + 1 to include cycle of l with itself
                        lower = peer_boson
                        higher = l
                        total_force += self.separate_cycle_close_probability(lower, higher) \
                                       * (-1.0) * self.Evaluate_dEkn_on_atom(l, j, N=higher+1, k=higher-lower+1)

                    if l != self._N - 1:
                        total_force += self.direct_link_probability(l) \
                                       * (-1.0) * self.Evaluate_dEkn_on_atom_full_ring(l, j)

                if j == 0:
                    for peer_boson in range(l, self._N):
                        lower = l
                        higher = peer_boson
                        total_force += self.separate_cycle_close_probability(lower, higher) \
                                       * (-1.0) * self.Evaluate_dEkn_on_atom(l, j, N=higher+1, k=higher-lower+1)

                    if l != 0:
                        total_force += self.direct_link_probability(l - 1) \
                                       * (-1.0) * self.Evaluate_dEkn_on_atom_full_ring(l, j)

                F[j, 3 * l: 3 * (l + 1)] = total_force

        return F

    def next_bead_k_ring(self, atom_index, bead_index, k, N):
        """
        The next atom and bead indices in a ring polymer of k beads over particles R_{N-k+1},...,R_N.
        """
        l = atom_index
        j = bead_index

        # Taking care of boundary conditions.
        # Usually r_l_jp1 is the next bead of same atom.
        next_bead_ind = j + 1
        next_atom_ind = l
        if j == P - 1:
            # If on the last bead, r_l_jp1 is the first bead of next atom
            next_bead_ind = 0
            next_atom_ind = l + 1

            if l == N - 1:
                # If on the last bead of last atom, r_l_jp1 is the first bead of N-k atom
                next_atom_ind = N - k
        return next_atom_ind, next_bead_ind

    def Evaluate_dEkn_on_atom(self, l, j, N, k):
        """
        Returns dE_N^{(k)} as defined in Equation 3 of SI to arXiv:1905.09053.
        That is, the force on bead j of atom l due to k particles
        R_{N-k+1},...,R_N, connected sequentially into a ring polymer.
        j and l go from 0 to N-1 and P-1, respectively, for indexing. (In the paper they start from 1)
        """

        m = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson
        omegaP_sq = self.omegan2

        q = np.zeros((self._P, 3 * self._N), float)
        qall = dstrip(self.beads.q).copy()

        # Stores coordinates just for bosons in separate arrays with new indices 1,...,Nbosons
        for ind, boson in enumerate(self.bosons):
            q[:, 3 * ind : (3 * ind + 3)] = qall[:, 3 * boson : (3 * boson + 3)]

        # q[j,:] stores 3*natoms xyz coordinates of all atoms.
        # Index of bead #(j+1) of atom #(l+1) is [l,3*l]
        r = q[j, 3 * l : 3 * (l + 1)]
        next_bead_ind = j + 1
        next_atom_ind = 3 * l
        prev_bead_ind = j - 1
        prev_atom_ind = 3 * l

        if j == self._P - 1:
            # If on the last bead, r_l_jp1 is the first bead of next atom
            next_bead_ind = 0
            next_atom_ind = 3 * (l + 1)

            if l == N - 1:
                # If on the last bead of last atom, r_l_jp1 is the first bead of N-k atom
                next_atom_ind = 3 * (N - k)

        if j == 0:
            # If on the first bead, r_l_j-1 is the last bead of previous atom
            prev_bead_ind = self._P - 1
            prev_atom_ind = 3 * (l - 1)

            if l == N - k:
                # If on the first bead of N-k atom, r_l_j-1 is the last bead of last atom
                prev_atom_ind = 3 * (N - 1)

        r_next = q[next_bead_ind, next_atom_ind : (next_atom_ind + 3)]
        r_prev = q[prev_bead_ind, prev_atom_ind : (prev_atom_ind + 3)]
        diff = 2 * r - r_next - r_prev

        return m * omegaP_sq * diff

    def Evaluate_dEkn_on_atom_full_ring(self, l, j):
        """
        TODO:
        """
        return self.Evaluate_dEkn_on_atom(l, j, self._N, self._N)

    def Ek_N(self, k, m):
        end_of_m = m * (m + 1) // 2
        return self._Ek_N[end_of_m - k]

    def Evaluate_Ek_N(self):
        mass = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson

        omegaP_sq = self.omegan2

        q = np.zeros((self._P, 3 * self._N), float)
        qall = dstrip(self.beads.q).copy()

        # Stores coordinates just for bosons in separate arrays with new indices 1,...,Nbosons
        for ind, boson in enumerate(self.bosons):
            q[:, 3 * ind: (3 * ind + 3)] = qall[:, 3 * boson: (3 * boson + 3)]

        # q[j,:] stores 3*natoms xyz coordinates of all atoms.
        # Index of bead #(j+1) of atom #(l+1) is [l,3*l]
        # TODO: extract somewhere
        def r_of(atom_index, bead_index):
            return q[bead_index, 3 * atom_index: 3 * (atom_index + 1)]

        def r_diff_squared(atom1, bead1, atom2, bead2):
            diff = r_of(atom2, bead2) - r_of(atom1, bead1)
            return np.dot(diff, diff)

        def r_diff_squared_within_ring(atom_index, bead_index):
            assert bead_index + 1 < self._P
            return r_diff_squared(atom_index, bead_index,
                                  atom_index, bead_index + 1)

        save_Ek_N = np.zeros(self._N * (self._N + 1) // 2, float)

        intraparticle_spring_energies = np.zeros(self._N)
        for l in range(0, self._N):
            intraparticle_spring_energies[l] = sum(r_diff_squared_within_ring(l, j)
                                                   for j in range(self._P - 1))

        count = 0
        for m in range(1, self._N + 1):
            Emks = np.zeros(m + 1, float)

            for k in range(0, m):
                added_atom_index = m - k - 1
                # TODO: vectorize
                added_atom_potential = intraparticle_spring_energies[added_atom_index]
                close_chain_to_added_atom = r_diff_squared(added_atom_index, 0, m - 1, self._P - 1)
                if k > 0:
                    connect_added_atom_to_rest = r_diff_squared(added_atom_index, self._P - 1, added_atom_index + 1, 0)
                    break_existing_ring = r_diff_squared(added_atom_index + 1, 0, m - 1, self._P - 1)
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
    

class FactorialMemoization(object):
    def __init__(self):
        self._cache = {}

    def factorial(self, n):
        if n in self._cache:
            return self._cache[n]
        return np.math.factorial(n)