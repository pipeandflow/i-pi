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
        self.beads = nm.beads
        self.nbeads = nm.nbeads # TODO: make dependence on positions explicit
        self.natoms = nm.natoms
        self.omegan2 = nm.omegan2
        self.ensemble = nm.ensemble

        self._E_Ns, self._V = self.Evaluate_VB()

    def direct_link_probability(self, l):
        assert 0 <= l < self.nbeads - 1
        return 1 - (V_forward(l) * V_backward(l_next)) / self.partition_function(V_b)

    def get_vspring_and_fspring(self):
        """
        Calculates spring forces and potential for bosons.
        Evaluated using recursion relation from arXiv:1905.090.
        """
        P = self.nbeads

        F = self.evaluate_dVB_from_VB()

        return [self._V[-1], F]

    def evaluate_dVB_from_VB(self):
        P = self.nbeads
        N = len(self.bosons)

        F = np.zeros((P, 3 * self.natoms), float)
        for ind, l in enumerate(self.bosons):
            # force on intermediate beads is independent of the permutation
            for j in range(1, P - 1):
                F[j, 3 * l: 3 * (l + 1)] = -1.0 * self.Evaluate_dEkn_on_atom_full_ring(l, j)

        for ind, l in enumerate(self.bosons):
            for j in [0, P - 1]:
                # total_force = 0.0
                # for peer_boson in range(self.natoms):
                #     if peer_boson == l:
                #         continue
                #     if peer_boson == l + 1 and j == P - 1:
                #         total_force += self.direct_link_probability(l) \
                #                        * (-1.0) * self.Evaluate_dEkn_on_atom_full_ring(l, j)
                #     if peer_boson == l - 1 and j == 0:
                #         total_force += self.direct_link_probability(l - 1) \
                #                        * (-1.0) * self.Evaluate_dEkn_on_atom_full_ring(l, j)
                #         # TODO: handle case of l-1 < 0 (wrap around to P - 1)

                F[j, 3 * l: 3 * (l + 1)] = self.Evaluate_dVB(ind, j)

        return F

    def Evaluate_E_Ns(self, N):
        """
        Returns a list [E_N^{(1)},...,E_N^{(P)}] as defined in Equation 5 of arXiv:1905.09053.
        That is, the energy of k particles R_{N-k+1},...,R_N, connected sequentially into a ring polymer.
        j and l go from 0 to N-1 and P-1, respectively, for indexing. (In the paper they start from 1)
        Calculation using a recursive relation (unlike in the paper)
        """
        m = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson

        P = self.nbeads
        omegaP_sq = self.omegan2

        q = np.zeros((P, 3 * len(self.bosons)), float)
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
            assert bead_index + 1 < P
            return r_diff_squared(atom_index, bead_index,
                                  atom_index, bead_index + 1)

        res = np.zeros(N + 1, float)

        for k in range(0, N):
            added_atom_index = N - k - 1
            # TODO: vectorize
            added_atom_potential = sum(r_diff_squared_within_ring(added_atom_index, j) for j in range(P - 1))
            close_chain_to_added_atom = r_diff_squared(added_atom_index, 0, N-1, P-1)
            if k > 0:
                connect_added_atom_to_rest = r_diff_squared(added_atom_index, P - 1, added_atom_index + 1, 0)
                break_existing_ring = r_diff_squared(added_atom_index + 1, 0, N-1, P-1)
            else:
                connect_added_atom_to_rest = 0
                break_existing_ring = 0

            coefficient = 0.5 * m * omegaP_sq
            res[k + 1] = res[k] + coefficient * (- break_existing_ring
                                                 + added_atom_potential + connect_added_atom_to_rest
                                                 + close_chain_to_added_atom)

        return res


    def Evaluate_EkN(self, N, k):
        """
        Returns E_N^{(k)} as defined in Equation 5 of arXiv:1905.09053.
        That is, the energy of k particles R_{N-k+1},...,R_N, connected sequentially into a ring polymer.
        j and l go from 0 to N-1 and P-1, respectively, for indexing. (In the paper they start from 1)
        """
        # TODO: depracated by Evaluate_E_Ns
        m = dstrip(self.beads.m)[self.bosons[0]]  # Take mass of first boson

        P = self.nbeads
        omegaP_sq = self.omegan2

        q = np.zeros((P, 3 * len(self.bosons)), float)
        qall = dstrip(self.beads.q).copy()

        # Stores coordinates just for bosons in separate arrays with new indices 1,...,Nbosons
        for ind, boson in enumerate(self.bosons):
            q[:, 3 * ind : (3 * ind + 3)] = qall[:, 3 * boson : (3 * boson + 3)]

        sumE = 0.0
        # Here indices go from 0 to N-1 and P-1, respectively. In paper they start from 1.
        for l in range(N - k, N):
            for j in range(P):
                # q[j,:] stores 3*natoms xyz coordinates of all atoms.
                # Index of bead #(j+1) of atom #(l+1) is [l,3*l]
                r = q[j, 3 * l : 3 * (l + 1)]

                next_atom_ind, next_bead_ind = self.next_bead_k_ring(l, j, k, N)

                r_next = q[next_bead_ind, 3 * next_atom_ind : 3 * (next_atom_ind + 1)]
                diff = r_next - r

                sumE = sumE + np.dot(diff, diff)

        return 0.5 * m * omegaP_sq * sumE

    def next_bead_k_ring(self, atom_index, bead_index, k, N):
        """
        The next atom and bead indices in a ring polymer of k beads over particles R_{N-k+1},...,R_N.
        """
        P = self.nbeads
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
        P = self.nbeads
        omegaP_sq = self.omegan2

        q = np.zeros((P, 3 * len(self.bosons)), float)
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

        if j == P - 1:
            # If on the last bead, r_l_jp1 is the first bead of next atom
            next_bead_ind = 0
            next_atom_ind = 3 * (l + 1)

            if l == N - 1:
                # If on the last bead of last atom, r_l_jp1 is the first bead of N-k atom
                next_atom_ind = 3 * (N - k)

        if j == 0:
            # If on the first bead, r_l_j-1 is the last bead of previous atom
            prev_bead_ind = P - 1
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
        N = len(self.bosons)
        return self.Evaluate_dEkn_on_atom(l, j, N, N)

    def Evaluate_VB(self):
        """
        Evaluate VB_m, m = {0,...,N}. VB0 = 0.0 by definition.
        Evaluation of each VB_m is done using Equation 5 of arXiv:1905.0905.
        Returns all VB_m and all E_m^{(k)} which are required for the forces later.
        """

        N = len(self.bosons)
        betaP = 1.0 / (self.beads.nbeads * units.Constants.kb * self.ensemble.temp)

        V = np.zeros(N + 1, float)
        save_Ek_N = np.zeros(N * (N + 1) // 2, float)

        count = 0
        for m in range(1, N + 1):
            sig = 0.0

            E_Ns = self.Evaluate_E_Ns(m)

            # Reversed sum order to be able to use energy of longest ring polymer in Elong
            for k in range(m, 0, -1):
                E_k_N = E_Ns[k]

                # This is required for numerical stability. See SI of arXiv:1905.0905
                if k == m:
                    Elong = 0.5 * (E_k_N + V[m - 1])

                sig = sig + np.exp(-betaP * (E_k_N + V[m - k] - Elong))

                save_Ek_N[count] = E_k_N # TODO: no longer necessary
                count += 1

            V[m] = Elong - np.log(sig / m) / betaP

        return (save_Ek_N, V)


    def Evaluate_dVB(self, l, j):
        """
        Evaluates dVB_m, m = {0,...,N} for bead #(j+1) of atom #(l+1). dVB_0 = 0.0 by definition.
        Evalaution of dVB_m for endpoint beads is based on Equation 2 of SI to arXiv:1905.09053.
        Returns -dVB_N, the force acting on bead #(j+1) of atom #(l+1).
        """

        N = len(self.bosons)
        betaP = 1.0 / (self.beads.nbeads * units.Constants.kb * self.ensemble.temp)

        dV = np.zeros((N + 1, 3), float)

        # Reversed sum order to agree with Evaluate_VB() above
        for m in range(1, N + 1):
            sig = 0
            if l + 1 > m:  # l goes from 0 to N-1 so check for l+1
                pass  # dV[m,:] is initialized to zero vector already
            else:
                count = m * (m - 1) // 2
                for k in range(m, 0, -1):

                    if (
                        l + 1 >= m - k + 1 and l + 1 <= m
                    ):  # l goes from 0 to N-1 so check for l+1
                        dE_k_N = self.Evaluate_dEkn_on_atom(l, j, m, k)
                    else:
                        dE_k_N = np.zeros(3, float)
                    sig += (dE_k_N + dV[m - k, :]) * np.exp(
                        -betaP * (self._E_Ns[count] + self._V[m - k])
                    )
                    count += 1

                dV[m, :] = sig / (m * np.exp(-betaP * self._V[m]))

        return -1.0 * dV[N, :]
