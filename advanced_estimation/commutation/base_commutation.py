from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList


class BaseCommutation(ABC):

    def find_commuting_cliques(self, paulis: PauliList) -> list[NDArray[np.int_]]:

        commutation_table = self.commutation_table(paulis)

        graph = nx.from_numpy_array(commutation_table)

        cliques = []
        for clique in nx.find_cliques(graph):
            cliques.append(np.array(clique))

        return cliques

    def advanced_find_commuting_cliques(
        self, paulis: PauliList
    ) -> list[NDArray[np.int_]]:
        """
        Trouve une couverture minimale de cliques pour un graphe de commutation.
        Utilise un algorithme greedy pour résoudre le problème de Minimum Clique Cover.
        NOTE : Je vais essayer de coder une meilleure solution, c'est surtout un test.
        """
        commutation_table = self.commutation_table(paulis)
        num_paulis = len(commutation_table)

        graph = nx.from_numpy_array(commutation_table)

        all_maximal_cliques = list(nx.find_cliques(graph))
        all_maximal_cliques = [
            np.array(clique, dtype=np.int_) for clique in all_maximal_cliques
        ]

        uncovered = set(range(num_paulis))

        selected_cliques = []
        while uncovered:
            best_clique = None
            best_coverage = 0

            for clique in all_maximal_cliques:
                coverage = len(set(clique) & uncovered)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = clique

            if best_clique is None or best_coverage == 0:
                for node in uncovered:
                    selected_cliques.append(np.array([node], dtype=np.int_))
                break

            selected_cliques.append(best_clique)

            uncovered -= set(best_clique)

        return selected_cliques

    @abstractmethod
    def commutation_table(self, paulis: PauliList) -> NDArray[np.bool]:
        """
        Return the commutation table for a given list of Pauli Strings. The element (i,j) is True if the ith Pauli string commutes with the jth Pauli string in the sense given by the class.

        Args:
            paulis (PauliList): A list of Pauli strings

        Returns:
            NDArray[np.bool]: The commutation table
        """

        raise NotImplementedError("Commutation must implement a commutation table")

    @abstractmethod
    def diagonalize_paulis_with_circuit(
        self, paulis: PauliList
    ) -> tuple[PauliList, QuantumCircuit]:
        """
        Diagonalize many commuting (in the class sense) Pauli strings and return diagonalizing unitary as a quantum circuit.

        Args:
            paulis (PauliList): A list of Pauli strings

        Returns:
            PauliList: A list of diagonal Pauli strings
            QuantumCircuit: The unitary transformation as a QuantumCircuit
        """

        raise NotImplementedError(
            "Commutation must implement a diagonalisation with circuit procedure"
        )
