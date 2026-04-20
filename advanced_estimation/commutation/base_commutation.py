from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList


class BaseCommutation(ABC):

    def find_maximal_commuting_cliques(
        self, paulis: PauliList
    ) -> list[NDArray[np.int_]]:
        """
        Return the maximal commuting cliques for a given list of Pauli Strings.
        """
        commutation_table = self.commutation_table(paulis)

        graph = nx.from_numpy_array(commutation_table)

        cliques = []
        for clique in nx.find_cliques(graph):
            cliques.append(np.array(clique))

        return cliques

    def find_min_commuting_cliques(self, paulis: PauliList) -> list[NDArray[np.int_]]:
        """
        Trouve une partition des opérateurs de Pauli en un nombre minimal de cliques commutantes.
        Utilise la coloration de graphe sur le graphe de non-commutation (anti-commutation).
        C'est l'approche standard pour minimiser le nombre de mesures (Groupement par Couleur).
        """
        commutation_table = self.commutation_table(paulis)

        anti_commutation_graph = nx.from_numpy_array(~commutation_table)

        coloring = nx.coloring.greedy_color(
            anti_commutation_graph, strategy="largest_first"
        )

        color_groups = {}
        for node_idx, color in coloring.items():
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(node_idx)

        cliques = [
            np.array(indices, dtype=np.int_) for indices in color_groups.values()
        ]

        return cliques

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
