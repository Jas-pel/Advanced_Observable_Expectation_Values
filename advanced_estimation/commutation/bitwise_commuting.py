import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList

from advanced_estimation.commutation.base_commutation import BaseCommutation


class BitwiseCommutation(BaseCommutation):
    """
    Conservative commutation strategy using qubit-by-qubit matching.
    
    Two Pauli operators commute (bitwise) if they commute on every qubit independently.
    This is a stricter condition than general commutation, resulting in smaller measurement
    groups but with simpler circuits using only single-qubit Clifford gates.
    
    Suitable for scenarios where circuit depth is critical or when conservative grouping
    is preferred over shot efficiency.
    
    Examples:
        - XX and YY commute bitwise (both act as {X,Y} on qubits 0 and 1)
        - XZ and YY do NOT commute bitwise (X and Y don't commute)
    """

    def commutation_table(self, paulis: PauliList) -> NDArray[np.bool]:
        """
        Return the bitwize commutation table within a PauliList. The element (i,j) is True if the ith Pauli string bitwise commutes with the jth Pauli string.

        Args:
            paulis (PauliList): A list of Pauli strings

        Returns:
            NDArray[np.bool]: The commutation table
        """

        ovlp_1 = paulis.z[:, None] * paulis.x[None, :]
        ovlp_2 = paulis.x[:, None] * paulis.z[None, :]

        return np.all(~np.logical_xor(ovlp_1, ovlp_2), axis=-1)

    def diagonalize_paulis_with_circuit(self, paulis: PauliList) -> tuple[PauliList, QuantumCircuit]:
        """
        Diagonalize many bitwize commuting Pauli strings and return diagonalizing unitary as a quantum circuit. The quantum circuit is made of single qubit operators.

        Args:
            paulis (PauliList): Bitwize commuting Pauli strings

        Returns:
            PauliList: Diagonal Pauli strings
            QuantumCircuit: The unitary transformation
        """

        assert np.all(self.commutation_table(paulis))

        num_qubits = paulis.num_qubits

        all_positions = np.arange(num_qubits)
        x_positions = all_positions[
            np.any(
                np.logical_and(paulis.x, ~paulis.z),
                axis=0,
            )
        ]
        y_positions = all_positions[
            np.any(
                np.logical_and(paulis.x, paulis.z),
                axis=0,
            )
        ]

        diag_circuit = QuantumCircuit(num_qubits)

        if x_positions.size > 0:
            diag_circuit.h(x_positions)
        if y_positions.size > 0:
            diag_circuit.sdg(y_positions)
            diag_circuit.h(y_positions)

        diag_paulis = PauliList.from_symplectic(np.logical_or(paulis.z, paulis.x), np.zeros_like(paulis.z))

        return diag_paulis, diag_circuit
