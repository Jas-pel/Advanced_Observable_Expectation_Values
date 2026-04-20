import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList

from advanced_estimation.commutation.base_commutation import BaseCommutation


class NoCommutation(BaseCommutation):
    """
    Baseline commutation strategy with no grouping.
    
    Treats each Pauli operator as incompatible with all others (except itself).
    This results in N measurement circuits for N Pauli operators - one per operator.
    
    Used as a baseline for performance comparison. While inefficient for shot usage,
    it provides a simple reference point and generates minimal circuit depth per measurement.
    
    This is equivalent to saying no two Paulis commute, so they must be measured separately.
    """

    def commutation_table(self, paulis: PauliList) -> NDArray[np.bool]:
        """
        Return the bitwize commutation table within a PauliList. The element (i,j) is True if the ith Pauli string bitwise commutes with the jth Pauli string.

        Args:
            paulis (PauliList): A list of Pauli strings

        Returns:
            NDArray[np.bool]: The commutation table
        """

        return np.identity(paulis.size, dtype=bool)

    def diagonalize_paulis_with_circuit(
        self, paulis: PauliList
    ) -> tuple[PauliList, QuantumCircuit]:
        """
        Diagonalize many bitwize commuting Pauli strings and return diagonalizing unitary as a quantum circuit. The quantum circuit is made of single qubit operators.

        Args:
            paulis (PauliList): Bitwize commuting Pauli strings

        Returns:
            PauliList: Diagonal Pauli strings
            QuantumCircuit: The unitary transformation
        """

        assert paulis.size == 1

        diag_circuit = QuantumCircuit(paulis.num_qubits)

        for q, (z, x) in enumerate(zip(paulis[0].z, paulis[0].x)):
            if x:
                if z:
                    diag_circuit.sdg(q)
                diag_circuit.h(q)

        new_x = np.zeros(paulis.num_qubits, dtype=np.bool_)
        new_z = np.logical_or(paulis.z, paulis.x)

        diag_paulis = PauliList.from_symplectic(new_z, new_x)

        return diag_paulis, diag_circuit
