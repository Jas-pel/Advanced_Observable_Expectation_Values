import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList

from advanced_estimation.commutation import clifford
from advanced_estimation.commutation.base_commutation import BaseCommutation


class GeneralCommutation(BaseCommutation):
    """
    Commutation strategy based on general quantum commutation relations.
    
    Implements the algorithm from Gokhale et al. (arXiv:1907.13623) for optimal
    Pauli grouping. Groups operators by computing commutation relationships using
    the anticommutation count, then synthesizes efficient diagonalization circuits.
    
    This strategy produces the minimal number of measurement groups, making it
    the most efficient for shot budget allocation.
    
    Attributes:
        force_single_qubit_generators (bool): Whether to include single-qubit generators
            in addition to multi-qubit generators for improved circuit efficiency.
    """

    def __init__(self, force_single_qubit_generators=True):
        """
        Initialize the GeneralCommutation strategy.
        
        Args:
            force_single_qubit_generators (bool, optional): If True, add single-qubit
                generators to the set for potentially shorter circuits. Default: True.
        """
        self.force_single_qubit_generators = force_single_qubit_generators

    def commutation_table(self, paulis: PauliList) -> NDArray[np.bool]:
        """
        Return the general commutation table within a PauliList.
        The element (i,j) is True if the ith Pauli string commutes with the jth Pauli string.

        Args:
            paulis (PauliList): A list of Pauli strings

        Returns:
            NDArray[np.bool]: The commutation table
        """
        x = paulis.x.astype(int)
        z = paulis.z.astype(int)

        commutation_table = np.einsum("iq, jq -> ij", x, z) + np.einsum(
            "iq, jq -> ij", z, x
        )
        return np.mod(commutation_table, 2) == 0

    def diagonalize_paulis_with_circuit(
        self, paulis: PauliList
    ) -> tuple[PauliList, QuantumCircuit]:
        """
        Diagonalize many general commuting Pauli strings and return diagonalizing unitary as a quantum circuit.
        Based on Gokhale et al. (2019, arXiv:1907.13623)

        Args:
                paulis (PauliList): A list of Pauli strings

            Returns:
                PauliList: A list of diagonal Pauli strings
                QuantumCircuit: The unitary transformation as a QuantumCircuit
        """

        assert np.all(self.commutation_table(paulis))

        diag_paulis = paulis.copy()
        circuit = QuantumCircuit(paulis.num_qubits)

        generators_paulis = self._paulis_to_generators(paulis)

        if self.force_single_qubit_generators:
            single_qubit_generators_paulis = self._single_qubit_cummuting_generators(
                generators_paulis
            )
            generators_paulis = generators_paulis + single_qubit_generators_paulis

        qubit_order = np.arange(paulis.num_qubits)

        x_table = generators_paulis.x
        z_table = generators_paulis.z

        qubit_order = np.arange(paulis.num_qubits)

        # change generator and qubit orders to have as many 1 on the diagonal of x from the upper left corner
        x_table, row_op, col_order, start_index = self._pack_diagonal(x_table, 0)
        z_table = np.mod((row_op.astype(np.uint) @ z_table.astype(np.uint)), 2).astype(
            bool
        )[:, col_order]
        qubit_order = qubit_order[col_order]

        # print()
        # print(np.concatenate((z_table, x_table), axis=1).astype(int))

        end_block_x = start_index

        # change generator and qubit orders to have as many 1 on the diagonal of z from the last diagonal x
        z_table, row_op, col_order, start_index = self._pack_diagonal(
            z_table, start_index
        )
        x_table = np.mod((row_op.astype(np.uint) @ x_table.astype(np.uint)), 2).astype(
            bool
        )[:, col_order]
        qubit_order = qubit_order[col_order]

        end_block_z = start_index

        # apply hadamards to convert the diagonal z into x
        tmp = x_table[:, end_block_x:end_block_z].copy()
        x_table[:, end_block_x:end_block_z] = z_table[:, end_block_x:end_block_z].copy()
        z_table[:, end_block_x:end_block_z] = tmp

        _is = qubit_order[end_block_x:end_block_z]
        if len(_is) > 0:
            diag_paulis = clifford.h(diag_paulis, _is)
            circuit.h(_is)

        # clear X with CNOT
        for i in range(end_block_z):
            if np.any(x_table[i, i + 1 :]):
                _js = np.where(x_table[i, i + 1 :])[0] + i + 1
                _is = i * np.ones(_js.shape, dtype=int)
                x_table[:, _js] = np.logical_xor(x_table[:, _js], x_table[:, _is])
                z_table[:, i] = np.logical_xor(
                    z_table[:, i],
                    np.mod(np.sum(z_table[:, _js], axis=1), 2).astype(bool),
                )

                diag_paulis = clifford.cx(
                    diag_paulis, qubit_order[_is], qubit_order[_js]
                )
                circuit.cx(qubit_order[_is], qubit_order[_js])

        # clear Z block with CZ
        for i in range(1, end_block_z):
            if np.any(z_table[i, :i]):
                _js = np.where(z_table[i, :i])[0]
                _is = i * np.ones(_js.shape, dtype=int)
                z_table[:, _js] = np.logical_xor(z_table[:, _js], x_table[:, _is])
                z_table[:, i] = np.logical_xor(
                    z_table[:, i],
                    np.mod(np.sum(x_table[:, _js], axis=1), 2).astype(bool),
                )

                diag_paulis = clifford.cz(
                    diag_paulis, qubit_order[_is], qubit_order[_js]
                )
                circuit.cz(qubit_order[_is], qubit_order[_js])

        # clear diag Z with S
        _is = np.where(z_table.diagonal()[:end_block_z])[0]
        z_table[_is, _is] = False
        if len(_is) > 0:
            diag_paulis = clifford.s(diag_paulis, qubit_order[_is])
            circuit.s(qubit_order[_is])

        # clear all diag X with H
        _is = np.arange(end_block_z)
        x_table[_is, _is] = False
        z_table[_is, _is] = True
        if len(_is) > 0:
            diag_paulis = clifford.h(diag_paulis, qubit_order[_is])
            circuit.h(qubit_order[_is])

        assert np.all(~diag_paulis.x), (
            "\n"
            + str(np.concatenate((z_table, x_table), axis=1).astype(int))
            + "\n"
            + str(paulis)
            + "\n"
            + str(diag_paulis)
        )

        return diag_paulis, circuit

    @staticmethod
    def _pack_diagonal(
        bit_table: "np.ndarray[np.bool]", start_index: int = 0
    ) -> tuple["np.ndarray[np.bool]", "np.ndarray[np.bool]", "np.ndarray[np.int]", int]:
        """
        Apply row operations, and column reordering to place "1" on the diagonal starting from the "start_index" diagonal element.

        Returns:
            _type_: _description_
        """

        bit_table = bit_table.copy()

        num_rows, num_cols = bit_table.shape

        row_range = np.arange(num_rows)
        col_order = np.arange(num_cols)
        row_op = np.eye(num_rows)

        while start_index < num_rows:
            failed = True
            for hot_col in range(start_index, num_cols):
                if np.any(bit_table[start_index:, hot_col]):
                    hot_row = start_index + np.argmax(bit_table[start_index:, hot_col])
                    failed = False

                    break
            if failed:
                break

            if hot_col != start_index:
                bit_table[:, [start_index, hot_col]] = bit_table[
                    :, [hot_col, start_index]
                ]
                col_order[[start_index, hot_col]] = col_order[[hot_col, start_index]]

            if hot_row != start_index:
                bit_table[[start_index, hot_row], :] = bit_table[
                    [hot_row, start_index], :
                ]
                row_op[[start_index, hot_row], :] = row_op[[hot_row, start_index], :]

            cond_rows = np.logical_and(
                bit_table[:, start_index], (row_range != start_index)
            )

            bit_table[cond_rows, :] = np.logical_xor(
                bit_table[cond_rows, :], bit_table[start_index, :][None, :]
            )
            row_op[cond_rows, :] = np.logical_xor(
                row_op[cond_rows, :], row_op[start_index, :][None, :]
            )

            start_index += 1

        return bit_table, row_op, col_order, start_index

    @staticmethod
    @staticmethod
    def _paulis_to_generators(paulis: PauliList) -> PauliList:
        """
        Extract generators from a set of Pauli strings using binary row reduction.
        
        Converts the Pauli strings to their symplectic representation (Z|X matrix)
        and performs Gaussian elimination to find a minimal spanning set of generators.
        The generators form a basis that can reconstruct all input Paulis.
        
        Args:
            paulis (PauliList): Input Pauli strings to find generators for.
        
        Returns:
            PauliList: Minimal set of generator Paulis (as PauliList).
        
        Notes:
            Uses binary row reduction (GF(2)) to maintain the symplectic structure.
        """

        num_qubits = paulis.num_qubits

        zx_table = np.concatenate((paulis.z, paulis.x), axis=1)

        row_zx_table = GeneralCommutation._row_echelon(zx_table)
        null_rows = np.all(~row_zx_table, axis=1)

        new_zx_table = row_zx_table[~null_rows, :]

        return PauliList.from_symplectic(
            new_zx_table[:, :num_qubits], new_zx_table[:, num_qubits:]
        )

    @staticmethod
    def _single_qubit_cummuting_generators(paulis: PauliList):
        """
        Identify single qubit commutation generators in a list of Pauli strings. Such a generator exist if a qubit is acted upon at most by the identity and a single Pauli for every Pauli string.

        Args:
            paulis (PauliList): _description_

        Returns:
            _type_: _description_
        """

        num_qubits = paulis.num_qubits

        new_z_table = np.zeros((num_qubits, num_qubits), dtype=bool)
        new_x_table = np.zeros((num_qubits, num_qubits), dtype=bool)

        zx_table = np.concatenate((paulis.z, paulis.x), axis=1)

        for q in range(num_qubits):
            u_qubit_zx = np.unique(zx_table[:, [q, num_qubits + q]], axis=0)
            if u_qubit_zx.shape[0] == 2 and np.all(u_qubit_zx[0] == 0):
                new_z_table[q, q] = u_qubit_zx[1, 0]
                new_x_table[q, q] = u_qubit_zx[1, 1]
            elif u_qubit_zx.shape[0] == 1:
                if np.all(u_qubit_zx[0] == 0):
                    new_z_table[q, q] = True
                else:
                    new_z_table[q, q] = u_qubit_zx[0, 0]
                    new_x_table[q, q] = u_qubit_zx[0, 1]

        return PauliList.from_symplectic(new_z_table, new_x_table)

    @staticmethod
    def _row_echelon(bit_matrix: "np.ndarray[np.bool]") -> "np.ndarray[np.bool]":
        """
        Applies Gauss-Jordan elimination on a binary matrix to produce row echelon form.

        Args:
            bit_matrix ("np.ndarray[np.bool]"): Input binary matrix.

        Returns:
            "np.ndarray[np.bool]": Row echelon form of the provided matrix.
        """
        re_bit_matrix = bit_matrix.copy()

        n_rows = re_bit_matrix.shape[0]
        n_cols = re_bit_matrix.shape[1]

        row_range = np.arange(n_rows)

        h_row = 0
        k_col = 0

        while h_row < n_rows and k_col < n_cols:
            if np.all(re_bit_matrix[h_row:, k_col] == 0):
                k_col += 1
            else:
                i_row = h_row + np.argmax(re_bit_matrix[h_row:, k_col])
                if i_row != h_row:
                    re_bit_matrix[[i_row, h_row], :] = re_bit_matrix[[h_row, i_row], :]

                cond_rows = np.logical_and(
                    re_bit_matrix[:, k_col], (row_range != h_row)
                )

                re_bit_matrix[cond_rows, :] = np.logical_xor(
                    re_bit_matrix[cond_rows, :], re_bit_matrix[h_row, :][None, :]
                )

                h_row += 1
                k_col += 1

        return re_bit_matrix
