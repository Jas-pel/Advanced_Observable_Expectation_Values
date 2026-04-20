import numpy as np
from qiskit.quantum_info import PauliList, pauli_basis
from utils import check_diag_transformation

from advanced_estimation.commutation.general_commuting import GeneralCommutation


def test_general_commutation_table():

    paulis = PauliList(["XXXX", "XXXZ", "XXYY", "ZXYZ"])

    commutation_module = GeneralCommutation()

    commutation_table = commutation_module.commutation_table(paulis)

    ref_commutation_table = np.array(
        [[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]
    ).astype(np.bool)

    assert np.all(commutation_table == ref_commutation_table)


def test_diagonalize_paulis_with_circuit():

    num_qubits = 3

    paulis = pauli_basis(num_qubits)

    commutation_module = GeneralCommutation()
    commutation_module.force_single_qubit_generators = False

    cliques_paulis_indices = commutation_module.find_min_commuting_cliques(paulis)

    cliques_paulis = []
    for clique_paulis_indices in cliques_paulis_indices:
        clique_paulis = paulis[clique_paulis_indices]
        assert np.all(commutation_module.commutation_table(clique_paulis))
        cliques_paulis.append(clique_paulis)

    for clique_paulis in cliques_paulis:
        print(clique_paulis)
        clique_diag_paulis, clique_diag_circuit = (
            commutation_module.diagonalize_paulis_with_circuit(clique_paulis)
        )

        assert check_diag_transformation(clique_paulis, clique_diag_paulis, clique_diag_circuit)
