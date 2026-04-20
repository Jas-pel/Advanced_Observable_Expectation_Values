import numpy as np
from qiskit import transpile
from qiskit.circuit.random import random_circuit as build_random_circuit
from qiskit.quantum_info import PauliList, Statevector, pauli_basis
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from utils import check_expectation_values_within_range

from advanced_estimation.commutation import BitwiseCommutation, GeneralCommutation
from advanced_estimation.estimation.pauli_estimation import (
    bitstrings_to_bits,
    diag_paulis_expectation_values_and_covariances,
    estimate_cliques_expectation_values_and_covariances,
    get_paulis_shots,
)


def test_bitstrings_to_bits():

    basis_states = ["000", "001", "010", "011", "100", "101", "110", "111"]
    ref_bits = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    print(bitstrings_to_bits(basis_states))
    print(ref_bits)
    assert np.all(bitstrings_to_bits(basis_states) == ref_bits)


def test_diag_paulis_expectation_values_and_covariances():

    paulis = PauliList(["III", "IIZ", "IZI", "IZZ", "ZII", "ZIZ", "ZZI", "ZZZ"])
    basis_states = ["000", "001", "010", "011", "100", "101", "110", "111"]
    values = list(range(1, 9))

    counts = dict(zip(basis_states, values))

    expectation_values, covariances = diag_paulis_expectation_values_and_covariances(paulis, counts)

    statevector = np.array(np.sqrt(values)) / np.sqrt(np.sum(values))

    paulis_matrices = paulis.to_matrix(array=True)
    ref_expectation_values = np.einsum("pij,i,j -> p", paulis_matrices, statevector, statevector)
    ref_covariances = np.einsum(
        "pij,qjk,i,k -> pq", paulis_matrices, paulis_matrices, statevector, statevector
    ) - np.einsum("p,q->pq", ref_expectation_values, ref_expectation_values)

    assert np.allclose(expectation_values, ref_expectation_values)
    assert np.allclose(covariances, ref_covariances)


def test_estimate_cliques_expectation_values_and_covariances_random_state():
    """
    Test if the Pauli expectation value within cliques are within tolerance.
    This test might statisticaly fail sometimes.
    """

    num_qubits = 2
    state_circuit_depth = 4
    total_shots = 4096

    simulator = AerSimulator()
    sampler = Sampler(mode=simulator)

    # Get random Paulis
    all_paulis = pauli_basis(num_qubits)
    mask = np.random.choice(a=[False, True], size=all_paulis.size, p=[0.7, 0.3])
    paulis = all_paulis[mask]
    num_paulis = paulis.size

    # Exact computation
    paulis_matrices = paulis.to_matrix(array=True)
    state_circuit = transpile(build_random_circuit(num_qubits=num_qubits, depth=state_circuit_depth), simulator)
    state_vector = Statevector(state_circuit).data
    exact_expectation_values = np.einsum("pij,i,j->p", paulis_matrices, state_vector.conj(), state_vector).real

    module_tuples = [
        ("Bitwise", BitwiseCommutation()),
        ("General", GeneralCommutation(False)),
    ]

    for i, (module_label, commutation_module) in enumerate(module_tuples):

        cliques_paulis_indices = commutation_module.find_commuting_cliques(paulis)
        num_cliques = len(cliques_paulis_indices)

        cliques_shots = [max(total_shots // num_cliques, 1) for _ in range(num_cliques)]

        cliques_expectation_values, cliques_covariances = estimate_cliques_expectation_values_and_covariances(
            paulis,
            cliques_paulis_indices,
            cliques_shots,
            commutation_module,
            state_circuit,
            sampler,
        )

        for paulis_expectation_values, paulis_covariances, clique_paulis_indices, clique_shots in zip(
            cliques_expectation_values, cliques_covariances, cliques_paulis_indices, cliques_shots
        ):
            paulis_variances = paulis_covariances.diagonal() / clique_shots
            ref_paulis_expectation_values = exact_expectation_values[clique_paulis_indices]

            assert check_expectation_values_within_range(
                paulis_expectation_values, ref_paulis_expectation_values, paulis_variances
            ), "This test may fails sometimes"


