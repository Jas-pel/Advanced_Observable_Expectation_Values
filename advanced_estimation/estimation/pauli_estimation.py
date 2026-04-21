import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import PauliList
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from advanced_estimation.commutation.base_commutation import BaseCommutation


def estimate_cliques_expectation_values_and_covariances(
    paulis: PauliList,
    cliques_paulis_indices: list[NDArray[np.int_]],
    cliques_shots: NDArray[np.float64],
    commutation_module: BaseCommutation,
    state_circuit: QuantumCircuit,
    sampler: BaseSamplerV2,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    Estimate the expectation values and covariance matrix for a list of Pauli strings assigned to a list of cliques. All the Pauli strings inside a clique should commute in the sense given by the `commutation_module`. the expectation values and covariance matrix are returned for each clique.

    Args:
        paulis (PauliList): The Pauli strings
        cliques_paulis_indices (list[NDArray[np.int_]]): Element `i` in the list contains the indices of the Paulis in the ith clique
        cliques_shots (NDArray[np.float64]): The number of shots for each clique
        commutation_module (BaseCommutation): A module that defines a commutation relation.
        state_circuit (QuantumCircuit): The state on each to estimate the expectation values and covariances.
        sampler (BaseSamplerV2): The sampler on which to run the QuantumCircuits.

    Returns:
        list[NDArray[np.float64]]: Element `i` in the list contains the expectation values of the Paulis in clique i.
        list[NDArray[np.float64]]]: Element `i` in the list contains the covariance matrix of the Paulis in clique i.
    """

    assert len(cliques_shots) == len(cliques_paulis_indices)

    # Verify that Pauli strings inside each clique commute
    cliques_paulis = []
    for clique_paulis_indices in cliques_paulis_indices:
        clique_paulis = paulis[clique_paulis_indices]
        assert np.all(
            commutation_module.commutation_table(clique_paulis)
        ), "The Pauli strings inside a clique should all commute in the sense given by the `commutation_module`"
        cliques_paulis.append(clique_paulis)

    # Diagonalize the Pauli strings of each clique and get the corresponding circuits
    cliques_diag_paulis = []
    cliques_diag_circuits = []
    for clique_paulis in cliques_paulis:
        clique_diag_paulis, clique_diag_circuit = (
            commutation_module.diagonalize_paulis_with_circuit(clique_paulis)
        )
        cliques_diag_paulis.append(clique_diag_paulis)
        cliques_diag_circuits.append(clique_diag_circuit)

    # Construct the estimation circuits
    estimation_circuits = []
    for clique_diag_circuit in cliques_diag_circuits:
        estimation_circuit = state_circuit.compose(clique_diag_circuit)
        estimation_circuit.measure_all()
        estimation_circuits.append(estimation_circuit)

    # Transpile the estimation circuits to the target backend
    pass_manager = generate_preset_pass_manager(backend=sampler.mode, optimization_level=1)
    isa_circuits = pass_manager.run(estimation_circuits)

    # Run the circuits and get the results
    pubs = []
    for isa_circuit, clique_shots in zip(isa_circuits, cliques_shots):
        pubs.append((isa_circuit, None, clique_shots))

    job = sampler.run(pubs)
    results = job.result()

    cliques_expectation_values = []
    cliques_covariances = []
    for result, clique_diag_paulis in zip(results, cliques_diag_paulis):
        counts = result.data.meas.get_counts()
        clique_expectation_values, clique_covariances = (
            diag_paulis_expectation_values_and_covariances(clique_diag_paulis, counts)
        )

        cliques_expectation_values.append(clique_expectation_values)
        cliques_covariances.append(clique_covariances)

    return cliques_expectation_values, cliques_covariances


def overall_paulis_expectation_values_and_covariances(
    num_paulis: int,
    cliques_paulis_indices: list[NDArray[np.int_]],
    cliques_expectation_values: list[NDArray[np.float64]],
    cliques_covariances: list[NDArray[np.float64]],
    cliques_shots: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Combine the expectation values and covariance matrices of multiple cliques into overall expectation values and covariance matrix.

    Args:
        num_paulis (int): Total number of paulis
        cliques_paulis_indices (list[NDArray[np.int_]]): Element `i` in the list contains the indices of the Paulis in the ith clique
        cliques_expectation_values (list[NDArray[np.float64]]): Element `i` in the list contains the expectation values of the Paulis in clique i.
        cliques_covariances (list[NDArray[np.float64]]): Element `i` in the list contains the covariance matrix of the Paulis in clique i.
        cliques_shots (NDArray[np.int_]): The number of shots for each clique

    Returns:
        NDArray[np.float64]: The overall expectation values
        NDArray[np.float64]: The overall covariance matrix
    """
    total_shots_per_paulis = get_paulis_shots(num_paulis, cliques_paulis_indices, cliques_shots)
    paulis_expectation_values = np.zeros(num_paulis)
    paulis_covariances = np.zeros((num_paulis, num_paulis))

    for (
        clique_paulis_indices,
        clique_expectation_values,
        clique_covariances,
        clique_shot,
    ) in zip(
        cliques_paulis_indices,
        cliques_expectation_values,
        cliques_covariances,
        cliques_shots,
    ):

        paulis_expectation_values[clique_paulis_indices] += clique_shot * clique_expectation_values

        grid = np.ix_(clique_paulis_indices, clique_paulis_indices)
        paulis_covariances[grid] += clique_shot * clique_covariances

    paulis_expectation_values /= total_shots_per_paulis
    paulis_covariances *= np.sum(cliques_shots) / (
        np.outer(total_shots_per_paulis, total_shots_per_paulis)
    )

    return paulis_expectation_values, paulis_covariances


def get_paulis_shots(
    num_paulis: int,
    cliques_paulis_indices: list[NDArray[np.int_]],
    cliques_shots: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Compute the number of shots for each Pauli

    Args:
        num_paulis (int): Total number of paulis
        cliques_paulis_indices (list[NDArray[np.int_]]): Element `i` in the list contains the indices of the Paulis in the ith clique
        cliques_shots (NDArray[np.int_]): The number of shots for each clique

    Returns:
        NDArray[np.int_]: The number of shots for each Pauli
    """
    paulis_shots = np.zeros(num_paulis)

    for clique_indices, clique_shots in zip(cliques_paulis_indices, cliques_shots):
        paulis_shots[clique_indices] += clique_shots

    return paulis_shots


def bitstrings_to_bits(bit_strings: list[str]) -> NDArray[np.bool]:
    """
    Convert a list of bitstrings (str) into a 2d numpy.ndarray of bools.

    Args:
        bit_strings (str): Strings of '0' and '1'. Little endian assumed.

    Returns:
        NDArray[np.bool_]: Array of bits as bools. Each row is a bit string.
    """
    num_bitstrings = len(bit_strings)
    num_qubits = len(bit_strings[0])
    bits = np.zeros((num_bitstrings, num_qubits), dtype=bool)

    for i, bitstring in enumerate(bit_strings):
        bits[i] = np.array(list(bitstring[::-1])) == "1"

    return bits


def diag_paulis_expectation_values_and_covariances(
    diag_paulis: PauliList, counts: dict
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes the expectation value and the covariances of a list of diagonal Pauli string based on counts.

    Args:
        diag_paulis (PauliList): A list of diagonal Pauli strings
        counts (dict): Keys : basis state bitstring (ex : '1100'),
                       Values : number of times this state was obtained

    Returns:
        NDArray[np.float64]: The expectation values
        NDArray[np.float64]: The covariances matrix
    """
    assert np.all(~diag_paulis.x)

    # Parse measurement data
    bitstrings, measurements = zip(*counts.items())
    bits_matrix = np.array(bitstrings_to_bits(bitstrings), dtype=int)
    probabilities = np.array(measurements, dtype=float) / np.sum(measurements)

    # Extract Pauli properties: phase_sign ∈ {+1, -1} from phase ∈ {0,1,2,3}
    phases_signs = 1 - 2 * (diag_paulis.phase // 2)
    paulis_z = np.array(diag_paulis.z, dtype=int)

    # Compute eigenvalues: λ(P,|b⟩) = phase(P) × (-1)^(# of Z on |1⟩)
    n_active_z = np.einsum("bq,pq->bp", bits_matrix, paulis_z)
    parity_signs = 1 - 2 * (n_active_z % 2)
    eigen_vals = phases_signs * parity_signs

    # Expectation values: ⟨P⟩ = Σ_b prob(b) × λ(P, b)
    exp_values = np.einsum("b,bp->p", probabilities, eigen_vals)

    # Covariances: Cov(P,Q) = ⟨PQ⟩ - ⟨P⟩⟨Q⟩
    expected_products = np.einsum("b,bp,bk->pk", probabilities, eigen_vals, eigen_vals)
    covariances = expected_products - np.outer(exp_values, exp_values)

    return exp_values, covariances
