
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import pauli_basis

from advanced_estimation.commutation.base_commutation import BaseCommutation
from advanced_estimation.estimation.pauli_estimation import estimate_cliques_expectation_values_and_covariances, overall_paulis_expectation_values_and_covariances


def iterative_quantum_state_tomography(
    state_circuit: QuantumCircuit,
    sampler: BaseSamplerV2,
    commutation_module: BaseCommutation,
    shots_budget: int,
    num_iterations: int = 3,
    advanced_cliques: bool = True,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform iterative quantum state tomography to reconstruct the density matrix of a quantum state.
    
    This method updates the shot allocation across cliques at each iteration to prioritize 
    measurements with higher variance (Neyman-like allocation).

    Args:
        state_circuit: QuantumCircuit that prepares the quantum state to be reconstructed.
        sampler: A Qiskit Sampler primitive for executing quantum circuits.
        commutation_module: Module defining the commutation relation for grouping.
        shots_budget: Total target number of shots for the entire process.
        num_iterations: Number of steps to refine the shot allocation.
        advanced_cliques: Whether to use advanced commuting clique finding.

    Returns:
        iter_density_matrices: Reconstructed density matrix for each iteration [num_iterations, dim, dim].
        iter_expectation_values: Pauli expectation values for each iteration [num_iterations, 4^n].
        iter_covariances: Covariance matrices for each iteration [num_iterations, 4^n, 4^n].
    """
    all_rhos = []
    all_exp_vals = []
    all_covs = []

    num_qubits = state_circuit.num_qubits
    paulis = pauli_basis(num_qubits)
    num_paulis = len(paulis)
    dimension = 2**num_qubits
    
    if advanced_cliques:
        cliques_paulis_indices = commutation_module.advanced_find_commuting_cliques(paulis)
    else:
        cliques_paulis_indices = commutation_module.find_commuting_cliques(paulis)
    num_cliques = len(cliques_paulis_indices)
    
    # Répartition uniforme pour la première itération
    cliques_shots = np.full(num_cliques, max(1, shots_budget // (num_cliques * num_iterations)), dtype=int)
    
    for i in range(num_iterations):

        exp_vals_cliques, covs_cliques = estimate_cliques_expectation_values_and_covariances(
            paulis, cliques_paulis_indices, cliques_shots, commutation_module, state_circuit, sampler
        )
        
        paulis_exp, paulis_cov = overall_paulis_expectation_values_and_covariances(
            num_paulis, cliques_paulis_indices, exp_vals_cliques, covs_cliques, cliques_shots
        )
        
        rho = np.zeros((dimension, dimension), dtype=np.complex128)
        for pauli, val in zip(paulis, paulis_exp):
            rho += val * pauli.to_matrix()
        rho /= dimension
        
        all_rhos.append(rho)
        all_exp_vals.append(paulis_exp)
        all_covs.append(paulis_cov)
        
        if i < num_iterations - 1:
            clique_variances = np.array([np.trace(c) for c in covs_cliques])
            clique_stds = np.sqrt(clique_variances)
            total_std = np.sum(clique_stds)
            
            # Allocation proportionnelle à l'écart-type (Neyman Allocation)
            if total_std > 0:
                cliques_shots = np.ceil((clique_stds / total_std) * (shots_budget / num_iterations)).astype(int)
            else:
                cliques_shots = np.full(num_cliques, max(1, shots_budget // (num_cliques * num_iterations)), dtype=int)
            cliques_shots[cliques_shots == 0] = 1

    return np.array(all_rhos), np.array(all_exp_vals), np.array(all_covs)


def density_matrix_to_state_vector(density_matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Convert a density matrix to a state vector by finding the eigenvector corresponding to the largest eigenvalue.
    Args:
        density_matrix: The density matrix to convert.
    Returns:
        The state vector corresponding to the largest eigenvalue of the density matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    max_index = np.argmax(eigenvalues)
    return eigenvectors[:, max_index]


def state_vector_to_dirac(state_vec, threshold=0.01) -> str:
    """
    Représente un vecteur d'état en notation de Dirac, en retirant la phase globale
    et en ne gardant que les termes significatifs.
    """
    first_idx = np.where(np.abs(state_vec) > threshold)[0]
    
    if len(first_idx) > 0:
        first_coeff = state_vec[first_idx[0]]
        global_phase_correction = np.conj(first_coeff) / np.abs(first_coeff)
        state_vec = state_vec * global_phase_correction

    num_qubits = int(np.log2(len(state_vec)))
    terms = []
    for i, coeff in enumerate(state_vec):
        if np.abs(coeff) > threshold:
            c_round = np.round(coeff, 3)
            label = format(i, f'0{num_qubits}b')
            terms.append(f"({c_round})|{label}>")
    
    return " + ".join(terms)
