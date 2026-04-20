import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import SparsePauliOp

from advanced_estimation.commutation.base_commutation import BaseCommutation
from advanced_estimation.estimation.pauli_estimation import (
    estimate_cliques_expectation_values_and_covariances,
    overall_paulis_expectation_values_and_covariances,
)


def iterative_estimate_sparse_pauli_op_expectation_value(
    observable: SparsePauliOp,
    state_circuit: QuantumCircuit,
    sampler: BaseSamplerV2,
    commutation_module: BaseCommutation,
    shots_budget: int,
    num_iterations: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Implement an iterative estimation for an observable.
    Devides the Pauli strings into clique based on the commutation_module.
    Assign an equal number of shots to each clique.
    Estimate the expectation values and covariances of the Paulis in the cliques.
    Update the shots per cliques to increase the one giving the largest variance.
    Repeat.

    Args:
        observable (SparsePauliOp): The observable to estimate
        state_circuit (QuantumCircuit): A quantum state as a QuantumCircuit
        sampler (BaseSamplerV2): The sampler on which to run the QuantumCircuits.
        commutation_module (BaseCommutation): A module that defines a commutation relation.
        shots_budget (int): The target number of total shots
        num_iterations (int): The number of iteration to update the shots per cliques

    Returns:
        NDArray[np.float64]: The Observable expectation values for each iteration
        NDArray[np.float64]]: The Observable variance for each iteration
    """

    paulis = observable.paulis
    coeffs = observable.coeffs.real

    num_paulis = paulis.size

    cliques_paulis_indices = commutation_module.find_min_commuting_cliques(paulis)
    num_cliques = len(cliques_paulis_indices)

    # Assume uniform distribution first
    cliques_shots = [max(shots_budget // num_cliques, 1) for _ in range(num_cliques)]

    iter_expectation_values = np.zeros(num_iterations)
    iter_variances = np.zeros(num_iterations)
    for i in range(num_iterations):

        actual_tot_shots = np.sum(cliques_shots)

        # For each clique, estimate the expectation values and covariance matrix of the diagonalized Paulis
        cliques_expectation_values, cliques_covariances = (
            estimate_cliques_expectation_values_and_covariances(
                paulis,
                cliques_paulis_indices,
                cliques_shots,
                commutation_module,
                state_circuit,
                sampler,
            )
        )

        # Combine the expectation values and covariance matrices of the cliques into overall expectation values and covariance matrix for the original Paulis
        paulis_expectation_values, paulis_covariances = (
            overall_paulis_expectation_values_and_covariances(
                num_paulis,
                cliques_paulis_indices,
                cliques_expectation_values,
                cliques_covariances,
                cliques_shots,
            )
        )

        weighted_cliques_variances = compute_weighted_cliques_variances(
            coeffs, cliques_paulis_indices, cliques_covariances
        )

        observable_expectation_value = np.matmul(coeffs, paulis_expectation_values)
        observable_variance = (
            np.einsum("ij,i,j", paulis_covariances, coeffs, coeffs) / actual_tot_shots
        )

        iter_expectation_values[i] = observable_expectation_value
        iter_variances[i] = observable_variance

        weighted_cliques_stds = np.sqrt(weighted_cliques_variances)

        # Handle potential numerical instabilities (zeros or NaNs)
        total_std = np.sum(weighted_cliques_stds)
        if total_std > 0 and not np.isnan(total_std):
            cliques_shots = np.ceil(
                weighted_cliques_stds / total_std * shots_budget
            ).astype(int)
        else:
            # Fallback to current allocation if we can't compute a better one
            cliques_shots = np.copy(cliques_shots).astype(int)

        cliques_shots[cliques_shots <= 0] = 1

    return iter_expectation_values, iter_variances


def compute_weighted_cliques_variances(
    coeffs: NDArray[np.float64],
    cliques_paulis_indices: list[NDArray[np.int_]],
    cliques_covariances: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Compute the weighted variance contribution of each measurement clique.
    
    For an observable Â = Σᵢ cᵢ Pᵢ, the variance is:
    Var(Â) = Σᵢⱼ cᵢ cⱼ Cov(Pᵢ, Pⱼ)
    
    This function computes per-clique contributions:
    weighted_var[k] = Σᵢⱼ ∈ clique_k cᵢ cⱼ Cov(Pᵢ, Pⱼ)
    
    Used for adaptive shot allocation: cliques with higher variance get more shots.
    
    Args:
        coeffs (NDArray[np.float64]): Observable coefficients [c₀, c₁, ..., cₙ].
        cliques_paulis_indices (list[NDArray[np.int_]]): Pauli indices for each clique.
        cliques_covariances (list[NDArray[np.float64]]): Covariance matrix per clique.
    
    Returns:
        NDArray[np.float64]: Weighted variance for each clique (shape: n_cliques).
    """

    num_cliques = len(cliques_paulis_indices)

    weighted_cliques_variances = np.zeros(num_cliques)
    for i, (clique_paulis_indices, clique_covariances) in enumerate(
        zip(cliques_paulis_indices, cliques_covariances)
    ):

        clique_coeffs = coeffs[clique_paulis_indices]
        variance = np.einsum(
            "jk,j,k", clique_covariances, clique_coeffs, clique_coeffs
        )
        weighted_cliques_variances[i] = np.maximum(variance, 0.0)

    return weighted_cliques_variances
