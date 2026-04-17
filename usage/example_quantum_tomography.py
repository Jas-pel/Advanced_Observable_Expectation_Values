import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import (DensityMatrix, Statevector, purity,
                                 state_fidelity)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from advanced_estimation.commutation.general_commuting import \
    GeneralCommutation
from usage.state_tomography import (density_matrix_to_state_vector,
                                    iterative_quantum_state_tomography,
                                    state_vector_to_dirac)


def create_tab_synthesis(state_circuit, sampler, commutation_module, shots_list, state_name="Etat_Quantique"):
    """
    Effectue la tomographie, calcule les métriques de manière robuste et génère les exports.
    Version corrigée pour éviter les QiskitError de validation de matrice.
    """
    fidelities = []
    rho_list = []
    actual_state_vec = Statevector.from_instruction(state_circuit)
    sigma_ideal = DensityMatrix(actual_state_vec)
    
    print(f"--- Début de l'analyse : {state_name} ---")
    
    for shots in shots_list:
        # 1. Exécution de la tomographie itérative
        all_rho, _, _ = iterative_quantum_state_tomography(
            state_circuit=state_circuit,
            sampler=sampler,
            commutation_module=commutation_module,
            shots_budget=shots,
            advanced_cliques=False,
            num_iterations=1
        )
        
        rho_est = all_rho[-1]
        # On s'assure d'extraire les données numériques si c'est un objet Qiskit
        rho_data = rho_est.data if hasattr(rho_est, 'data') else rho_est
        rho_list.append(rho_data)
        
        # 2. Calcul des métriques
        est_state_vec = density_matrix_to_state_vector(rho_est)
        fid = state_fidelity(actual_state_vec, est_state_vec)
        fidelities.append(fid)
        
        print(f"Shots: {shots:5} | Fidélité: {fid:.4f}")

 
    # Distance de Trace : 1/2 * ||rho - sigma||_1 (via SVD)
    distances = []
    for r in rho_list:
        diff = r - sigma_ideal.data
        d_trace = 0.5 * np.sum(np.linalg.svd(diff, compute_uv=False))
        distances.append(d_trace)
    
    data = {
        "Shots": shots_list,
        "Fidélité (F)": [f"{f:.4f}" for f in fidelities],
        "Dist. Trace": [f"{d:.4f}" for d in distances],
        "Incertitude (1-F)": [f"{1-f:.4e}" for f in fidelities]
    }
    
    df = pd.DataFrame(data)

    # 4. Exports (LaTeX, PDF, PNG)
    # Export LaTeX (propre pour ton rapport à Sherbrooke)
    df.to_latex(f"tableau_{state_name}.tex", index=False, 
                caption=f"Analyse de convergence pour {state_name}", label=f"tab:{state_name}")
    
    # Graphique de convergence
    plt.figure(figsize=(8, 5))
    plt.plot(shots_list, fidelities, marker='s', linestyle='-', color='#d62728', label=f'Fidélité {state_name}')
    plt.xscale('log')
    plt.xlabel('Nombre de Shots (log)')
    plt.ylabel('Fidélité')
    plt.title(f'Analyse de Convergence - {state_name}')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.savefig(f"graphique_{state_name}.png", dpi=300)
    plt.close() # Ferme la figure pour libérer la mémoire
    
    # Image du tableau PDF
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center').scale(1.2, 1.8)
    plt.savefig(f"tableau_{state_name}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"\nSynthèse terminée avec succès. Fichiers : tableau_{state_name}.tex/.pdf et graphique_{state_name}.png")
    return df


def analyse_impact_nb_shot_onfidelity(state_circuit, sampler, commutation_module, shots_list):
    """
    Analyse de l'impact du nombre de shots sur la fidélité de l'estimation d'état quantique.
    Genere un graphique de la fidélité en fonction du nombre de shots.
    """
    fidelities = []
    actual_state_vec = Statevector.from_instruction(state_circuit)
    for shots in shots_list:
        all_rho, _, _ = iterative_quantum_state_tomography(
            state_circuit=state_circuit,
            sampler=sampler,
            commutation_module=commutation_module,
            shots_budget=shots,
            advanced_cliques=False,
            num_iterations=1
        )
        rho = all_rho[-1]
        estimate_state_vec = density_matrix_to_state_vector(rho)
        fidelity = state_fidelity(actual_state_vec, estimate_state_vec)
        fidelities.append(fidelity)
        print(f"Shots: {shots}, Fidelity: {fidelity:.4f}")

    plt.plot(shots_list, fidelities, marker='o')
    plt.xlabel('Number of Shots')
    plt.ylabel('Fidelity')
    plt.title('Impact of Number of Shots on Quantum State Tomography Fidelity')
    plt.grid(True)
    plt.savefig("impact_nb_shots_on_fidelity.png")

# 1. Préparation du circuit d'état (ex: un état Bell)
# qc = QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)

# Circuit ghz n qubit
# n = 3
# qc = QuantumCircuit(n)
# qc.h(0)
# for i in range(1, n):
#     qc.cx(0, i)
qc = QuantumCircuit(3)
qc.h(range(3))

simulator = AerSimulator()
sampler = Sampler(mode=simulator)
commutation_module = GeneralCommutation() 
    
create_tab_synthesis(
    state_circuit=qc,
    sampler=sampler,
    commutation_module=commutation_module,
    shots_list=[256, 512, 1024, 2048, 4096, 8192, 16384],
    state_name="TRIPLE_HADAMARD"
)    commutation_module=commutation_module,
    shots_list=[256, 512, 1024, 2048, 4096, 8192, 16384],
    state_name="TRIPLE_HADAMARD"
)