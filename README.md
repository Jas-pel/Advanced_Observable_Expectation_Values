# Advanced Quantum Observable Estimation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-orange)

**Author:** Jasmin Pelletier

An efficient framework for estimating quantum observable expectation values by grouping commuting Pauli operators using graph theory, Clifford diagonalization, and adaptive shot allocation.

## 📋 Table of Contents

- [Motivation & Problem](#motivation--problem)
- [Project Architecture](#project-architecture)
- [Estimation Pipeline](#estimation-pipeline)
- [Commutation Strategies](#commutation-strategies)
- [Clifford Synthesis](#clifford-synthesis)
- [Estimation Module](#estimation-module)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [References](#references)

---

## Motivation & Problem

In quantum computing, **measurement destroys the quantum state**. To estimate the average value of an observable:

$$\langle \hat{A} \rangle = \sum_i c_i \langle P_i \rangle$$

we must repeat measurements (*shots*) many times. Measuring each Pauli string separately is prohibitively expensive (e.g., 100 Paulis × 1000 shots = 100,000 circuits).

**Key insight:** Group Pauli strings that **commute** into **cliques** and measure them simultaneously with a single circuit. This drastically reduces the total number of required measurements.

### Why This Matters

- **Shot Budget Efficiency:** Measure multiple compatible observables in one circuit
- **Variance Reduction:** Adaptive shot allocation targets high-variance measurements
- **Hardware Compatibility:** Works with both simulators and real quantum devices
- **Scalability:** Optimal for variational quantum algorithms (VQE, QAOA)

---

## Project Architecture

```
advanced_estimation/
├── commutation/                      # Grouping strategies
│   ├── __init__.py
│   ├── base_commutation.py           # Abstract base + NetworkX clique finding
│   ├── no_commuting.py               # No grouping (1 Pauli = 1 circuit)
│   ├── bitwise_commuting.py          # Qubit-by-qubit commutation
│   ├── general_commuting.py          # Full commutation + Clifford synthesis
│   └── clifford.py                   # Symplectic transformations (H, S, CX, CZ)
│
├── estimation/                       # Measurement & statistics
│   ├── __init__.py
│   ├── pauli_estimation.py           # Single Pauli expectation values & covariance
│   ├── observable_estimation.py      # Observable estimation + iterative refinement
│   └── state_tomography.py           # Quantum state tomography
│
├── tests/                            # Unit tests
│   ├── test_clifford.py
│   ├── test_bitwise.py
│   ├── test_general.py
│   └── test_estimation.py
│
├── usage/                            # Examples & comparison scripts
│   └── example_quantum_tomography.py
│
├── pyproject.toml                    # Project metadata & dependencies
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Estimation Pipeline

```
Observable Â = Σᵢ cᵢ Pᵢ
        │
        ▼
┌─────────────────────────────────┐
│ Step 1: Commutation Mapping     │
│ Build commutation_table()       │
│ Nodes=Paulis, Edges=Commute     │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Step 2: Min-Clique Grouping     │
│ Find maximal cliques (NetworkX) │
│ Each clique → one measurement   │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Step 3: Clifford Synthesis      │
│ Generate diagonalization        │
│ circuits (H, S, CX, CZ gates)   │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Step 4: Hardware Execution      │
│ Execute on simulator or device  │
│ Collect bitstring measurements  │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Step 5: Expectation Synthesis   │
│ Compute ⟨Pᵢ⟩ & covariance      │
│ Combine clique results          │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Step 6: Stability Analysis      │
│ Measure variance               │
│ Optionally iterate with adaptive│
│ shot reallocation              │
└─────────────────────────────────┘
         │
         ▼
    Final: ⟨Â⟩ ± σ
```

---

## Commutation Strategies

All commutation strategies inherit from `BaseCommutation` and implement:
- `commutation_table()` - Boolean matrix where (i,j)=True if Pauli_i commutes with Pauli_j
- `diagonalize_paulis_with_circuit()` - Returns diagonalized Paulis and transformation circuit

### Strategy Comparison

| Strategy | Condition | Cliques | Circuits | Efficiency |
|----------|-----------|---------|----------|-----------|
| **No Commutation** | Identity (each Pauli alone) | N | N | Baseline |
| **Bitwise Commutation** | Qubit-by-qubit match | Moderate | Moderate | Conservative |
| **General Commutation** | Full anticommutation count | Minimal | Minimal | **Optimal** |

### BaseCommutation (Abstract)

```python
def find_maximal_commuting_cliques(paulis) → list[NDArray]
    # Build commutation graph and find all maximal cliques using NetworkX
```

### NoCommutation

- **Commutation Table:** Identity matrix (only commutes with self)
- **Diagonalization:** H gates for X, S†H for Y, identity for Z
- **Result:** One circuit per Pauli (baseline worst-case)

### BitwiseCommutation

- **Condition:** Pᵢ and Pⱼ commute **element-wise per qubit**
- **Table:** `¬(Z₁·X₂ ⊕ X₁·Z₂)` per qubit, then `all()` across qubits
- **Diagonalization:** Single-qubit gates only (H, S†)
- **Use Case:** Conservative grouping, easy to understand

### GeneralCommutation

- **Condition:** Standard quantum commutation relation

$$[P_i, P_j] = 0 \iff \sum_q (z_i^{(q)} x_j^{(q)} + x_i^{(q)} z_j^{(q)}) \equiv 0 \pmod{2}$$

- **Algorithm:** [Gokhale et al., arXiv:1907.13623](https://arxiv.org/abs/1907.13623)
  1. Generator reduction (binary echelon form)
  2. Diagonal packing on X and Z blocks
  3. Elimination with H, CX, CZ, S gates
- **Result:** Minimum cliques → minimum total circuits

---

## Clifford Synthesis

Symplectic transformations on Pauli strings using the **Gottesman-Knill tableau** representation. Operates on binary form `(Z|X|φ)` in O(n) time instead of manipulating 2ⁿ × 2ⁿ matrices.

| Gate | Z Transformation | X Transformation | Phase Update |
|------|------------------|------------------|--------------|
| **H** | Z ↔ X | X ↔ Z | φ + 2·z·x |
| **S** | Z ← Z ⊕ X | X unchanged | φ + 2·z·x |
| **CX** | Zc ← Zc ⊕ Zt | Xt ← Xt ⊕ Xc | φ + 2·xc·zt·(1 − zc ⊕ xt) |
| **CZ** | Zc ← Zc ⊕ Xt, Zt ← Zt ⊕ Xc | Unchanged | φ + 2·xc·xt·(zc ⊕ zt) |

These transformations are used by `GeneralCommutation.diagonalize_paulis_with_circuit()` to rotate Paulis into the computational (Z) basis while generating the corresponding quantum circuit.

---

## Estimation Module

### Core Functions

**`pauli_estimation.py`**

| Function | Purpose |
|----------|---------|
| `bitstrings_to_bits()` | Convert measurement bitstrings to boolean arrays |
| `diag_paulis_expectation_values_and_covariances()` | Compute ⟨Pᵢ⟩ and Cov(Pᵢ, Pⱼ) from measurement counts |
| `estimate_cliques_expectation_values_and_covariances()` | Orchestrate measurement and statistics for each clique |
| `overall_paulis_expectation_values_and_covariances()` | Combine all clique results (shot-weighted averaging) |
| `get_paulis_shots()` | Calculate total shots allocated to each Pauli |

**`observable_estimation.py`**

| Function | Purpose |
|----------|---------|
| `iterative_estimate_sparse_pauli_op_expectation_value()` | Adaptive shot allocation across iterations |
| `compute_weighted_cliques_variances()` | Calculate Σⱼₖ cⱼ · Cov(Pⱼ, Pₖ) · cₖ per clique |

### Iterative Algorithm

```
Initialize: Uniform shot distribution
Loop for N iterations:
  1. Estimate ⟨Pᵢ⟩ and covariances for each clique
  2. Compute observable ⟨Â⟩ and variance
  3. Calculate weighted variance per clique
  4. Reallocate shots proportional to √(variance)
  5. Continue with next iteration
Result: Convergence toward optimal shot allocation
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-estimation.git
cd advanced-estimation

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing & symplectic operations |
| `networkx` | Commutation graphs & clique algorithms |
| `qiskit` | Quantum circuits & Pauli representation |
| `qiskit-aer` | Local quantum simulator |
| `qiskit-ibm-runtime` | Sampler V2 interface |
| `scipy` | Scientific computing utilities |
| `pytest` | Unit testing framework |
| `jupyter` | Interactive notebooks |

---

## Quick Start

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

from advanced_estimation.commutation import GeneralCommutation
from advanced_estimation.estimation.observable_estimation import (
    iterative_estimate_sparse_pauli_op_expectation_value,
)

# Initialize sampler
simulator = AerSimulator()
sampler = Sampler(mode=simulator)

# Create observable (weighted sum of Paulis)
observable = SparsePauliOp(
    ["XX", "YY", "ZZ", "IZ"],
    coeffs=[0.5, -0.5, 1.0, 0.2]
)

# Prepare quantum state
state_circuit = QuantumCircuit(2)
state_circuit.h(0)
state_circuit.cx(0, 1)

# Estimate with general commutation strategy
exp_values, variances = iterative_estimate_sparse_pauli_op_expectation_value(
    observable=observable,
    state_circuit=state_circuit,
    sampler=sampler,
    commutation_module=GeneralCommutation(),
    shots_budget=2000,
    num_iterations=5,
)

# Print results
final_estimate = exp_values[-1]
final_std = np.sqrt(variances[-1])
print(f"⟨Â⟩ = {final_estimate:.4f} ± {final_std:.4f}")
```

### Example: Comparing Strategies

```python
from advanced_estimation.commutation import (
    NoCommutation, BitwiseCommutation, GeneralCommutation
)

strategies = {
    "No Grouping": NoCommutation(),
    "Bitwise": BitwiseCommutation(),
    "General": GeneralCommutation(),
}

for name, module in strategies.items():
    cliques = module.find_maximal_commuting_cliques(observable.paulis)
    print(f"{name}: {len(cliques)} cliques")
```

---

## Testing

Run the test suite to verify functionality:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_general.py -v

# With coverage report
pytest tests/ --cov=advanced_estimation
```

### Test Structure

- `test_clifford.py` - Symplectic transformations and Gottesman-Knill tableau
- `test_bitwise.py` - Bitwise commutation strategy
- `test_general.py` - General commutation strategy
- `test_estimation.py` - Expectation value estimation

---

## References

1. **Gokhale et al.** (2019) - *Minimizing State Preparations in Variational Quantum Eigensolver by Partitioning into Commuting Families*  
   arXiv:[1907.13623](https://arxiv.org/abs/1907.13623)

2. **Aaronson & Gottesman** (2004) - *Improved Simulation of Stabilizer Circuits*  
   Phys. Rev. A **70**, 052328

3. **Qiskit Documentation** - https://qiskit.org/documentation/

4. **Pauli Strings & Stabilizer Codes**  
   Nielsen & Chuang, *Quantum Computation and Quantum Information* (2010)

---

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit issues or pull requests to improve this project.

## Contact

**Author:** Jasmin Pelletier  
**Email:** pelletierjasmin7@gmail.com
