# Solving Helmholtz problems with finite elements on a quantum annealer

In this repository, users will find the code able to reproduce the results of [remi2025]. The repository contains two main components:

- **aqae.py**: A home-made library that implements the Adaptive Quantum Annealer Eigensolver (aqae) algorithm to solve generalized eigenvalue problems (gEVPs) of the form
  $$
  Hx = \lambda Mx,
  $$
  where **H** is a Hermitian matrix and **M** is a positive definite matrix.

- **fem.py**: A high order finite element code used to generate the stiffness and mass matrices necessary for solving a one-dimensional Helmholtz problem.

## Methodology

In this section, we briefly explain the main principles behind the method used in [remi2025].

The goal is to solve the generalized eigenvalue problem:
$$
Hx = \lambda Mx,
$$
where:
- **H** represents the stiffness matrix obtained via a finite element discretization,
- **M** represents the mass matrix,
- $\lambda$ denotes the eigenvalues,
- $x$ denotes the eigenvectors.

*Key steps include:*
- **Finite Element Discretization:** Formulating the one-dimensional Helmholtz problem and constructing a suitable high order finite element basis.
- **Matrix Assembly:** Computing the stiffness and mass matrices that discretize the problem.
- **Adaptive Quantum Annealer Eigensolver (aqae):** Implementing an adaptive algorithm that leverages quantum annealing techniques to solve the gEVP efficiently.

*Additional theoretical details, convergence analysis, and error estimates can be provided here.*

## Getting Started

Follow these instructions to run the provided code.

### Prerequisites

- **Python 3.x**
- Required Python libraries (e.g., NumPy, SciPy). Install them using:
  ```bash
  pip install -r requirements.txt
