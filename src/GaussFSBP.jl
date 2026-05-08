"""
    GaussFSBP

A Julia package for building generalized SBP/FEM/DG/SEM-style element operators
from arbitrary approximation bases.

Currently, the package provides:
- An abstract basis interface (`AbstractBasis`) and a concrete `FunctionBasis` type.
- Utilities for evaluating basis functions and their derivatives at node vectors.
- A quadrature exactness checker (`check_quadrature_exactness`).

Operator construction (mass/norm matrix, differentiation matrix, SBP matrices,
etc.) is planned for a future release.
"""
module GaussFSBP

using LinearAlgebra

# ── Basis ────────────────────────────────────────────────────────────────────
include("basis/Basis.jl")
include("basis/FunctionBasis.jl")
include("basis/BasisEvaluation.jl")

# ── Utilities ────────────────────────────────────────────────────────────────
include("utils/ReferenceIntegrals.jl")

# ── Verification ─────────────────────────────────────────────────────────────
include("verification/Verification.jl")

# ── Builders (placeholder) ───────────────────────────────────────────────────
include("builders/OperatorBuilders.jl")

# ── Exports ──────────────────────────────────────────────────────────────────
# Basis interface
export AbstractBasis
export nbasis, basis_functions
export eval_basis, eval_basis_derivative
export eval_basis_matrix, eval_basis_derivative_matrix

# Concrete basis types
export FunctionBasis

# Verification
export check_quadrature_exactness, QuadratureExactnessReport

# Reference integrals
export reference_integral_gausslegendre

end # module GaussFSBP
