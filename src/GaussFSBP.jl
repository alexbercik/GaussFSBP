"""
    GaussFSBP

A Julia package for building generalized SBP/FEM/DG/SEM-style element operators
from arbitrary approximation bases.

The package provides:
- An abstract basis interface (`AbstractBasis`) and a concrete `FunctionBasis` type.
- Utilities for evaluating basis functions and their derivatives at node vectors.
- FSBP operator construction (`build_fsbp_operator`) with automatic quadrature
  generation via `GeneralizedGauss.jl`.
- Comprehensive verification suites for quadrature exactness and operator
  properties (`check_quadrature_exactness`, `check_fsbp_operator`).
"""
module GaussFSBP

using LinearAlgebra
using Printf
using GeneralizedGauss

# ── Basis ────────────────────────────────────────────────────────────────────
include("basis/Basis.jl")
include("basis/FunctionBasis.jl")
include("basis/GeneralizedGaussInterop.jl")
include("basis/BasisEvaluation.jl")

# ── Utilities ────────────────────────────────────────────────────────────────
include("utils/ReferenceIntegrals.jl")

# ── Verification (quadrature — no dependency on builders) ───────────────────
include("verification/QuadratureVerification.jl")

# ── Builders ────────────────────────────────────────────────────────────────
include("builders/OperatorBuilders.jl")

# ── Verification (operator — depends on FSBPOperator from builders) ─────────
include("verification/OperatorVerification.jl")

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
export check_fsbp_operator, FSBPOperatorReport

# Operator construction
export FSBPOperator, build_fsbp_operator

# Reference integrals
export reference_integral_gausslegendre

# Re-exported from GeneralizedGauss
export quadbasis,
    compute_moments,
    compute_gauss_rule,
    compute_gauss_rules,
    orthogonalize_basis,
    check_ECT_system,
    check_T_system,
    gauss_legendre

end # module GaussFSBP
