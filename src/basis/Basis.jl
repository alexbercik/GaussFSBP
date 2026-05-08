"""
    Basis.jl

Defines the abstract basis interface for GaussFSBP.

Every concrete basis type should subtype `AbstractBasis` and implement:
- `nbasis(basis)` — number of basis functions.
- `basis_functions(basis)` — return the underlying function objects (optional).
- `eval_basis(basis, x)` — evaluate all basis functions at scalar point `x`.
- `eval_basis_derivative(basis, x)` — evaluate all basis-function derivatives at `x`.
- `eval_basis_matrix(basis, xnodes)` — evaluate all basis functions at a vector of nodes.
- `eval_basis_derivative_matrix(basis, xnodes)` — evaluate all derivatives at a vector of nodes.

Fallback implementations provided here throw informative errors when a concrete
type has not yet implemented a required method.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Abstract type
# ─────────────────────────────────────────────────────────────────────────────

"""
    AbstractBasis

Supertype for all basis representations in GaussFSBP.

Concrete subtypes must implement `nbasis`, `eval_basis`, and optionally
`eval_basis_derivative` (required for derivative-based operators).
"""
abstract type AbstractBasis end

# ─────────────────────────────────────────────────────────────────────────────
# Generic interface with informative fallbacks
# ─────────────────────────────────────────────────────────────────────────────

"""
    nbasis(basis::AbstractBasis) -> Int

Return the number of basis functions in `basis`.
"""
function nbasis(basis::AbstractBasis)
    error("nbasis not implemented for $(typeof(basis)). " *
          "Please implement `nbasis(::$(typeof(basis)))`.")
end

"""
    basis_functions(basis::AbstractBasis)

Return the underlying callable function objects for `basis`, if available.
Not all basis types are required to implement this method.
"""
function basis_functions(basis::AbstractBasis)
    error("basis_functions not implemented for $(typeof(basis)). " *
          "Please implement `basis_functions(::$(typeof(basis)))`.")
end

"""
    eval_basis(basis::AbstractBasis, x) -> Vector

Evaluate all basis functions at scalar point `x`.
Returns a vector of length `nbasis(basis)`.
"""
function eval_basis(basis::AbstractBasis, x)
    error("eval_basis not implemented for $(typeof(basis)). " *
          "Please implement `eval_basis(::$(typeof(basis)), x)`.")
end

"""
    eval_basis_derivative(basis::AbstractBasis, x) -> Vector

Evaluate the derivatives of all basis functions at scalar point `x`.
Returns a vector of length `nbasis(basis)`.
"""
function eval_basis_derivative(basis::AbstractBasis, x)
    error("eval_basis_derivative not implemented for $(typeof(basis)). " *
          "Please implement `eval_basis_derivative(::$(typeof(basis)), x)`, " *
          "or supply derivative functions when constructing the basis.")
end

"""
    eval_basis_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate all basis functions at each node in `xnodes`.

The returned matrix `V` satisfies:

    V[i, j] = basis function j evaluated at node xnodes[i]

so `V` has size `(length(xnodes), nbasis(basis))`.

A default implementation is provided in `BasisEvaluation.jl` using
`eval_basis`. Concrete subtypes may override this for performance.
"""
function eval_basis_matrix end

"""
    eval_basis_derivative_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate the derivatives of all basis functions at each node in `xnodes`.

The returned matrix `Vx` satisfies:

    Vx[i, j] = derivative of basis function j evaluated at node xnodes[i]

so `Vx` has size `(length(xnodes), nbasis(basis))`.

A default implementation is provided in `BasisEvaluation.jl` using
`eval_basis_derivative`. Concrete subtypes may override this for performance.
"""
function eval_basis_derivative_matrix end
