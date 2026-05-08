"""
    FunctionBasis.jl

Defines `FunctionBasis`, a concrete `AbstractBasis` backed by user-supplied
callable functions (and optionally their derivatives).
"""

"""
    FunctionBasis{F,D} <: AbstractBasis

A basis defined by explicit callable functions.

# Fields
- `funcs::Vector{F}` — vector of callable objects, one per basis function.
- `derivs::Union{Nothing,Vector{D}}` — optional vector of derivative callables.
  If `nothing`, calling `eval_basis_derivative` / `eval_basis_derivative_matrix`
  will throw an informative error.
- `interval::Tuple{Float64,Float64}` — the reference interval `(a, b)` over
  which the basis is defined (default `(-1.0, 1.0)`).

# Constructor

    FunctionBasis(funcs; derivs=nothing, interval=(-1.0, 1.0))

# Examples

```julia
# Monomial basis on [-1, 1] with explicit derivatives
funcs  = [x -> 1.0, x -> x, x -> x^2, x -> x^3]
derivs = [x -> 0.0, x -> 1.0, x -> 2x, x -> 3x^2]
basis  = FunctionBasis(funcs; derivs=derivs)
```
"""
struct FunctionBasis{F,D} <: AbstractBasis
    funcs::Vector{F}
    derivs::Union{Nothing,Vector{D}}
    interval::Tuple{Float64,Float64}
end

"""
    FunctionBasis(funcs; derivs=nothing, interval=(-1.0, 1.0))

Construct a `FunctionBasis` from a vector of callable functions `funcs`.

Optionally supply a matching vector `derivs` of derivative callables, and
specify the reference `interval` as a `(a, b)` tuple.
"""
function FunctionBasis(funcs::Vector{F};
                       derivs = nothing,
                       interval::Tuple{Float64,Float64} = (-1.0, 1.0)) where {F}
    if derivs === nothing
        return FunctionBasis{F,Nothing}(funcs, nothing, interval)
    else
        derivs_vec = Vector(derivs)
        if length(derivs_vec) != length(funcs)
            throw(ArgumentError(
                "length(derivs) ($(length(derivs_vec))) must equal " *
                "length(funcs) ($(length(funcs)))."))
        end
        D = eltype(derivs_vec)
        return FunctionBasis{F,D}(funcs, derivs_vec, interval)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# AbstractBasis interface implementation
# ─────────────────────────────────────────────────────────────────────────────

"""
    nbasis(basis::FunctionBasis) -> Int

Return the number of basis functions.
"""
nbasis(basis::FunctionBasis) = length(basis.funcs)

"""
    basis_functions(basis::FunctionBasis)

Return the vector of callable basis functions.
"""
basis_functions(basis::FunctionBasis) = basis.funcs

"""
    eval_basis(basis::FunctionBasis, x) -> Vector

Evaluate all basis functions at scalar point `x`.
"""
eval_basis(basis::FunctionBasis, x) = [f(x) for f in basis.funcs]

"""
    eval_basis_derivative(basis::FunctionBasis, x) -> Vector

Evaluate the derivatives of all basis functions at scalar point `x`.

Throws an error if no derivative functions were supplied at construction time.
"""
function eval_basis_derivative(basis::FunctionBasis, x)
    if basis.derivs === nothing
        error("No derivative functions were supplied for this FunctionBasis. " *
              "Pass a `derivs` vector to the FunctionBasis constructor to enable " *
              "derivative evaluation.")
    end
    return [d(x) for d in basis.derivs]
end
