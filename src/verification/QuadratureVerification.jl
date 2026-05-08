"""
    QuadratureVerification.jl

Implements `check_quadrature_exactness`, which tests whether a candidate
quadrature rule `(x, w)` exactly integrates a supplied set of basis functions
(the "quadrature basis").
"""

# ─────────────────────────────────────────────────────────────────────────────
# Report struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    QuadratureExactnessReport

Result returned by `check_quadrature_exactness`.

# Fields
- `passed::Bool` — `true` iff all basis functions were integrated to within
  tolerance.
- `max_error::Float64` — maximum absolute error over all basis functions.
- `errors::Vector{Float64}` — per-basis-function absolute errors.
- `reference_orders::Vector{Int}` — final Gauss-Legendre order used for each
  basis function's reference integral.
- `min_weight::Float64` — minimum quadrature weight in the candidate rule.
"""
struct QuadratureExactnessReport
    passed::Bool
    max_error::Float64
    errors::Vector{Float64}
    reference_orders::Vector{Int}
    min_weight::Float64
    _atol::Float64
    _rtol::Float64
    _reference_integrals::Vector{Float64}
end

function Base.show(io::IO, r::QuadratureExactnessReport)
    status = r.passed ? "PASSED" : "FAILED"
    println(io, "QuadratureExactnessReport [$status]")
    println(io, "  max_error        : ", r.max_error)
    println(io, "  min_weight       : ", r.min_weight)
    println(io, "  num basis funcs  : ", length(r.errors))
    if !r.passed
        # Identify basis functions whose error exceeds the tolerance threshold
        failed = findall(
            k -> r.errors[k] > r._atol + r._rtol * abs(r._reference_integrals[k]),
            1:length(r.errors))
        println(io, "  failed indices   : ", failed)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_quadrature_exactness(quadbasis, x, w;
                               interval=(-1.0, 1.0),
                               atol=1e-12,
                               rtol=1e-12,
                               max_ref_order=4096) -> QuadratureExactnessReport

Check whether the candidate quadrature rule with nodes `x` and weights `w`
exactly integrates every function in `quadbasis`.

# Arguments
- `quadbasis` — an `AbstractBasis` **or** a `Vector` of callable functions.
- `x` — quadrature nodes (length-`n` vector).
- `w` — quadrature weights (length-`n` vector).
- `interval` — reference interval `(a, b)` (default `(-1.0, 1.0)`).
- `atol` — absolute tolerance (default `1e-12`).
- `rtol` — relative tolerance (default `1e-12`).
- `max_ref_order` — maximum Gauss-Legendre order for reference integral
  (default `4096`).

# Returns
A `QuadratureExactnessReport`.

# Details

The candidate quadrature integral of a function `g` is computed as:

    I_candidate = sum(w[i] * g(x[i]) for i in eachindex(x))

The reference integral is computed by `reference_integral_gausslegendre`,
which adaptively increases the Gauss-Legendre order until the estimate
stabilises.

The test passes for function `g` iff:

    abs(I_candidate - I_ref) <= atol + rtol * abs(I_ref)
"""
function check_quadrature_exactness(quadbasis, x, w;
                                    interval = (-1.0, 1.0),
                                    atol::Float64 = 1e-12,
                                    rtol::Float64 = 1e-12,
                                    max_ref_order::Int = 4096)
    # Normalise quadbasis to a plain vector of callables
    funcs = _to_func_vector(quadbasis)

    nf       = length(funcs)
    errors   = Vector{Float64}(undef, nf)
    orders   = Vector{Int}(undef, nf)
    refs     = Vector{Float64}(undef, nf)

    for (k, g) in enumerate(funcs)
        # Candidate integral
        I_cand = sum(w[i] * g(x[i]) for i in eachindex(x))

        # Reference integral (adaptive Gauss-Legendre)
        I_ref, ord = reference_integral_gausslegendre(
            g, interval; atol = atol / 10, rtol = rtol / 10,
            max_order = max_ref_order)

        refs[k]   = I_ref
        errors[k] = abs(I_cand - I_ref)
        orders[k] = ord
    end

    max_err = maximum(errors)
    # Reuse the already-computed reference integrals to determine pass/fail
    passed = all(errors[k] <= atol + rtol * abs(refs[k]) for k in 1:nf)
    min_wt = minimum(w)

    return QuadratureExactnessReport(passed, max_err, errors, orders, min_wt,
                                     atol, rtol, refs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _to_func_vector(quadbasis) -> Vector

Convert an `AbstractBasis` or a plain `Vector` of callables to a `Vector` of
callable objects.
"""
function _to_func_vector(quadbasis::AbstractBasis)
    return basis_functions(quadbasis)
end

function _to_func_vector(quadbasis::Vector)
    return quadbasis
end
