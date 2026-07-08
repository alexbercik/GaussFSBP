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
    QuadratureExactnessReport{T<:AbstractFloat}

Result returned by `check_quadrature_exactness`.

# Fields
- `passed::Bool` — `true` iff all basis functions were integrated to within
  tolerance.
- `max_error::T` — maximum absolute error over all basis functions.
- `errors::Vector{T}` — per-basis-function absolute errors.
- `reference_orders::Vector{Int}` — final Gauss-Legendre order used for each
  basis function's reference integral, or `0` when exact moments were supplied.
- `min_weight::T` — minimum quadrature weight in the candidate rule.
"""
struct QuadratureExactnessReport{T<:AbstractFloat}
    passed::Bool
    max_error::T
    errors::Vector{T}
    reference_orders::Vector{Int}
    min_weight::T
    _atol::T
    _rtol::T
    _reference_integrals::Vector{T}
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
                               max_ref_order=4096,
                               quad_moments=nothing) -> QuadratureExactnessReport

Check whether the candidate quadrature rule with nodes `x` and weights `w`
exactly integrates every function in `quadbasis`.

The arithmetic precision is inferred from the element type of `w`.

# Arguments
- `quadbasis` — an `AbstractBasis` **or** a `Vector` of callable functions.
- `x` — quadrature nodes (length-`n` vector).
- `w` — quadrature weights (length-`n` vector).
- `interval` — reference interval `(a, b)` (default `(-1.0, 1.0)`).
- `atol` — absolute tolerance (default: precision-dependent).
- `rtol` — relative tolerance (default: precision-dependent).
- `max_ref_order` — maximum Gauss-Legendre order for reference integral
  (default `4096`).
- `quad_moments` — optional exact moments of `quadbasis`, in the same order as
  the supplied basis functions.  When supplied, these moments are used directly
  as the reference integrals instead of adaptive Gauss-Legendre integration.

# Returns
A `QuadratureExactnessReport`.

# Details

The candidate quadrature integral of a function `g` is computed as:

    I_candidate = sum(w[i] * g(x[i]) for i in eachindex(x))

Unless `quad_moments` is supplied, the reference integral is computed by
`reference_integral_gausslegendre`, which adaptively increases the
Gauss-Legendre order until the estimate stabilises.

The test passes for function `g` iff:

    abs(I_candidate - I_ref) <= atol + rtol * abs(I_ref)
"""
function check_quadrature_exactness(quadbasis, x, w;
                                    interval = (-1.0, 1.0),
                                    atol = nothing,
                                    rtol = nothing,
                                    max_ref_order::Int = 4096,
                                    quad_moments = nothing)
    x = collect(x)
    w = collect(w)
    T = _array_element_type(w, "quadrature weights")
    T <: AbstractFloat || throw(ArgumentError(
        "check_quadrature_exactness: weights must use a floating-point element type, got $T."))
    _, _, T_interval = _interval_endpoint_type(interval, "interval")
    _require_uniform_type("check_quadrature_exactness", [
        T, _array_element_type(x, "quadrature nodes"), T_interval])

    # Default tolerances based on precision
    _atol = atol !== nothing ? T(atol) : T(10) * eps(T)
    _rtol = rtol !== nothing ? T(rtol) : T(10) * eps(T)

    # Normalise quadbasis to a plain vector of callables
    funcs = _to_func_vector(quadbasis)

    nf       = length(funcs)
    errors   = Vector{T}(undef, nf)
    orders   = Vector{Int}(undef, nf)
    refs     = Vector{T}(undef, nf)

    if quad_moments !== nothing
        raw_quad_moments = collect(quad_moments)
        raw_quad_moments isa AbstractVector || throw(ArgumentError(
            "quad_moments must be a vector-like collection with one moment " *
            "per quadrature basis function."))
        length(raw_quad_moments) == nf || throw(ArgumentError(
            "quad_moments has length $(length(raw_quad_moments)), expected " *
            "$nf for the supplied quadrature basis."))

        refs .= T.(raw_quad_moments)
    end

    for (k, g) in enumerate(funcs)
        # Candidate integral from the supplied rule.
        I_cand = sum(w[i] * g(x[i]) for i in eachindex(x))

        if quad_moments === nothing
            # Reference integral from adaptive Gauss-Legendre integration.
            I_ref, ord = reference_integral_gausslegendre(
                g, interval; T = T,
                atol = _atol / 10, rtol = _rtol / 10,
                max_order = max_ref_order)
            refs[k] = T(I_ref)
            orders[k] = ord
        else
            # Exact moments were supplied, so no reference integration was run.
            orders[k] = 0
        end

        errors[k] = abs(T(I_cand) - refs[k])
    end

    max_err = maximum(errors)
    # Reuse the already-computed reference integrals to determine pass/fail
    passed = all(errors[k] <= _atol + _rtol * abs(refs[k]) for k in 1:nf)
    min_wt = T(minimum(w))

    return QuadratureExactnessReport{T}(passed, max_err, errors, orders, min_wt,
                                        _atol, _rtol, refs)
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
