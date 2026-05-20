"""
    ReferenceIntegrals.jl

High-order numerical integration utilities, backed by `GeneralizedGauss.jl`.

Provides adaptive-order Gauss-Legendre integration via
`reference_integral_gausslegendre`, which is used internally for computing
reference integrals in quadrature exactness checks and other verification
routines.

The arithmetic precision is controlled by the `T` type parameter (defaults
to `Float64`).  Pass `T=BigFloat` for arbitrary-precision reference integrals.
"""

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Internal: n-point Gauss-Legendre rule on [-1,1] via GeneralizedGauss.jl
# ─────────────────────────────────────────────────────────────────────────────

"""
    _gauss_legendre_nodes_weights(n::Int, ::Type{T}=Float64) where T

Compute the `n`-point Gauss-Legendre nodes and weights on `[-1, 1]`
in arithmetic type `T`.

Delegates to `GeneralizedGauss.gauss_legendre`.
"""
function _gauss_legendre_nodes_weights(n::Int, ::Type{T}=Float64) where T
    nodes, weights = GeneralizedGauss.gauss_legendre(n, T)
    return nodes, weights
end

# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

"""
    reference_integral_gausslegendre(f, interval;
                                     T=Float64,
                                     atol=eps(T)^(3/4),
                                     rtol=eps(T)^(3/4),
                                     max_order=4096) -> (T, Int)

Compute the definite integral of `f` over `interval = (a, b)` using
adaptive-order Gauss-Legendre quadrature in arithmetic type `T`.

The quadrature order is doubled starting from 64 until the integral estimate
stabilises to the requested tolerance or `max_order` is reached.

Returns a tuple `(I, order)` where `I` is the estimated integral and `order`
is the final Gauss-Legendre order used.

# Arguments
- `f` — any callable `f(x)`.
- `interval` — a `(a, b)` tuple.
- `T` — arithmetic type for the computation (default `Float64`).
- `atol` — absolute tolerance for convergence.
- `rtol` — relative tolerance for convergence.
- `max_order` — maximum quadrature order before giving up (default `4096`).

Internally uses `GeneralizedGauss.gauss_legendre` for the quadrature nodes
and weights.
"""
function reference_integral_gausslegendre(f, interval;
                                          T::Type{<:AbstractFloat} = Float64,
                                          atol = eps(T)^(T(3)/T(4)),
                                          rtol = eps(T)^(T(3)/T(4)),
                                          max_order::Int = 4096)
    _, _, T_interval = _interval_endpoint_type(interval, "reference_integral_gausslegendre interval")
    T == T_interval || throw(ArgumentError(
        "reference_integral_gausslegendre: keyword T ($T) must match interval type ($T_interval)."))
    a, b = interval[1], interval[2]
    mid  = (a + b) / 2
    half = (b - a) / 2

    order = 64
    x_ref, w_ref = _gauss_legendre_nodes_weights(order, T)
    I_old = half * sum(w_ref[k] * f(mid + half * x_ref[k]) for k in eachindex(x_ref))

    while order < max_order
        order_new = min(2 * order, max_order)
        x_new, w_new = _gauss_legendre_nodes_weights(order_new, T)
        I_new = half * sum(w_new[k] * f(mid + half * x_new[k]) for k in eachindex(x_new))

        if abs(I_new - I_old) <= atol + rtol * abs(I_new)
            return I_new, order_new
        end

        I_old = I_new
        order = order_new
    end

    return I_old, order
end
