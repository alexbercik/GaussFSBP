"""
    ReferenceIntegrals.jl

Self-contained placeholder for high-order numerical integration.

Currently implements Gauss-Legendre quadrature via the Golub-Welsch
eigenvalue method, sufficient for testing smooth functions on a finite interval.

TODO: Replace `_gauss_legendre_nodes_weights` with a call to
      `GeneralizedGauss.gauss_legendre` once the `GeneralizedGauss.jl`
      dependency is available (see `lib/GeneralizedGauss.jl/`).
"""

using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Golub-Welsch construction of n-point Gauss-Legendre rule on [-1,1]
# ─────────────────────────────────────────────────────────────────────────────

"""
    _gauss_legendre_nodes_weights(n::Int) -> (Vector{Float64}, Vector{Float64})

Compute the `n`-point Gauss-Legendre nodes and weights on `[-1, 1]` using
the Golub-Welsch tridiagonal eigenvalue method.

TODO: Replace this with `GeneralizedGauss.gauss_legendre(n)` once the
      `GeneralizedGauss.jl` local dependency has been added to the project.
"""
function _gauss_legendre_nodes_weights(n::Int)
    # Build the symmetric tridiagonal Jacobi matrix
    β = [k / sqrt(4k^2 - 1) for k in 1:(n-1)]
    J = SymTridiagonal(zeros(n), β)

    # Eigendecomposition — eigenvectors give the weights
    vals, vecs = eigen(J)

    # Nodes are eigenvalues; weights come from first components of eigenvectors
    x = vals
    w = 2.0 .* vecs[1, :] .^ 2   # sum(w) == 2 (length of [-1,1])

    # Sort by node value (eigen may not guarantee order)
    perm = sortperm(x)
    return x[perm], w[perm]
end

# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

"""
    reference_integral_gausslegendre(f, interval;
                                     atol=1e-13, rtol=1e-13,
                                     max_order=4096) -> (Float64, Int)

Compute the definite integral of `f` over `interval = (a, b)` using
adaptive-order Gauss-Legendre quadrature.

The quadrature order is doubled starting from 64 until the integral estimate
stabilises to the requested tolerance or `max_order` is reached.

Returns a tuple `(I, order)` where `I` is the estimated integral and `order`
is the final Gauss-Legendre order used.

# Arguments
- `f` — any callable `f(x)::Float64`.
- `interval` — a `(a, b)` tuple.
- `atol` — absolute tolerance for convergence (default `1e-13`).
- `rtol` — relative tolerance for convergence (default `1e-13`).
- `max_order` — maximum quadrature order before giving up (default `4096`).

TODO: Replace the internal `_gauss_legendre_nodes_weights` call with
      `GeneralizedGauss.gauss_legendre(n)` once `GeneralizedGauss.jl`
      is available.
"""
function reference_integral_gausslegendre(f, interval;
                                          atol::Float64 = 1e-13,
                                          rtol::Float64 = 1e-13,
                                          max_order::Int = 4096)
    a, b = Float64(interval[1]), Float64(interval[2])
    mid  = (a + b) / 2
    half = (b - a) / 2

    order = 64
    x_ref, w_ref = _gauss_legendre_nodes_weights(order)
    I_old = half * sum(w_ref[k] * f(mid + half * x_ref[k]) for k in eachindex(x_ref))

    while order < max_order
        order_new = min(2 * order, max_order)
        x_new, w_new = _gauss_legendre_nodes_weights(order_new)
        I_new = half * sum(w_new[k] * f(mid + half * x_new[k]) for k in eachindex(x_new))

        if abs(I_new - I_old) <= atol + rtol * abs(I_new)
            return I_new, order_new
        end

        I_old = I_new
        order = order_new
    end

    return I_old, order
end
