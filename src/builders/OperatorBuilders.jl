"""
    OperatorBuilders.jl

Construction routines for function-space SBP (FSBP) operators.

Given an approximation basis F and a quadrature basis G, this module:
1. Optionally orthogonalizes both bases via GeneralizedGauss.
2. Constructs a quadrature rule from G via GeneralizedGauss.
3. Builds the first-derivative FSBP operator D = H⁻¹Q.
4. Constructs nodal extrapolation operators tL, tR.
5. Packages everything into an `FSBPOperator` struct.

Two construction paths are supported:
- **nn == nb** (number of nodes equals number of basis functions):
  the derivative operator is unique and computed via Vandermonde inversion.
- **nn > nb** (more nodes than basis functions):
  the system is underdetermined; a least-squares (minimum-norm) solution
  is used by default.  An optimization-based path (Glaubitz et al. 2025)
  can be selected via `use_optimization=true`.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FSBPOperator struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    FSBPOperator{T<:Real}

Holds all components of a first-derivative FSBP operator on a 1-D interval.

The type parameter `T` is the arithmetic precision (e.g. `Float64` or
`BigFloat`).  All matrix and vector fields use this element type.

# Fields
- `D::Matrix{T}`  — differentiation matrix (nn x nn), D = H⁻¹Q.
- `H::Diagonal{T,Vector{T}}`  — norm / mass matrix (diagonal, nn x nn).
- `Q::Matrix{T}`  — weak-derivative matrix, Q = H D.
- `S::Matrix{T}`  — skew-symmetric part, S = Q - E/2.
- `E::Matrix{T}`  — boundary matrix, E = tR tRᵀ - tL tLᵀ = Q + Qᵀ.
- `tL::Vector{T}` — left boundary extrapolation operator (length nn).
- `tR::Vector{T}` — right boundary extrapolation operator (length nn).
- `x::Vector{T}`  — quadrature nodes.
- `w::Vector{T}`  — quadrature weights.
- `op_basis`       — the approximation basis F (a `FunctionBasis`).
- `quad_basis`     — the quadrature basis G (a `FunctionBasis`).
- `interval::Tuple{T,T}` — the reference interval [a, b].
- `nn::Int`        — number of quadrature nodes.
- `nb::Int`        — number of approximation basis functions.
"""
struct FSBPOperator{T<:Real}
    D::Matrix{T}
    H::Diagonal{T,Vector{T}}
    Q::Matrix{T}
    S::Matrix{T}
    E::Matrix{T}
    tL::Vector{T}
    tR::Vector{T}
    x::Vector{T}
    w::Vector{T}
    op_basis
    quad_basis
    interval::Tuple{T,T}
    nn::Int
    nb::Int
end

function Base.show(io::IO, op::FSBPOperator{T}) where T
    println(io, "FSBPOperator{$T} on [$(op.interval[1]), $(op.interval[2])]")
    println(io, "  nodes            : $(op.nn)")
    println(io, "  basis funcs      : $(op.nb)")
    println(io, "  quad basis funcs : $(nbasis(op.quad_basis))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_fsbp_operator(op_basis, quad_basis;
                        orthogonalize=true,
                        use_optimization=false,
                        principal=:lower,
                        add_endpoint=nothing,
                        verbose=false) -> FSBPOperator

Construct a first-derivative FSBP operator from an approximation basis
`op_basis` (F) and a quadrature basis `quad_basis` (G).

Both arguments should be `FunctionBasis` objects (with derivatives supplied).
The quadrature basis G is typically the space (F²)' — the derivatives of
products of pairs of functions from F.

# Keyword arguments
- `orthogonalize::Bool=true` — orthogonalize both bases via
  `GeneralizedGauss.orthogonalize_basis` before computing the quadrature.
- `use_optimization::Bool=false` — if `true`, use the optimization-based
  construction for the case where the quadrature nodes and weights are known.
- `principal::Symbol=:lower` — which principal representation to use in
  the Generalized Gauss continuation algorithm.  `:lower` yields
  Gauss-Legendre-type (GL) rules; `:upper` yields Gauss-Lobatto-type
  (GLL) rules (for even-length quadrature bases).
- `add_endpoint::Union{Nothing,Symbol}=nothing` — which endpoint to anchor
  during continuation: `:left` or `:right`.  When `nothing`, the default
  pairing for the chosen `principal` is used (`:left` for `:lower`,
  `:right` for `:upper`).
- `verbose::Bool=false` — forward verbose diagnostic output to
  `GeneralizedGauss.compute_gauss_rule`.
- `extrapolation_norm::Symbol=:Hinv` — norm for the minimum-norm extrapolation
  operators `tL`, `tR` (`:Hinv`, `:H`, `:Euclidean`, or `:Frobenius`).  Used in
  both the exact and optimization-based construction paths.
- `rank_tol` — tolerance for rank decisions in Vandermonde checks and in
  optimization-based solves that use rank-truncated pseudoinverses.

# Quadrature rule types (for an even-length quadrature basis of 2n functions)
- `principal=:lower` → GL-type rule with **n** interior nodes (no endpoints).
- `principal=:upper` → GLL-type rule with **n+1** nodes including both endpoints.

For an odd-length basis of 2n+1 functions, both choices produce a Radau-type
rule with n+1 nodes including one endpoint.

To build a classical SBP operator on degree-p polynomials with p+1 GLL
nodes, pass `principal=:upper` and a quadrature basis spanning degrees
0 to 2p-1 (i.e., 2p functions).  Alternatively, keep `principal=:lower`
but extend the quadrature basis by 2 extra functions (degrees 0 to 2p+1,
i.e., 2(p+1) functions) to obtain a GL rule with p+1 interior nodes.

# Precision

The element type `T` is `eltype(op_basis)` (from the approximation-basis
interval).  The quadrature rule must return nodes and weights of the same
type.  Use matching `interval=(…)` on **both** `op_basis` and `quad_basis`
(e.g. all `Float64` or all `BigFloat`); mismatched basis interval types throw
an `ArgumentError`.

# Returns
An `FSBPOperator{T}` struct containing D, H, Q, S, E, tL, tR, x, w, and
references to the input bases.

# Construction procedure
1. Convert `quad_basis` to a GeneralizedGauss basis and (optionally)
   orthogonalize it.
2. Compute a generalized Gaussian quadrature rule (x, w) from the
   quadrature basis via `compute_gauss_rule`.
3. Build Vandermonde matrices V and Vₓ for `op_basis` at the nodes x, and
   verify `V` has full column rank `nb` (linearly independent basis at nodes).
4. Build extrapolation operators tL, tR (nodal or minimum-norm in `extrapolation_norm`)
   satisfying Vᵀ t ≈ v at boundaries:
   - If an endpoint is included in the quadrature nodes, use the corresponding
     nodal evaluation vector.
   - Otherwise if nn == nb: e = V⁻ᵀ t(xB).
   - Otherwise if nn > nb: use the minimum-norm solution.
5. Assemble the boundary matrix E = tR tRᵀ - tL tLᵀ.
6. Construct the differentiation matrix D and weak-derivative matrix Q:
   - If nn == nb: D = Vₓ / V  (unique solution), then verify Q + Qᵀ
     matches the extrapolation boundary matrix E.
   - If nn > nb: solve S V = H Vₓ - ½ E V for the minimum-norm
     skew-symmetric S, then Q = S + E/2.
"""
function build_fsbp_operator(op_basis, quad_basis;
                              orthogonalize::Bool = true,
                              use_optimization::Bool = false,
                              principal::Symbol = :lower,
                              add_endpoint::Union{Nothing,Symbol} = nothing,
                              test_functions = Function[],
                              test_derivatives = Function[],
                              test_weights = nothing,
                              extrapolation_objective_weights = (accuracy = 1//2, norm = 1//2),
                              S_objective_weights = (accuracy = 1//2, norm = 1//2),
                              extrapolation_norm::Symbol = :Hinv,
                              derivative_error_norm::Symbol = :H,
                              zero_boundary_scaling::Symbol = :fallback,
                              rank_tol = nothing,
                              compatibility_tol = nothing,
                              compatibility_action::Symbol = :warn,
                              extrapolation_scale_tol = nothing,
                              derivative_scale_tol = nothing,
                              objective_tol = nothing,
                              verbose::Bool = false)

    op_basis isa FunctionBasis && quad_basis isa FunctionBasis ||
        throw(ArgumentError("build_fsbp_operator requires FunctionBasis for op_basis and quad_basis."))
    _require_function_basis_intervals_match(op_basis, quad_basis, "build_fsbp_operator")

    # ── Extract interval ─────────────────────────────────────────────────
    interval = op_basis.interval
    a, b = interval
    T = eltype(op_basis)
    typeof(a) == T || throw(ArgumentError(
        "build_fsbp_operator: left endpoint type ($(typeof(a))) must match " *
        "op_basis interval type ($T)."))
    typeof(b) == T || throw(ArgumentError(
        "build_fsbp_operator: right endpoint type ($(typeof(b))) must match " *
        "op_basis interval type ($T)."))
    _validate_norm_symbol(extrapolation_norm, (:Hinv, :H, :Euclidean, :Frobenius),
                          "extrapolation_norm")

    # ── Step 1: Build GeneralizedGauss basis for the quadrature basis ────
    gg_quad_basis = _to_gg_basis(quad_basis; require_derivs=true)

    if orthogonalize
        gg_quad_basis, _ = GeneralizedGauss.orthogonalize_basis(gg_quad_basis)
    end

    # ── Step 2: Compute quadrature rule ──────────────────────────────────
    # Pass principal and add_endpoint to control the rule type:
    #   principal=:upper → GLL (both endpoints, n+1 nodes for 2n basis)
    #   principal=:lower → GL  (no endpoints, n nodes for 2n basis)
    #   principal=:right → Right-Radau (right endpoint, n+1 nodes for 2n+1 basis)
    #   principal=:left → Left-Radau (right endpoint, n+1 nodes for 2n+1 basis)
    w, x = if add_endpoint === nothing
        GeneralizedGauss.compute_gauss_rule(gg_quad_basis; principal, verbose)
    else
        GeneralizedGauss.compute_gauss_rule(gg_quad_basis;
                                            principal, verbose, add_endpoint)
    end
    x = collect(x)
    w = collect(w)
    _require_uniform_type("build_fsbp_operator quadrature", [
        T, _array_element_type(x, "quadrature nodes"), _array_element_type(w, "quadrature weights")])
    nn = length(x)
    nb = nbasis(op_basis)

    if nn < nb
        error("Number of quadrature nodes ($nn) is less than the number of " *
              "basis functions ($nb). Either use `principal=:upper` to get a " *
              "GLL-type rule with more nodes, or extend the quadrature basis " *
              "with additional functions.")
    end

    # Validate positive weights
    if any(w .<= 0)
        println(stderr, "GaussFSBP warning: Quadrature has non-positive weights. min(w) = $(minimum(w))")
    end

    if use_optimization
        # ── Steps 3-6: Construct D, Q, S, E, tL, tR
        return optimize_fsbp_operator(x, w, a, b, op_basis;
                                      test_functions = test_functions,
                                      test_derivatives = test_derivatives,
                                      test_weights = test_weights,
                                      extrapolation_objective_weights = extrapolation_objective_weights,
                                      S_objective_weights = S_objective_weights,
                                      extrapolation_norm = extrapolation_norm,
                                      derivative_error_norm = derivative_error_norm,
                                      zero_boundary_scaling = zero_boundary_scaling,
                                      rank_tol = rank_tol,
                                      compatibility_tol = compatibility_tol,
                                      compatibility_action = compatibility_action,
                                      extrapolation_scale_tol = extrapolation_scale_tol,
                                      derivative_scale_tol = derivative_scale_tol,
                                      objective_tol = objective_tol,
                                      verbose = verbose,
                                      quad_basis = quad_basis)
    end

    # -- use_optimization = false case ──────────────────────────────────────

    # ── Step 3: Build Vandermonde matrices ───────────────────────────────
    V  = eval_basis_matrix(op_basis, x)             # nn × nb
    Vx = eval_basis_derivative_matrix(op_basis, x)   # nn × nb
    _check_vandermonde_rank(V, nb, T; rank_tol,
                                      context = "build_fsbp_operator")

    # ── Step 4: Save quadrature H ────────────────────────────────────────
    H = Diagonal(w)

    # ── Step 5: Build extrapolation operators ────────────────────────────
    vL = eval_basis_vector(op_basis, a)
    vR = eval_basis_vector(op_basis, b)
    tL, tR = _build_extrapolation(V, x, w, vL, vR, a, b, extrapolation_norm)

    # ── Step 6: Boundary matrix E ────────────────────────────────────────
    E = tR * tR' - tL * tL'

    # ── Step 7: Construct D, Q, and S using the extrapolation boundary E ──
    if nn == nb
        D, Q, S = _build_operator_square(V, Vx, H, E, nn)
    else
        D, Q, S = _build_operator_rectangular(V, Vx, H, E, nn, nb)
    end

    return FSBPOperator{T}(D, H, Q, S, E, tL, tR, x, w,
                           op_basis, quad_basis, (a, b), nn, nb)
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Square case (nn == nb) — unique operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_operator_square(V, Vx, H, E, nn) -> (D, Q, S)

Construct the FSBP operator when the number of nodes equals the number
of basis functions (nn == nb). The derivative operator is unique:
D = Vₓ V⁻¹ and Q = H D.  The resulting boundary matrix Q + Qᵀ is checked
against the extrapolation boundary matrix E, then S = Q - E/2.
"""
function _build_operator_square(V, Vx, H, E, nn)
    T = eltype(V)

    # D = Vx / V  (i.e., D * V = Vx  =>  D = Vx V⁻¹)
    D = Vx / V

    # Q = H * D
    Q = Matrix(H) * D

    # The unique square construction determines Q + Qᵀ. It must agree
    # with the boundary matrix induced by the extrapolation operators.
    E_from_Q = Q + Q'
    _check_boundary_matrix_match(E_from_Q, E)

    # S = Q - E/2  (skew-symmetric part)
    S = Q - E / T(2)

    return D, Q, S
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Rectangular case (nn > nb) — least-squares
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_operator_rectangular(V, Vx, H, E, nn, nb) -> (D, Q, S)

Construct the FSBP operator when nn > nb. The operator is not unique;
we find the minimum-norm skew-symmetric S satisfying the accuracy
conditions S V = H Vₓ - ½ E V, then set Q = S + E/2. Here E is the
boundary matrix already constructed from tR tRᵀ - tL tLᵀ.
"""
function _build_operator_rectangular(V, Vx, H, E, nn, nb)
    T = eltype(V)
    w = diag(H)

    # Right-hand side: RHS = H * Vx - (1/2) * E * V   (nn × nb)
    RHS = Vx .* reshape(w, :, 1) - E * V / T(2)

    # We need to find skew-symmetric S (nn × nn) such that S * V = RHS.
    # S is skew-symmetric: S[i,j] = -S[j,i], S[i,i] = 0.
    # The independent entries are the strictly lower-triangular part:
    #   q_k for k = 1, ..., L  where L = nn*(nn-1)/2.
    #
    # Vectorize: build a matrix M such that M * q = vec(RHS),
    # where q contains the L independent entries of S.

    L = nn * (nn - 1) ÷ 2   # number of independent entries in skew-symmetric matrix

    # Build the linear system: for each pair (i,j) with i<j, the entry S[i,j]
    # appears in row i of S * V with coefficient V[j, :] and in row j with
    # coefficient -V[i, :] (due to skew-symmetry).

    # System has nn*nb equations and L unknowns
    n_eq = nn * nb
    M = zeros(T, n_eq, L)
    rhs_vec = zeros(T, n_eq)

    # Map (i,j) with i < j to a linear index
    idx = 0
    pair_map = Dict{Tuple{Int,Int},Int}()
    for j in 2:nn
        for i in 1:(j-1)
            idx += 1
            pair_map[(i,j)] = idx
        end
    end

    # Fill the system:
    # Equation for (S * V)[row, col] = RHS[row, col]
    # (S * V)[row, col] = Σ_k S[row, k] * V[k, col]
    # For each k ≠ row:
    #   if row < k: S[row,k] = +q_{pair_map[(row,k)]}
    #   if row > k: S[row,k] = -q_{pair_map[(k,row)]}

    for col in 1:nb
        for row in 1:nn
            eq_idx = (col - 1) * nn + row
            rhs_vec[eq_idx] = RHS[row, col]

            for k in 1:nn
                k == row && continue
                if row < k
                    pidx = pair_map[(row, k)]
                    M[eq_idx, pidx] += V[k, col]
                else
                    pidx = pair_map[(k, row)]
                    M[eq_idx, pidx] -= V[k, col]
                end
            end
        end
    end

    # Solve least-squares for minimum-norm q
    q = M \ rhs_vec

    # Reconstruct S from q
    S = zeros(T, nn, nn)
    for ((i, j), pidx) in pair_map
        S[i, j] =  q[pidx]
        S[j, i] = -q[pidx]
    end

    # Q = S + E/2
    Q = S + E / T(2)

    # D = H⁻¹ Q
    D = Q ./ reshape(w, :, 1)

    return D, Q, S
end

function _check_boundary_matrix_match(E_from_Q, E)
    eltype(E_from_Q) == eltype(E) || throw(ArgumentError(
        "Boundary matrix type mismatch: $(eltype(E_from_Q)) vs $(eltype(E))."))
    T = eltype(E)
    residual = maximum(abs.(E_from_Q - E))
    scale = max(one(T), maximum(abs.(E_from_Q)), maximum(abs.(E)))
    tol = T(100) * sqrt(eps(T)) * scale
    if residual > tol
        error("Boundary matrix from the unique square operator does not match " *
              "the extrapolation boundary matrix: residual $residual exceeds " *
              "tolerance $tol.")
    end
    return residual
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Boundary extrapolation operators
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_extrapolation(V, x, w, vL, vR, xL, xR, extrapolation_norm) -> (tL, tR)

Build boundary extrapolation operators `tL`, `tR` (minimum norm solutions).

- If a boundary point is a quadrature node, use the nodal evaluation vector.
- Otherwise use the minimum-norm solution of `V' t = v` in the norm given by
  `extrapolation_norm` (`:Hinv`, `:H`, `:Euclidean`, or `:Frobenius`).
  Requires full column rank of `V` (checked before this is called).
"""
function _build_extrapolation(V, x, w, vL, vR, xL, xR, extrapolation_norm::Symbol)
    T = eltype(V)
    nn = size(V, 1)
    left_idx = _endpoint_node_index(x, xL)
    right_idx = _endpoint_node_index(x, xR)

    tL = if left_idx !== nothing
        _nodal_evaluation_vector(T, nn, left_idx)
    else
        _minimum_extrapolation_solution(V, w, vL, extrapolation_norm)
    end
    tR = if right_idx !== nothing
        _nodal_evaluation_vector(T, nn, right_idx)
    else
        _minimum_extrapolation_solution(V, w, vR, extrapolation_norm)
    end

    return tL, tR
end

function _endpoint_node_index(x::AbstractVector{T}, endpoint::T) where T
    _, idx = findmin(abs.(x .- endpoint))
    tol = T(100) * eps(T) * max(one(T), abs(endpoint))
    return abs(x[idx] - endpoint) <= tol ? idx : nothing
end

function _nodal_evaluation_vector(::Type{T}, nn::Int, idx) where T
    idx === nothing && return nothing
    t = zeros(T, nn)
    t[idx] = one(T)
    return t
end
