"""
    OperatorBuilders.jl

High-level construction routines for one-dimensional diagonal-norm
function-space SBP (FSBP) operators.

The public entry point is [`build_fsbp_operator`](@ref).  It takes an
approximation basis `op_basis` and a quadrature basis `quad_basis`, optionally
orthogonalizes the quadrature basis, computes a GeneralizedGauss rule, and
assembles the first-derivative operator `D = H⁻¹Q` together with the boundary
extrapolation vectors `tL`, `tR`.

Two construction modes are available:
- The default mode, `use_optimization=false`, builds exact or minimum-norm
  extrapolation operators and then constructs the compatible SBP operator
  directly.  If `nn == nb`, the derivative operator is unique; if `nn > nb`,
  the skew-symmetric part is chosen by a minimum-norm solve.
- The optimized mode, `use_optimization=true`, computes the same quadrature
  rule but delegates the operator construction to [`optimize_fsbp_operator`](@ref).
  Optimization-specific keywords are collected as `opt_kwargs...` by
  `build_fsbp_operator` and forwarded to the optimized builder.
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
                        extrapolation_norm=:Hinv,
                        rank_tol=nothing,
                        quad_moments=nothing,
                        quad_kwargs=NamedTuple(),
                        verbose=false,
                        opt_kwargs...) -> FSBPOperator

Construct a first-derivative FSBP operator from an approximation basis
`op_basis` and a quadrature basis `quad_basis`.

Both arguments must be `FunctionBasis` objects with matching interval element
types.  The quadrature basis is typically the space `(F²)'`, the derivatives of
products of pairs of approximation functions.

# Keyword Arguments
- `orthogonalize::Bool=true` — orthogonalize the GeneralizedGauss quadrature
  basis before computing the quadrature rule.
- `use_optimization::Bool=false` — when `false`, use the direct exact/minimum-norm
  construction in this file.  When `true`, compute the quadrature rule here and
  forward the known nodes and weights to [`optimize_fsbp_operator`](@ref).
- `principal::Symbol=:lower` — GeneralizedGauss principal representation.
  For an even-length quadrature basis, `:lower` gives a GL-type rule and
  `:upper` gives a GLL-type rule.
- `extrapolation_norm::Symbol=:Hinv` — the weighted norm to use for tL, tR.
  Allowed values are `:Hinv`, `:H`, `:Euclidean`.
- `rank_tol=nothing` — tolerance for Vandermonde rank checks and, in the
  optimized path, rank-truncated pseudoinverse/nullspace computations.
- `sbp_check_action::Symbol=:error` — action when a construction-time SBP check
  fails in the direct path: `:error` (default), `:warn`, or `:ignore`.  When
  `use_optimization=true`, this keyword is forwarded to
  [`optimize_fsbp_operator`](@ref) together with `opt_kwargs...`.
- `quad_moments=nothing` — optional exact moments of the original `quad_basis`
  in its original order.  When supplied, these moments are passed to
  `GeneralizedGauss.compute_gauss_rule` instead of being computed numerically.
  If `orthogonalize=true`, the builder applies the same change of basis to the
  moments before solving the quadrature rule.
- `quad_kwargs=NamedTuple()` — additional keywords forwarded to
  `GeneralizedGauss.compute_gauss_rule`, such as `lost_digits`,
  `add_endpoint`, `solver_tolerance`, `intermediate_tolerance`,
  `differentiable`, `measure`, and nonlinear solver options.  When
  `orthogonalize=true`, `measure` is also used for the GeneralizedGauss
  orthogonalization.  `principal` remains a top-level keyword.  An optional
  `verbose` entry in `quad_kwargs` overrides quadrature diagnostics only;
  when omitted, quadrature verbosity follows the top-level `verbose` flag.
- `verbose::Bool=false` — print FSBP construction diagnostics.  The direct
  path reports its selected construction and exactness residuals; the
  optimized path additionally reports optimization diagnostics.  Quadrature
  rule generation uses the same flag unless overridden by
  `quad_kwargs=(verbose=...,)`.
- `opt_kwargs...` — additional optimization keywords forwarded unchanged to
  [`optimize_fsbp_operator`](@ref) when `use_optimization=true` and ignored by
  the direct construction path when `use_optimization=false` (the quadrature
  basis is passed separately as the required `quad_basis` argument).  This is
  where optimization controls such as
  `test_functions`, `test_derivatives`, `test_weights`,
  `extrapolation_objective_weights`, `S_objective_weights`,
  `derivative_error_norm`, `zero_boundary_scaling`, `extrapolation_symmetry`,
  `sbp_check_action`, objective tolerances, `opt_method`, and simultaneous
  solver/search options should be supplied.

# Quadrature Rules
For an even-length quadrature basis of `2n` functions, `principal=:lower`
produces a GL-type rule with `n` interior nodes, while `principal=:upper`
produces a GLL-type rule with `n+1` nodes including both endpoints.  For an
odd-length basis of `2n+1` functions, the rule is Radau-type and includes one
endpoint.

# Returns
An `FSBPOperator{T}` containing `D`, `H`, `Q`, `S`, `E`, `tL`, `tR`, the nodes
and weights, and references to the input bases.

# Direct Construction
The non-optimized path builds `V`, `Vx`, `vL`, and `vR`; constructs nodal or
minimum-norm extrapolation vectors satisfying `V' * tB = vB`; forms
`E = tR*tR' - tL*tL'`; and then constructs `Q = S + E/2`.  If `nn == nb`, the
derivative operator is unique.  If `nn > nb`, the skew-symmetric matrix `S` is
chosen by the minimum-norm reduced solve.
"""
function build_fsbp_operator(op_basis, quad_basis;
                              orthogonalize::Bool = true,
                              use_optimization::Bool = false,
                              principal::Symbol = :lower,
                              extrapolation_norm::Symbol = :Hinv,
                              rank_tol = nothing,
                              sbp_check_action::Symbol = :error,
                              quad_moments = nothing,
                              quad_kwargs::NamedTuple = NamedTuple(),
                              verbose::Bool = false,
                              opt_kwargs...)

    op_basis isa FunctionBasis && quad_basis isa FunctionBasis ||
        throw(ArgumentError("build_fsbp_operator requires FunctionBasis for op_basis and quad_basis."))
    if :add_endpoint in keys(opt_kwargs)
        throw(ArgumentError(
            "add_endpoint is a quadrature keyword. Pass it as " *
            "quad_kwargs=(add_endpoint=...,), not as a top-level keyword."))
    end
    _require_function_basis_intervals_match(op_basis, quad_basis, "build_fsbp_operator")
    _validate_sbp_check_action(sbp_check_action)

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
    reserved_quad_keys = (:principal, :moments)
    conflicting_quad_keys = [key for key in keys(quad_kwargs) if key in reserved_quad_keys]
    if !isempty(conflicting_quad_keys)
        names = join(string.(conflicting_quad_keys), ", ")
        throw(ArgumentError(
            "quad_kwargs must not contain top-level quadrature keyword(s): $names. " *
            "Pass explicit moments with the top-level quad_moments keyword."))
    end
    quad_verbose = get(quad_kwargs, :verbose, verbose)
    remaining_quad_kwargs = NamedTuple(
        key => quad_kwargs[key] for key in keys(quad_kwargs) if key != :verbose)

    # ── Step 1: Build GeneralizedGauss basis for the quadrature basis ────
    # GeneralizedGauss can use analytic derivatives, finite differences, or
    # derivative-free MADS depending on its `differentiable` keyword, so the
    # quadrature basis itself does not need analytic derivatives here.
    gg_quad_basis = _to_gg_basis(quad_basis)
    gg_moments = nothing
    if quad_moments !== nothing
        raw_quad_moments = collect(quad_moments)
        raw_quad_moments isa AbstractVector || throw(ArgumentError(
            "quad_moments must be a vector-like collection with one moment " *
            "per quadrature basis function."))
        length(raw_quad_moments) == nbasis(quad_basis) || throw(ArgumentError(
            "quad_moments has length $(length(raw_quad_moments)), expected " *
            "$(nbasis(quad_basis)) for the supplied quad_basis."))
        gg_moments = T.(raw_quad_moments)
    end

    if orthogonalize
        # Weighted quadrature should orthogonalize with the same measure used
        # later for moment computation.
        orth_measure = haskey(quad_kwargs, :measure) ? quad_kwargs.measure : nothing
        gg_quad_basis, moment_transform =
            GeneralizedGauss.orthogonalize_basis(gg_quad_basis;
                                                 measure=orth_measure)
        if gg_moments !== nothing
            # orthogonalize_basis returns ψ = T φ, so the moment vector must
            # be transformed by the same matrix before the quadrature solve.
            gg_moments = Vector{T}(Matrix{T}(moment_transform) * gg_moments)
        end
    end

    # ── Step 2: Compute quadrature rule ──────────────────────────────────
    # Pass principal and optional quad_kwargs.add_endpoint to control the rule type:
    #   principal=:upper → GLL (both endpoints, n+1 nodes for 2n basis)
    #   principal=:lower → GL  (no endpoints, n nodes for 2n basis)
    #   principal=:right → Right-Radau (right endpoint, n+1 nodes for 2n+1 basis)
    #   principal=:left → Left-Radau (right endpoint, n+1 nodes for 2n+1 basis)
    if gg_moments === nothing
        w, x = GeneralizedGauss.compute_gauss_rule(gg_quad_basis;
                                                   principal,
                                                   verbose = quad_verbose,
                                                   remaining_quad_kwargs...)
    else
        w, x = GeneralizedGauss.compute_gauss_rule(gg_quad_basis, gg_moments;
                                                     principal,
                                                     verbose = quad_verbose,
                                                     remaining_quad_kwargs...)
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
        return optimize_fsbp_operator(x, w, a, b, op_basis, quad_basis;
                                      extrapolation_norm = extrapolation_norm,
                                      rank_tol = rank_tol,
                                      verbose = verbose,
                                      sbp_check_action = sbp_check_action,
                                      opt_kwargs...)
    end

    # -- use_optimization = false case ──────────────────────────────────────

    # ── Step 3: Build Vandermonde matrices ───────────────────────────────
    V  = eval_basis_matrix(op_basis, x)             # nn × nb
    Vx = eval_basis_derivative_matrix(op_basis, x)   # nn × nb
    rankV = _check_vandermonde_rank(V, nb, T; rank_tol,
                                              context = "build_fsbp_operator")

    # ── Step 4: Save quadrature H ────────────────────────────────────────
    H = Diagonal(w)

    # ── Step 5: Build extrapolation operators ────────────────────────────
    vL = eval_basis_vector(op_basis, a)
    vR = eval_basis_vector(op_basis, b)
    _check_sbp_compatibility(V, Vx, w, vL, vR, T; action = sbp_check_action)
    tL, tR = _build_extrapolation(V, x, w, vL, vR, a, b, extrapolation_norm)

    # ── Step 6: Boundary matrix E ────────────────────────────────────────
    E = tR * tR' - tL * tL'

    # ── Step 7: Construct D, Q, and S using the extrapolation boundary E ──
    if nn == nb
        D, Q, S = _build_operator_square(V, Vx, H, E, nn; action = sbp_check_action)
    else
        D, Q, S = _build_operator_rectangular(V, Vx, H, E, nn, nb)
    end

    if verbose
        # Report only inexpensive diagnostics already available from the
        # construction.  Full quadrature verification remains opt-in through
        # check_fsbp_operator because it may require reference integrations.
        left_endpoint_idx = _endpoint_node_index(x, a)
        right_endpoint_idx = _endpoint_node_index(x, b)
        construction = nn == nb ? "square (unique)" :
                                  "rectangular (minimum-norm)"
        left_extrapolation = left_endpoint_idx === nothing ?
            "minimum-norm ($extrapolation_norm)" :
            "nodal (node $left_endpoint_idx)"
        right_extrapolation = right_endpoint_idx === nothing ?
            "minimum-norm ($extrapolation_norm)" :
            "nodal (node $right_endpoint_idx)"

        # Compatibility depends only on the quadrature and approximation
        # basis, whereas the remaining residuals verify the constructed
        # extrapolation and operator matrices.
        HVx = Vx .* reshape(w, :, 1)
        compatibility_residual = norm(
            V' * HVx + HVx' * V - (vR * vR' - vL * vL'))
        left_extrapolation_residual = norm(V' * tL - vL)
        right_extrapolation_residual = norm(V' * tR - vR)
        derivative_residual = norm(D * V - Vx)
        sbp_residual = norm(Q + Q' - E)
        skew_residual = norm(S + S')

        println("\nDirect FSBP construction")
        println("  num of nodes = $nn")
        println("  dim of basis = $nb")
        println("  rank(V) = $rankV")
        println("  construction = $construction")
        println("  left extrapolation = $left_extrapolation")
        println("  right extrapolation = $right_extrapolation")
        println("  extrapolation norm = $extrapolation_norm")
        println("Quadrature/SBP compatibility residual = $compatibility_residual")
        println("After direct construction:")
        println("  ||V^T tL - v_L|| = $left_extrapolation_residual")
        println("  ||V^T tR - v_R|| = $right_extrapolation_residual")
        println("  ||D V - V_x|| = $derivative_residual")
        println("  ||Q + Q^T - E|| = $sbp_residual")
        println("  ||S + S^T|| = $skew_residual")
        println("\n")
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
function _build_operator_square(V, Vx, H, E, nn; action::Symbol = :error)
    T = eltype(V)

    # D = Vx / V  (i.e., D * V = Vx  =>  D = Vx V⁻¹)
    D = Vx / V

    # Q = H * D
    Q = Matrix(H) * D

    # The unique square construction determines Q + Qᵀ. It must agree
    # with the boundary matrix induced by the extrapolation operators.
    E_from_Q = Q + Q'
    _check_boundary_matrix_match(E_from_Q, E; action = action)

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
