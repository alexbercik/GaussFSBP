"""
    OperatorBuilders.jl

Construction routines for function-space SBP (FSBP) operators.

Given an approximation basis F and a quadrature basis G, this module:
1. Optionally orthogonalizes both bases via GeneralizedGauss.
2. Constructs a quadrature rule from G via GeneralizedGauss.
3. Builds the first-derivative FSBP operator D = H⁻¹Q.
4. Constructs boundary extrapolation operators tL, tR.
5. Packages everything into an `FSBPOperator` struct.

Two construction paths are supported:
- **nn == nb** (number of nodes equals number of basis functions):
  the derivative operator is unique and computed via Vandermonde inversion.
- **nn > nb** (more nodes than basis functions):
  the system is underdetermined; a least-squares (minimum-norm) solution
  is used by default.  An optimization-based path (Glaubitz et al. 2025)
  can be selected via `use_optimization=true` (currently a stub).
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
- `S::Matrix{T}`  — skew-symmetric part, S = Q - B/2.
- `E::Matrix{T}`  — boundary matrix, E = tR tRᵀ - tL tLᵀ.
- `B::Matrix{T}`  — SBP boundary matrix, B = Q + Qᵀ.
- `tL::Vector{T}` — left boundary extrapolation operator.
- `tR::Vector{T}` — right boundary extrapolation operator.
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
    B::Matrix{T}
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
    println(io, "  nodes (nn)        : $(op.nn)")
    println(io, "  basis funcs (nb)  : $(op.nb)")
    case = op.nn == op.nb ? "unique (nn == nb)" : "least-squares (nn > nb)"
    println(io, "  construction      : $case")
    println(io, "  min weight        : $(minimum(op.w))")
    println(io, "  max |S + Sᵀ|      : $(maximum(abs.(op.S + op.S')))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: interval scalar type (for precision mismatch warnings)
# ─────────────────────────────────────────────────────────────────────────────

_stderr_warn(msg::AbstractString) = println(stderr, "GaussFSBP warning: ", msg)

function _basis_scalar_interval_type(basis::AbstractBasis)
    hasfield(typeof(basis), :interval) || return nothing
    a, b = basis.interval
    return promote_type(typeof(a), typeof(b))
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
  construction (Glaubitz et al. 2025) that simultaneously determines P and Q.
  **Currently not implemented** — raises an error if `true`.
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

The element type `T` of matrices and vectors is taken from the quadrature
nodes and weights returned by GeneralizedGauss (see `promote_type` of
`eltype(x)`, `eltype(w)`).  That follows the interval endpoints passed into
the quadrature basis: `Float64` endpoints yield a `Float64` rule; `BigFloat`
endpoints yield a `BigFloat` rule.  Use `interval=(BigFloat(-1), BigFloat(1))`
on **both** `op_basis` and `quad_basis` when you want a fully `BigFloat`
operator (the stored `interval` tuple comes from `op_basis`).

If the interval scalar types of `op_basis` and `quad_basis` disagree (e.g.
`Float64` vs `BigFloat`), a warning is emitted but construction continues.

# Returns
An `FSBPOperator{T}` struct containing D, H, Q, S, E, B, tL, tR, x, w, and
references to the input bases.

# Construction procedure
1. Convert `quad_basis` to a GeneralizedGauss basis and (optionally)
   orthogonalize it.
2. Compute a generalized Gaussian quadrature rule (x, w) from the
   quadrature basis via `compute_gauss_rule`.
3. Build Vandermonde matrices V and Vₓ for `op_basis` at the nodes x.
4. Construct the differentiation matrix D and weak-derivative matrix Q:
   - If nn == nb: D = Vₓ / V  (unique solution).
   - If nn > nb: solve Q_A V = P Vₓ - ½ B V for the skew-symmetric part
     Q_A in the least-squares sense, then Q = Q_A + B/2.
5. Build boundary extrapolation operators tL, tR:
   - If an endpoint is included in the quadrature nodes, use the corresponding
     nodal evaluation vector.
   - Otherwise if nn == nb: t = V⁻ᵀ v(xB).
   - Otherwise if nn > nb: use the minimum-norm solution.
6. Assemble the boundary matrix E = tR tRᵀ - tL tLᵀ.
"""
function build_fsbp_operator(op_basis, quad_basis;
                              orthogonalize::Bool = true,
                              use_optimization::Bool = false,
                              principal::Symbol = :lower,
                              add_endpoint::Union{Nothing,Symbol} = nothing,
                              verbose::Bool = false)

    if use_optimization
        error("Optimization-based FSBP construction is not yet implemented. " *
              "Set `use_optimization=false` to use the classical construction " *
              "with least-squares (Glaubitz et al. 2023).")
    end

    Ta = _basis_scalar_interval_type(op_basis)
    Tb = _basis_scalar_interval_type(quad_basis)
    if Ta !== nothing && Tb !== nothing && Ta != Tb
        _stderr_warn(
            "op_basis interval scalar type ($Ta) differs from quad_basis ($Tb); " *
            "quadrature arithmetic follows quad_basis while FSBPOperator.interval " *
            "follows op_basis. Consider using the same scalar type on both bases.")
    end

    # ── Extract interval ─────────────────────────────────────────────────
    interval = op_basis.interval
    a, b = interval

    # ── Step 1: Build GeneralizedGauss basis for the quadrature basis ────
    gg_quad_basis = _to_gg_basis(quad_basis; require_derivs=true)

    if orthogonalize
        gg_quad_basis, _ = GeneralizedGauss.orthogonalize_basis(gg_quad_basis)
    end

    # ── Step 2: Compute quadrature rule ──────────────────────────────────
    # Pass principal and add_endpoint to control the rule type:
    #   principal=:upper → GLL (both endpoints, n+1 nodes for 2n basis)
    #   principal=:lower → GL  (no endpoints, n nodes for 2n basis)
    gauss_kwargs = Dict{Symbol,Any}(:principal => principal,
                                    :verbose => verbose)
    if add_endpoint !== nothing
        gauss_kwargs[:add_endpoint] = add_endpoint
    end
    w_raw, x_raw = GeneralizedGauss.compute_gauss_rule(gg_quad_basis;
                                                        gauss_kwargs...)

    # Determine the working precision from the quadrature output
    T = promote_type(eltype(x_raw), eltype(w_raw))
    x = Vector{T}(x_raw)
    w = Vector{T}(w_raw)
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
        _stderr_warn("Quadrature has non-positive weights. min(w) = $(minimum(w))")
    end

    # ── Step 3: Build Vandermonde matrices ───────────────────────────────
    V  = eval_basis_matrix(op_basis, x)             # nn × nb
    Vx = eval_basis_derivative_matrix(op_basis, x)   # nn × nb

    # Promote Vandermonde matrices to working precision if needed
    VT  = Matrix{T}(V)
    VxT = Matrix{T}(Vx)

    # ── Step 4: Construct D and Q ────────────────────────────────────────
    H = Diagonal(w)

    if nn == nb
        D, Q, S, B_mat = _build_operator_square(VT, VxT, H, nn)
    else
        D, Q, S, B_mat = _build_operator_rectangular(VT, VxT, H, nn, nb)
    end

    # ── Step 5: Build extrapolation operators ────────────────────────────
    tL, tR = _build_extrapolation(VT, x, op_basis, T(a), T(b), nn, nb)

    # ── Step 6: Boundary matrix E ────────────────────────────────────────
    E = tR * tR' - tL * tL'

    return FSBPOperator{T}(D, H, Q, S, E, B_mat, tL, tR, x, w,
                           op_basis, quad_basis, (T(a), T(b)), nn, nb)
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Square case (nn == nb) — unique operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_operator_square(V, Vx, H, nn) -> (D, Q, S, B)

Construct the FSBP operator when the number of nodes equals the number
of basis functions (nn == nb). The derivative operator is unique:
D = Vₓ V⁻¹, Q = H D, S = Q - B/2.
"""
function _build_operator_square(V, Vx, H, nn)
    # D = Vx / V  (i.e., D * V = Vx  =>  D = Vx V⁻¹)
    D = Vx / V

    # Q = H * D
    Q = Matrix(H) * D

    # B = Q + Qᵀ  (should be diag(-1, 0, ..., 0, 1) for boundary nodes)
    B_mat = Q + Q'

    # S = Q - B/2  (skew-symmetric part)
    S = Q - B_mat / 2

    return D, Q, S, B_mat
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Rectangular case (nn > nb) — least-squares
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_operator_rectangular(V, Vx, H, nn, nb) -> (D, Q, S, B)

Construct the FSBP operator when nn > nb. The operator is not unique;
we find the minimum-norm skew-symmetric Q_A satisfying the accuracy
conditions Q_A V = P Vₓ - ½ B V, then set Q = Q_A + B/2.

The boundary matrix B = diag(-1, 0, ..., 0, 1) assumes the first and
last nodes are at the interval endpoints.
"""
function _build_operator_rectangular(V, Vx, H, nn, nb)
    T = eltype(V)

    # Build B = diag(-1, 0, ..., 0, 1)
    B_vec = zeros(T, nn)
    B_vec[1]  = -one(T)
    B_vec[nn] =  one(T)
    B_mat = Diagonal(B_vec)

    # Right-hand side: RHS = H * Vx - 0.5 * B * V   (nn × nb)
    RHS = Matrix(H) * Vx - Matrix(B_mat) * V / 2

    # We need to find skew-symmetric Q_A (nn × nn) such that Q_A * V = RHS.
    # Q_A is skew-symmetric: Q_A[i,j] = -Q_A[j,i], Q_A[i,i] = 0.
    # The independent entries are the strictly lower-triangular part:
    #   q_k for k = 1, ..., L  where L = nn*(nn-1)/2.
    #
    # Vectorize: build a matrix M such that M * q = vec(RHS),
    # where q contains the L independent entries of Q_A.

    L = nn * (nn - 1) ÷ 2   # number of independent entries in skew-symmetric matrix

    # Build the linear system: for each pair (i,j) with i<j, the entry Q_A[i,j]
    # appears in row i of Q_A * V with coefficient V[j, :] and in row j with
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
    # Equation for (Q_A * V)[row, col] = RHS[row, col]
    # (Q_A * V)[row, col] = Σ_k Q_A[row, k] * V[k, col]
    # For each k ≠ row:
    #   if row < k: Q_A[row,k] = +q_{pair_map[(row,k)]}
    #   if row > k: Q_A[row,k] = -q_{pair_map[(k,row)]}

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

    # Reconstruct Q_A from q
    Q_A = zeros(T, nn, nn)
    for ((i, j), pidx) in pair_map
        Q_A[i, j] =  q[pidx]
        Q_A[j, i] = -q[pidx]
    end

    # Q = Q_A + B/2
    Q = Q_A + Matrix(B_mat) / 2

    # D = H⁻¹ Q
    D = Diagonal(one(T) ./ diag(H)) * Q

    # S = Q_A  (the skew-symmetric part)
    S = Q_A

    return D, Q, S, Matrix(B_mat)
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: Boundary extrapolation operators
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_extrapolation(V, x, op_basis, a, b, nn, nb) -> (tL, tR)

Build boundary extrapolation operators tL and tR.

- If a boundary point is present in the node set, the corresponding
  extrapolation operator is the canonical nodal evaluation vector.
- If nn == nb: tL = V⁻ᵀ v(a), tR = V⁻ᵀ v(b) where v(x) is the vector
  of basis functions evaluated at x.
- If nn > nb: least-squares solution minimizing ‖tL‖ subject to Vᵀ tL = v(a),
  and similarly for tR.
"""
function _build_extrapolation(V, x, op_basis, a, b, nn, nb)
    T = eltype(V)

    # Evaluate basis at boundary points
    vL = T.(eval_basis(op_basis, a))   # length nb
    vR = T.(eval_basis(op_basis, b))   # length nb

    left_endpoint_idx = _endpoint_node_index(x, a)
    right_endpoint_idx = _endpoint_node_index(x, b)

    tL = _nodal_evaluation_vector(T, nn, left_endpoint_idx)
    tR = _nodal_evaluation_vector(T, nn, right_endpoint_idx)

    if nn == nb
        # tL = V⁻ᵀ * vL  (solve Vᵀ tL = vL)
        tL === nothing && (tL = V' \ vL)
        tR === nothing && (tR = V' \ vR)
    else
        # Least-squares: min ‖t‖ subject to Vᵀ t = v
        # This is the minimum-norm solution: t = V (VᵀV)⁻¹ v
        VtV = V' * V   # nb × nb
        tL === nothing && (tL = V * (VtV \ vL))
        tR === nothing && (tR = V * (VtV \ vR))
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
