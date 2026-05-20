"""
    OptimizedOperatorBuilders.jl

Optimization-based construction of one-dimensional diagonal-norm FSBP
operators when the quadrature nodes and weights are already known.
"""

"""
    optimize_fsbp_operator(x, w, xL, xR, basis; kwargs...) -> FSBPOperator
    optimize_fsbp_operator(x, w, xL, xR, basis_functions, basis_derivatives; kwargs...) -> FSBPOperator

Construct a 1-D diagonal-norm FSBP operator for known nodes `x`, quadrature
weights `w`, and endpoints `xL`, `xR`.  Returns an [`FSBPOperator`](@ref) using
the supplied basis as both the approximation and quadrature basis metadata.

The basis (exactness space) is supplied either as an `AbstractBasis` with derivatives
or as paired vectors of basis functions and derivatives.  The construction
first optimizes exact boundary extrapolation vectors, then constructs and
optionally optimizes the skew-symmetric matrix using the independent-equation
system of Marchildon and Zingg.

# Implementation layout

Both public methods share the same pipeline:

1. [`_optimize_fsbp_preamble`](@ref) — validate inputs and fix the working type `T`
2. Build Vandermonde data (`V`, `Vx`, `vL`, `vR`) — API differs by input type
3. [`_optimize_fsbp_operator_core`](@ref) — optimization math (single copy)

The split avoids duplicating the large core while letting `AbstractBasis` use
[`eval_basis_matrix`](@ref) (as in [`build_fsbp_operator`](@ref)) and raw
callables use [`_sample_matrix`](@ref).
"""
# ── Public entry points (two ways to supply the exactness basis) ─────────────

"""
    optimize_fsbp_operator(x, w, xL, xR, basis::AbstractBasis; kwargs...)

`AbstractBasis` entry: sample `V`/`Vx`/`vL`/`vR` via `eval_basis_*` (same as the
exact construction path in `build_fsbp_operator`), then run the shared core.
"""
function optimize_fsbp_operator(x, w, xL, xR, basis::AbstractBasis;
                                quad_basis = nothing, kwargs...)
    quad_basis === nothing && (quad_basis = basis)
    K = nbasis(basis)
    value_eval = z -> eval_basis(basis, z)
    deriv_eval = z -> eval_basis_derivative(basis, z)
    setup = _optimize_fsbp_preamble(x, w, xL, xR, value_eval, deriv_eval, K; kwargs...)
    V = eval_basis_matrix(basis, setup.x)
    Vx = eval_basis_derivative_matrix(basis, setup.x)
    vL = eval_basis_vector(basis, xL)
    vR = eval_basis_vector(basis, xR)
    _require_uniform_type("optimize_fsbp_operator Vandermonde", [
        setup.T, eltype(V), eltype(Vx), eltype(vL), eltype(vR)])
    return _optimize_fsbp_operator_core(setup, V, Vx, vL, vR, basis, quad_basis;
                                        _optimize_core_kwargs(kwargs)...)
end

"""
    optimize_fsbp_operator(x, w, xL, xR, basis_functions, basis_derivatives; kwargs...)

Callable-vector entry: wraps `funcs`/`derivs` as evaluators, samples via
[`_sample_matrix`](@ref) / [`_sample_vector`](@ref), then runs the shared core.
"""
function optimize_fsbp_operator(x, w, xL, xR,
                                basis_functions,
                                basis_derivatives;
                                quad_basis = nothing, kwargs...)
    funcs = collect(basis_functions)
    derivs = collect(basis_derivatives)
    length(funcs) == length(derivs) ||
        throw(ArgumentError("basis_functions and basis_derivatives must have the same length."))

    K = length(funcs)
    value_eval = z -> [f(z) for f in funcs]
    deriv_eval = z -> [df(z) for df in derivs]
    setup = _optimize_fsbp_preamble(x, w, xL, xR, value_eval, deriv_eval, K; kwargs...)
    V = _sample_matrix(value_eval, K, setup.x, setup.T; context = "optimize_fsbp_operator")
    Vx = _sample_matrix(deriv_eval, K, setup.x, setup.T; context = "optimize_fsbp_operator")
    vL = _sample_vector(value_eval, K, xL, setup.T; context = "optimize_fsbp_operator")
    vR = _sample_vector(value_eval, K, xR, setup.T; context = "optimize_fsbp_operator")
    T = setup.T
    op_basis = FunctionBasis(funcs; derivs = derivs, interval = (T(xL), T(xR)))
    quad_basis === nothing && (quad_basis = op_basis)
    return _optimize_fsbp_operator_core(setup, V, Vx, vL, vR, op_basis, quad_basis;
                                        _optimize_core_kwargs(kwargs)...)
end

# ── Shared setup, keyword routing, and core ──────────────────────────────────

"""
    _optimize_fsbp_preamble(x, w, xL, xR, value_eval, deriv_eval, K; kwargs...)

Common setup for both entry paths: validate norm/keyword options, normalize
`x`/`w`, resolve test functions, and determine a single working type `T` via
[`_require_uniform_working_type`](@ref).  Does not build Vandermonde matrices.
"""
function _optimize_fsbp_preamble(x, w, xL, xR, value_eval, deriv_eval, K::Int; kwargs...)
    test_functions = get(kwargs, :test_functions, Function[])
    test_derivatives = get(kwargs, :test_derivatives, Function[])
    extrapolation_norm = get(kwargs, :extrapolation_norm, :Hinv)
    derivative_error_norm = get(kwargs, :derivative_error_norm, :H)
    zero_boundary_scaling = get(kwargs, :zero_boundary_scaling, :fallback)

    K > 0 || throw(ArgumentError("The basis must contain at least one function."))
    _validate_norm_symbol(extrapolation_norm, (:Hinv, :H, :Euclidean, :Frobenius),
                          "extrapolation_norm")
    _validate_norm_symbol(derivative_error_norm, (:Hinv, :H, :Euclidean, :Frobenius),
                          "derivative_error_norm")
    zero_boundary_scaling in (:fallback, :omit) ||
        throw(ArgumentError("zero_boundary_scaling must be :fallback or :omit."))

    x = collect(x)
    w = collect(w)
    N = length(x)
    length(w) == N || throw(ArgumentError("x and w must have the same length."))
    N >= K || throw(ArgumentError(
        "The number of nodes ($N) is less than the basis dimension ($K)."))

    test_funcs, test_derivs = _normalise_test_functions(test_functions, test_derivatives)
    T = _require_uniform_working_type(x, w, xL, xR,
                                      value_eval, deriv_eval, test_funcs, test_derivs)
    any(w .<= zero(T)) &&
        throw(ArgumentError("All quadrature weights must be positive for diagonal-norm optimization."))

    return (; x, w, xL, xR, N, T, test_funcs, test_derivs, K)
end

# Keywords consumed only by `_optimize_fsbp_preamble` (must not be forwarded to core).
const _OPTIMIZE_PREAMBLE_KW = (
    :test_functions, :test_derivatives,
    :extrapolation_norm, :derivative_error_norm,
    :zero_boundary_scaling,
)

"""Strip preamble-only keywords before calling `_optimize_fsbp_operator_core`."""
function _optimize_core_kwargs(kwargs)
    return (; (k => v for (k, v) in pairs(kwargs) if k ∉ _OPTIMIZE_PREAMBLE_KW)...)
end

"""
    _optimize_fsbp_operator_core(setup, V, Vx, vL, vR; kwargs...)

Optimization from pre-built Vandermonde data.  All Marchildon-Zingg logic (rank
check, extrapolation, skew system, optional `S` refinement) lives here so it
is not duplicated between the `AbstractBasis` and callable-vector entry points.
"""
function _optimize_fsbp_operator_core(setup, V, Vx, vL, vR, op_basis, quad_basis;
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

    (; x, w, xL, xR, N, T, test_funcs, test_derivs, K) = setup

    rankV = _check_vandermonde_rank(V, K, T; rank_tol,
                                              context = "optimize_fsbp_operator")

    # -- Extract the optimization weights for the tL/tR and S optimizations
    #    _acc weights the accuracy on the test_funcs (either extrapolation or derivative)
    #    _norm weights the norm of the operators (tL/tR or A = D + H^(-1) tL tL^T)
    theta_ext_acc = T(_objective_weight(extrapolation_objective_weights, :accuracy, 1, 1//2))
    theta_ext_norm = T(_objective_weight(extrapolation_objective_weights, :norm, 2, 1//2))
    theta_S_acc = T(_objective_weight(S_objective_weights, :accuracy, 1, 1//2))
    theta_S_norm = T(_objective_weight(S_objective_weights, :norm, 2, 1//2))
    theta_ext_acc < zero(T) && throw(ArgumentError("extrapolation accuracy weight must be nonnegative."))
    theta_ext_norm < zero(T) && throw(ArgumentError("extrapolation norm weight must be nonnegative."))
    theta_S_acc < zero(T) && throw(ArgumentError("S objective accuracy weight must be nonnegative."))
    theta_S_norm < zero(T) && throw(ArgumentError("S objective norm weight must be nonnegative."))

    ext_scale_tol = extrapolation_scale_tol === nothing ? sqrt(eps(T)) : T(extrapolation_scale_tol)
    der_scale_tol = derivative_scale_tol === nothing ? sqrt(eps(T)) : T(derivative_scale_tol)
    obj_tol = objective_tol === nothing ? sqrt(eps(T)) : T(objective_tol)
    # -- Prepare the test functions (and their relative weights) by H-orthogonalizing w.r.t. V
    weights = _test_weights(T, length(test_funcs), test_weights)
    M = _exactness_gram_matrix(V, w) # modal mass matrix
    test_samples = _precompute_test_orthogonal_samples(
        test_funcs, test_derivs, x, xL, xR, V, Vx, w, vL, vR, M)

    if verbose
        println("\n")
        println("Optimization-based FSBP construction")
        println("  num of nodes = $N")
        println("  dim of basis = $K")
        println("  num of test funcs = $(length(test_funcs))")
        println("  rank(V) = $rankV")
        println("  dim null(V^T) = $(N - K)")
    end

    # -- Determine if the left and right endpoints are endpoints of the interval
    left_endpoint_idx = _endpoint_node_index(x, xL)
    right_endpoint_idx = _endpoint_node_index(x, xR)
    left_is_endpoint = left_endpoint_idx !== nothing
    right_is_endpoint = right_endpoint_idx !== nothing
    dim_null_V = N - K

    # -- Basis of ker(V') when N > K (used for tL/tR optimization and the skew system).
    ZV = dim_null_V > 0 ? _nullspace_basis(V'; rank_tol = rank_tol) :
                           zeros(T, N, 0)
    ZL = left_is_endpoint ? zeros(T, N, 0) : ZV
    ZR = right_is_endpoint ? zeros(T, N, 0) : ZV
    nL = size(ZL, 2)
    nR = size(ZR, 2)
    has_extrapolation_free_parameters = (nL + nR) > 0

    if verbose
        println("  num free tL/tR params = $(nL + nR)")
        println("\n")
    end

    # -- Compute initial tL/tR vectors OR get the exact tL=e1, tR=eN
    tL0, tR0 = _build_extrapolation(V, x, w, vL, vR, xL, xR, extrapolation_norm)

    if has_extrapolation_free_parameters || verbose
        # -- Prepare the tests for the extrapolation optimization using orthogonalized test space
        ext_tests = _build_extrapolation_tests(test_samples, weights, w,
                                               zero_boundary_scaling, ext_scale_tol)
        J_ext_acc_initial = _extrapolation_accuracy_objective(tL0, tR0, ext_tests)
        J_ext_norm_initial = _extrapolation_norm_objective(tL0, tR0, w, extrapolation_norm)
    else
        ext_tests = NamedTuple[]
    end

    if has_extrapolation_free_parameters
        tL, tR = _optimize_extrapolation(tL0, tR0, ZL, ZR, ext_tests, w,
                                         extrapolation_norm, theta_ext_acc,
                                         theta_ext_norm, J_ext_acc_initial,
                                         J_ext_norm_initial, obj_tol;
                                         rank_tol = rank_tol)
    else
        tL, tR = tL0, tR0
    end

    if verbose
        if has_extrapolation_free_parameters
            J_ext_acc_final = _extrapolation_accuracy_objective(tL, tR, ext_tests)
            J_ext_norm_final = _extrapolation_norm_objective(tL, tR, w, extrapolation_norm)
        else
            J_ext_acc_final = J_ext_acc_initial
            J_ext_norm_final = J_ext_norm_initial
        end
        ext_exact_L = _euclidean_norm(V' * tL - vL)
        ext_exact_R = _euclidean_norm(V' * tR - vR)

        println("After extrapolation stage:")
        println("  extrapolation optimization active = $has_extrapolation_free_parameters")
        println("  norm backend = $extrapolation_norm")
        println("  initial extrapolation accuracy objective = $J_ext_acc_initial")
        println("  final extrapolation accuracy objective = $J_ext_acc_final")
        println("  accuracy ratio final/initial = $(_objective_ratio(J_ext_acc_final, J_ext_acc_initial))")
        println("  initial extrapolation norm objective = $J_ext_norm_initial")
        println("  final extrapolation norm objective = $J_ext_norm_final")
        println("  norm ratio final/initial = $(_objective_ratio(J_ext_norm_final, J_ext_norm_initial))")
        println("  ||V^T tL - v_L|| = $ext_exact_L")
        println("  ||V^T tR - v_R|| = $ext_exact_R")
    end

    # -- Form the boundary matrix E = tR tRᵀ - tL tLᵀ
    E = tR * tR' - tL * tL'
    if verbose
        E_boundary_residual = _frobenius_norm(V' * E * V - (vR * vR' - vL * vL'))
        println("  ||V^T E V - (v_R v_R^T - v_L v_L^T)|| = $E_boundary_residual")
        println("\n")
    end

    # -- Get ready for S optimization: check the SBP compatibility first
    compat_residual_matrix = _sbp_compatibility_residual(V, Vx, w, vL, vR)
    compat_residual = _frobenius_norm(compat_residual_matrix)
    compat_scale = max(one(T), _frobenius_norm(V' * _scale_rows(Vx, w)),
                         _frobenius_norm(vR * vR' - vL * vL'))
    compat_tol_eff = compatibility_tol === nothing ?
        T(100) * sqrt(eps(T)) * compat_scale : T(compatibility_tol)
    if verbose
        println("Quadrature/SBP compatibility residual = $compat_residual")
    end
    if compat_residual > compat_tol_eff
        msg = "Quadrature/SBP compatibility residual $compat_residual exceeds tolerance $compat_tol_eff; exact construction of S may be impossible."
        if compatibility_action === :error
            error(msg)
        elseif compatibility_action === :warn
            _stderr_warn(msg)
        elseif compatibility_action !== :ignore
            throw(ArgumentError("compatibility_action must be :warn, :error, or :ignore."))
        end
    end

    # -- Construct the linear system for S optimization
    half = one(T) / T(2)
    pairs = _skew_pairs(N) # vector of index pairs (i, j) with i < j for S[i,j]
    # assemble the accuracy conditions S V = H Vₓ - ½ E V into a simple linear system C s = d
    C, d = _assemble_independent_skew_system(V, Vx, w, E, ZV, pairs, T)
    # C should be size (nE, nS), where
    #   nS = N * (N - 1) / 2  (number of independent entries in S)
    #   nE = K * (K - 1) / 2 + dim_null_V * K  (number of independent equations)
    nS = length(pairs)
    nE = length(d)
    expected_nS = N * (N - 1) ÷ 2
    expected_nE = K * (K - 1) ÷ 2 + dim_null_V * K # = K * (2 * N - K - 1) ÷ 2
    if nS != expected_nS
        _stderr_warn("n_S = $nS but expected N(N-1)/2 = $expected_nS for N = $N.")
    end
    if nE != expected_nE
        _stderr_warn("n_E = $nE but expected K(K-1)/2 + dim_null_V * K = $expected_nE for K = $K and dim_null_V = $dim_null_V.")
    end
    if size(C) != (expected_nE, expected_nS)
        _stderr_warn("size(C) = $(size(C)) but expected (n_E, n_S) = ($expected_nE, $expected_nS).")
    end
    expected_free = (N - K) * (N - K - 1) ÷ 2
    if expected_free != (nS - nE)
        _stderr_warn("expected_free = $expected_free but nS - nE = $(nS - nE) for N = $N, K = $K, and dim_null_V = $dim_null_V.")
    end

    # -- Get the minimum-norm solution s0 = pinv(C) d 
    #    We then parametrize all possible S matrices through the nullspace basis ZC of C,
    #    so that S = S0 + ZS * a, and we solve the optimization problem for a.
    s0 = _pseudoinverse_solve(C, d; rank_tol = rank_tol)
    ZS = _nullspace_basis(C; rank_tol = rank_tol)
    S0 = _unvec_skew(s0, N, pairs) # unwrap from the vector s0 into a proper matrix S0
    
    if verbose
        rankC = _matrix_rank(C; rank_tol = rank_tol)
        nullC = nS - rankC
        C_residual = _euclidean_norm(C * s0 - d)
        S0_accuracy_residual = _frobenius_norm(S0 * V - (_scale_rows(Vx, w) - half * E * V))

        println("Before S optimization:")
        println("  # of independent S entries = $nS")
        println("  # of independent equations = $nE")
        println("  # of free parameters = $(nS - nE)")
        println("  size(C) = $(size(C))")
        println("  rank(C) = $rankC")
        println("  null(C) = $nullC")
        println("  The following residuals test the exactness of the initial S0 solution:")
        println("  ||C s_0 - d|| = $C_residual")
        println("  ||S_0 V - (H V_x - (1/2) E V)|| = $S0_accuracy_residual")
        println("\n")
    end

    # -- Set initial Q, D, and J_up for the S optimization
    Q0 = S0 + half * E # weak-derivative operator
    D0 = _divide_rows(Q0, w) # derviative operator
    J0 = D0 + _divide_rows(tL * tL', w) # upwind linear convection operator

    nfreeS = size(ZS, 2)
    if nfreeS > 0 || verbose
        # -- Prepare the tests for the S optimization using orthogonalized test space
        der_tests = _build_derivative_tests(test_samples, weights, w, der_scale_tol)
        
        J_der_initial = _derivative_objective(D0, der_tests, w, derivative_error_norm)
        J_norm_initial = _full_jacobian_objective(J0)
    end

    if nfreeS == 0 # no free parameters in S: return base S, Q, D
        S = S0
        Q = Q0
        D = D0
        J_up = J0
    else
        S_basis = [_unvec_skew(ZS[:, j], N, pairs) for j in 1:nfreeS]
        # D = H⁻¹ S + ½ H⁻¹ E , so D_basis is affine in S_basis
        D_basis = [_divide_rows(Sj, w) for Sj in S_basis]
        y = _optimize_S_parameters(D0, J0, D_basis, der_tests, w,
                                   derivative_error_norm,
                                   theta_S_acc, theta_S_norm,
                                   J_der_initial, J_norm_initial, obj_tol;
                                   rank_tol = rank_tol)
        # -- Update to final Q, D, and S
        S = copy(S0)
        for j in 1:nfreeS
            S .+= y[j] .* S_basis[j]
        end
        Q = S + half * E
        D = _divide_rows(Q, w)
        J_up = D + _divide_rows(tL * tL', w)
    end

    if verbose
        rank_D, rank_J, eigvals_J, spectral_radius_J =
            _spectrum_diagnostics(D, J_up; rank_tol = rank_tol)
        if nfreeS > 0
            J_der_final = _derivative_objective(D, der_tests, w, derivative_error_norm)
            J_norm_final = _full_jacobian_objective(J_up)
        else
            J_der_final = J_der_initial
            J_norm_final = J_norm_initial
        end
        derivative_residual = _frobenius_norm(D * V - Vx)
        sbp_residual = _frobenius_norm(Q + Q' - E)

        println("After S optimization:")
        println("  # of S free params = $nfreeS")
        println("  initial derivative objective = $J_der_initial")
        println("  final derivative objective = $J_der_final")
        println("  derivative ratio final/initial = $(_objective_ratio(J_der_final, J_der_initial))")
        println("  initial Jacobian Frobenius objective = $J_norm_initial")
        println("  final Jacobian Frobenius objective = $J_norm_final")
        println("  Jacobian ratio final/initial = $(_objective_ratio(J_norm_final, J_norm_initial))")
        println("  ||D V - V_x|| = $derivative_residual")
        println("  ||Q + Q^T - E|| = $sbp_residual")
        println("  rank(D) = $rank_D")
        println("  rank(D + H⁻¹ tL tLᵀ) = $rank_J")
        println("  spectral radius (D + H⁻¹ tL tLᵀ) = $spectral_radius_J")
        if eigvals_J !== nothing && !isempty(eigvals_J)
            re_J = real.(eigvals_J)
            println("  Re(λ) range (D + H⁻¹ tL tLᵀ) = [$(minimum(re_J)), $(maximum(re_J))]")
        end
    end

    return FSBPOperator{T}(D, Diagonal(w), Q, S, E, tL, tR, x, w,
                         op_basis, quad_basis, (xL, xR), N, K)
end

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation and scalar-type helpers
# ─────────────────────────────────────────────────────────────────────────────

_stderr_warn(msg::AbstractString) = println(stderr, "GaussFSBP warning: ", msg)

function _collect_scalar_types_from_eval(eval, points)
    types = Type[]
    for x in points
        vals = eval(x)
        for v in vals
            push!(types, typeof(v))
        end
    end
    return types
end

function _collect_scalar_types_from_callables(callables, points)
    types = Type[]
    for f in callables, x in points
        push!(types, typeof(f(x)))
    end
    return types
end

function _require_uniform_working_type(x, w, xL, xR,
                                       value_eval, deriv_eval,
                                       test_funcs, test_derivs)
    isempty(x) && throw(ArgumentError("x must be nonempty."))
    sample_pts = (x[1], xL, xR)
    types = Type[
        _array_element_type(x, "x"),
        _array_element_type(w, "w"),
        typeof(xL),
        typeof(xR),
    ]
    append!(types, _collect_scalar_types_from_eval(value_eval, sample_pts))
    append!(types, _collect_scalar_types_from_eval(deriv_eval, (x[1],)))
    append!(types, _collect_scalar_types_from_callables(test_funcs, sample_pts))
    append!(types, _collect_scalar_types_from_callables(test_derivs, (x[1],)))
    return _require_uniform_type("optimize_fsbp_operator", types)
end

"""
    _normalise_test_functions(test_functions, test_derivatives) -> (funcs, derivs)

Convert optional extrapolation/derivative test input to `Vector{Function}` pairs.
Accepts `nothing`, a `FunctionBasis`, or a vector of callables; derivatives may
come from `test_derivatives` or from `test_functions.derivs` when empty.
Returns empty vectors when no extra tests are supplied.
"""
function _normalise_test_functions(test_functions, test_derivatives)
    funcs = if test_functions === nothing
        Function[]
    elseif test_functions isa FunctionBasis
        collect(basis_functions(test_functions))
    else
        collect(test_functions)
    end

    derivs = if test_derivatives === nothing
        Function[]
    elseif !(test_derivatives isa AbstractVector) && test_derivatives isa FunctionBasis
        test_derivatives.derivs === nothing ? Function[] : collect(test_derivatives.derivs)
    elseif test_functions isa FunctionBasis && isempty(test_derivatives)
        test_functions.derivs === nothing ? Function[] : collect(test_functions.derivs)
    else
        collect(test_derivatives)
    end

    if !isempty(derivs) && length(derivs) != length(funcs)
        throw(ArgumentError("test_derivatives must be empty or have the same length as test_functions."))
    end
    return funcs, derivs
end

function _test_weights(::Type{T}, M::Int, weights) where T
    if weights === nothing
        return ones(T, M)
    end
    length(weights) == M ||
        throw(ArgumentError("test_weights has length $(length(weights)), expected $M."))
    omega = T.(collect(weights))
    any(omega .<= zero(T)) && throw(ArgumentError("All test_weights must be positive."))
    return omega
end

function _objective_weight(weights, key::Symbol, idx::Int, default)
    if weights isa NamedTuple && key in keys(weights)
        return getproperty(weights, key)
    elseif weights isa Tuple && length(weights) >= idx
        return weights[idx]
    else
        return default
    end
end

function _validate_norm_symbol(value::Symbol, allowed, name::AbstractString)
    value in allowed ||
        throw(ArgumentError("$name must be one of $(allowed), got $value."))
end

# ─────────────────────────────────────────────────────────────────────────────
# Weighted operations and objectives
# ─────────────────────────────────────────────────────────────────────────────

# Column scaling: equivalent to `w .* A` for matrices; reshape keeps broadcasting explicit.
_scale_rows(A, w) = A .* reshape(w, :, 1)
_divide_rows(A, w) = A ./ reshape(w, :, 1)

function _euclidean_norm(v)
    return sqrt(sum(abs2, v))
end

function _frobenius_norm(A)
    return sqrt(sum(abs2, A))
end

function _weighted_norm2(v, w, norm_backend::Symbol)
    if norm_backend === :H
        return dot(v, w .* v)
    elseif norm_backend === :Hinv
        return dot(v, v ./ w)
    elseif norm_backend in (:Euclidean, :Frobenius)
        return dot(v, v)
    else
        throw(ArgumentError("Unsupported norm backend $norm_backend."))
    end
end

function _residual_norm_scale(w, norm_backend::Symbol)
    if norm_backend === :H
        return sqrt.(w)
    elseif norm_backend === :Hinv
        return one(eltype(w)) ./ sqrt.(w)
    elseif norm_backend in (:Euclidean, :Frobenius)
        return ones(eltype(w), length(w))
    else
        throw(ArgumentError("Unsupported norm backend $norm_backend."))
    end
end

function _objective_ratio(final, initial)
    # Guard against exact 0/0.
    if iszero(initial)
        return iszero(final) ? zero(final) : oftype(final, Inf)
    end
    return final / initial
end

"""
    _exactness_gram_matrix(V, w) -> Matrix

Gram matrix `M = V' H V` with `H = diag(w)`.  Build once and reuse for many
projections of nodal test values onto the exactness space.
Also the modal mass matrix for the exactness space.
"""
function _exactness_gram_matrix(V, w)
    # Equivalent to V' * _scale_rows(V, w); column scaling via broadcast avoids reshape.
    return V' * (w .* V)
end

"""
    _project_coefficients(V, g, w; M) -> Vector

Discrete H-norm projection of nodal values `g` onto the exactness space spanned
by the columns of `V`.  With `H = diag(w)`, returns `c = P g` where
`P = M⁻¹ V' H` is the discrete projection, and `M = V' H V` is the modal mass matrix.
"""
function _project_coefficients(V, g, w; M = nothing)
    M === nothing && (M = _exactness_gram_matrix(V, w))
    return M \ (V' * (w .* g))
end

"""
    _precompute_test_orthogonal_samples(...) -> Vector{NamedTuple}

For each test function, sample at nodes and boundaries, project onto `span(V)`
with the discrete projection operator, and store H-orthogonal parts used by both
extrapolation and derivative objective builders.
"""
function _precompute_test_orthogonal_samples(test_funcs, test_derivs, x, xL, xR,
                                             V, Vx, w, vL, vR, M)
    T = eltype(x)
    use_derivs = !isempty(test_derivs)
    if use_derivs && length(test_derivs) != length(test_funcs)
        throw(ArgumentError("test_derivatives length must match test_functions."))
    end
    samples = NamedTuple[]
    for m in eachindex(test_funcs)
        # -- get a vector of values of the test function at the nodes
        g = T.([test_funcs[m](xi) for xi in x])
        # -- project the test function onto the basis (exactness space)
        c = _project_coefficients(V, g, w; M = M)
        # -- isolate the part of g that is orthogonal to the basis (exactness space)
        g_perp = g - V * c
        gL_perp = T(test_funcs[m](xL)) - dot(vL, c)
        gR_perp = T(test_funcs[m](xR)) - dot(vR, c)
        gx_perp = if use_derivs
            gx = T.([test_derivs[m](xi) for xi in x])
            gx - Vx * c
        else
            nothing
        end
        push!(samples, (; g_perp, gL_perp, gR_perp, gx_perp))
    end
    return samples
end

function _minimum_extrapolation_solution(V, w, b, norm_backend::Symbol)
    if norm_backend === :Hinv
        MV = _scale_rows(V, w)
        G = V' * MV
    elseif norm_backend === :H
        MV = _divide_rows(V, w)
        G = V' * MV
    elseif norm_backend in (:Euclidean, :Frobenius)
        MV = V
        G = V' * V
    else
        throw(ArgumentError("Unsupported extrapolation norm $norm_backend."))
    end
    # The builders rank-check V before calling this helper, so G is nonsingular
    # for positive quadrature weights. This is the minimum-norm Gram solve.
    return MV * (G \ b)
end

# ─────────────────────────────────────────────────────────────────────────────
# Extrapolation optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_extrapolation_tests(samples, ...) -> Vector{NamedTuple}

From precomputed H-orthogonal samples, attach boundary scaling and activity
flags for the extrapolation accuracy objective.
"""
function _build_extrapolation_tests(samples, weights, w,
                                    zero_boundary_scaling, scale_tol)
    T = eltype(w)
    tests = NamedTuple[]
    for m in eachindex(samples)
        s = samples[m]
        # -- scale boundary errors by the exact boundary value of the
        #    projected-out component. If that scalar is too small, either omit
        #    that boundary residual or fall back to the H-norm of the same
        #    projected-out component.
        deltaL = abs(s.gL_perp)
        deltaR = abs(s.gR_perp)
        activeL = true
        activeR = true
        if deltaL <= scale_tol
            if zero_boundary_scaling === :omit
                activeL = false
            else
                deltaL = max(scale_tol, sqrt(_weighted_norm2(s.g_perp, w, :H)))
            end
        end
        if deltaR <= scale_tol
            if zero_boundary_scaling === :omit
                activeR = false
            else
                deltaR = max(scale_tol, sqrt(_weighted_norm2(s.g_perp, w, :H)))
            end
        end
        push!(tests, (g_perp = s.g_perp, gL_perp = s.gL_perp, gR_perp = s.gR_perp,
                      deltaL = deltaL, deltaR = deltaR,
                      activeL = activeL, activeR = activeR,
                      omega = weights[m]))
    end
    return tests
end

function _extrapolation_accuracy_objective(tL, tR, tests)
    T = eltype(tL)
    total = zero(T)
    for t in tests
        if t.activeL
            errL = (dot(tL, t.g_perp) - t.gL_perp) / t.deltaL
            total += t.omega * errL * errL
        end
        if t.activeR
            errR = (dot(tR, t.g_perp) - t.gR_perp) / t.deltaR
            total += t.omega * errR * errR
        end
    end
    return total
end

function _extrapolation_norm_objective(tL, tR, w, norm_backend::Symbol)
    return _weighted_norm2(tL, w, norm_backend) +
           _weighted_norm2(tR, w, norm_backend)
end

function _append_row!(rows, rhs, row, constant)
    push!(rows, row)
    push!(rhs, constant)
    return nothing
end

function _rows_to_matrix(rows, rhs, nvars::Int, ::Type{T}) where T
    A = zeros(T, length(rows), nvars)
    b = Vector{T}(undef, length(rhs))
    for i in eachindex(rows)
        A[i, :] = rows[i]
        b[i] = rhs[i]
    end
    return A, b
end

"""
    _optimize_extrapolation(...) -> (tL, tR)

Refine `tL0`, `tR0` in `ker(V')` with one weighted least-squares step.  Columns of
`ZL`, `ZR` span the nullspace of `V'`; exactness `V' t = v` is preserved because
`V' Z = 0`.

Parameterization (free coefficients `a = [aL; aR]`):

    tL = tL0 + ZL*aL,    tR = tR0 + ZR*aR

Objectives at the starting point (see `_extrapolation_accuracy_objective`,
`_extrapolation_norm_objective`):

    J_acc(tL,tR)  = Σ_m ω_m ( ⟨tL,g_m^⊥⟩ - g_{L,m}^⊥ )^2 / δ_{L,m}^2  +  (right)
    J_norm(tL,tR) = ‖tL‖_norm^2 + ‖tR‖_norm^2

Each block below writes its affine residuals in `a` as rows `row` with rhs `constant`
(`row' a + constant`), stacks them, and solves `A a ≈ -b` for the exact
quadratic minimizer, then updates `tL`, `tR`.
"""
function _optimize_extrapolation(tL0, tR0, ZL, ZR, tests, w, norm_backend,
                                 theta_acc, theta_norm, J_acc0, J_norm0, obj_tol;
                                 rank_tol)
    T = eltype(tL0)
    nL = size(ZL, 2)
    nR = size(ZR, 2)
    nvars = nL + nR
    nvars == 0 && return tL0, tR0

    rows = Vector{Vector{T}}()
    rhs = T[]

    # ── Accuracy block (active if θ_acc > 0 and J_acc0 > obj_tol) ───────────
    # Left-boundary residual for one test (ω = t.omega, δ = t.deltaL):
    #
    #   r_L(a) = (√ω / δ) ( ⟨tL, g^⊥⟩ - g_L^⊥ )
    #          = (√ω / δ) ( ⟨tL0, g^⊥⟩ - g_L^⊥ + aL' (ZL' g^⊥) )
    #
    # Now stack row' a + constant ≈ 0 with
    #   row[1:nL]   = (√ω / δ) (ZL' g^⊥) * global_scale,   global_scale = √(θ_acc / J_acc0)
    #   constant    = (√ω / δ) ( ⟨tL0, g^⊥⟩ - g_L^⊥ ) * global_scale  (= r_L(0) at a = 0)
    #
    # So row' a + constant = global_scale * r_L(a).  Minimize ‖global_scale * r_L(a)‖^2
    # After stacking all rows (accuracy + norm), we get A[i,:] = row_i,  b[i] = constant_i,
    # so minimizing ||A a + b||_2^2 drives the objective function toward zero.
    if theta_acc > zero(T) && J_acc0 > obj_tol
        global_scale = sqrt(theta_acc / J_acc0)
        for t in tests
            sqrtomega = sqrt(t.omega)
            if t.activeL && nL > 0
                row = zeros(T, nvars)
                row[1:nL] .= (ZL' * t.g_perp) .* (sqrtomega * global_scale / t.deltaL)
                constant = (dot(tL0, t.g_perp) - t.gL_perp) *
                           sqrtomega * global_scale / t.deltaL
                _append_row!(rows, rhs, row, constant)
            end
            if t.activeR && nR > 0
                row = zeros(T, nvars)
                row[(nL + 1):nvars] .= (ZR' * t.g_perp) .* (sqrtomega * global_scale / t.deltaR)
                constant = (dot(tR0, t.g_perp) - t.gR_perp) *
                           sqrtomega * global_scale / t.deltaR
                _append_row!(rows, rhs, row, constant)
            end
        end
    end

    # ── Norm block (active if θ_norm > 0 and J_norm0 > obj_tol) ──────────────
    # J_norm(tL,tR) = ‖tL‖_norm^2 + ‖tR‖_norm^2.  With s = norm_scale (so ‖t‖_norm^2 = ‖s⊙t‖_2^2),
    # and tL = tL0 + ZL*aL, the left nodal component i is affine in a:
    #
    #   u_{L,i}(a) = s_i t_{L,i}(a) = s_i t_{L0,i} + Σ_j Zscaled[i,j] a_{L,j}
    #              = base[i] + Zscaled[i,:]' aL,    base = s ⊙ tL0,  Zscaled[i,j] = s_i ZL[i,j]
    #
    # Stack row' a + constant ≈ 0 with  global_scale = √(θ_norm / J_norm0)
    #   row[1:nL]   = global_scale * Zscaled[i, :]
    #   constant    = global_scale * base[i]  (= global_scale * u_{L,i}(0) at a = 0)
    #
    # So row' a + constant = global_scale * u_{L,i}(a).  We minimize ‖global_scale * u_{L,i}(a)‖^2
    # Together with the accuracy rows, A[i,:] = row_i, b[i] = constant_i and
    # min ||A a + b||_2^2 is exact quadratic minimization.
    use_direct_lsq = theta_norm > zero(T) && J_norm0 > obj_tol
    if use_direct_lsq
        global_scale = sqrt(theta_norm / J_norm0)
        norm_scale = _residual_norm_scale(w, norm_backend)
        if nL > 0
            base = norm_scale .* tL0
            Zscaled = ZL .* reshape(norm_scale, :, 1)
            for i in eachindex(base)
                row = zeros(T, nvars)
                row[1:nL] .= global_scale .* Zscaled[i, :]
                _append_row!(rows, rhs, row, global_scale * base[i])
            end
        end
        if nR > 0
            base = norm_scale .* tR0
            Zscaled = ZR .* reshape(norm_scale, :, 1)
            for i in eachindex(base)
                row = zeros(T, nvars)
                row[(nL + 1):nvars] .= global_scale .* Zscaled[i, :]
                _append_row!(rows, rhs, row, global_scale * base[i])
            end
        end
    end

    isempty(rows) && return tL0, tR0
    A, b = _rows_to_matrix(rows, rhs, nvars, T)
    # When the norm block is present, its positive diagonal scaling of the
    # full-rank nullspace bases makes A full column rank. Use the faster
    # direct least-squares solve, with SVD fallback for degenerate cases.
    a = -_least_squares_solve(A, b; rank_tol = rank_tol,
                              prefer_direct = use_direct_lsq)
    tL = tL0 + ZL * a[1:nL]
    tR = tR0 + ZR * a[(nL + 1):nvars]
    return tL, tR
end

# ─────────────────────────────────────────────────────────────────────────────
# Independent-equation skew system
# ─────────────────────────────────────────────────────────────────────────────

function _sbp_compatibility_residual(V, Vx, w, vL, vR)
    return V' * _scale_rows(Vx, w) + Vx' * _scale_rows(V, w) -
           (vR * vR' - vL * vL')
end

# Upper-triangle index pairs (i, j) with i < j: independent entries of an N×N skew matrix.
function _skew_pairs(N::Int)
    pairs = Vector{Tuple{Int,Int}}()
    for j in 2:N
        for i in 1:(j - 1)
            push!(pairs, (i, j))
        end
    end
    return pairs
end

function _skew_bilinear_row(a, b, pairs, ::Type{T}) where T
    row = Vector{T}(undef, length(pairs))
    for p in eachindex(pairs)
        i, j = pairs[p]
        row[p] = a[i] * b[j] - a[j] * b[i]
    end
    return row
end

"""
    _assemble_independent_skew_system(V, Vx, w, E, ZV, pairs, T) -> (C, d)

Build the independent-equation linear system `C s = d` for the skew matrix `S`.

`S` is stored by its upper-triangle entries `s` (index layout from `pairs`).  For nodal
vectors `a`, `b`, the bilinear form `a' S b` is linear in `s`; `_skew_bilinear_row`
returns the corresponding row of `C`.

Two blocks of equations (Marchildon-Zingg):

Start from the full system:  S V = H Vₓ - ½ E V
Separate into two blocks:  Vᵀ S V =  Vᵀ H Vₓ - ½  Vᵀ E V
                          ZVᵀ S V = ZVᵀ H Vₓ - ½ ZVᵀ E V       <-- loop 2
Rearrange the first eqn:   Vᵀ S V = ½(Vᵀ H Vₓ - Vₓᵀ H V)       <-- loop 1
  (compatibility eqn)

1. **Basis pairs** (`V[:,α]`, `V[:,β]`, `α > β`): enforce the SBP derivative relation on
   the exactness space spanned by columns of `V`.
2. **Nullspace × basis** (`ZV[:,μ]`, `V[:,β]`): same relation tested against columns of
   `ZV` (a basis of `ker(V')`), so overdetermined nodal systems are covered.

The minimum-norm solution `s₀ = pinv(C) d` gives the initial `S₀`; `ker(C)` parametrizes
the remaining free skew degrees of freedom optimized later.
"""
function _assemble_independent_skew_system(V, Vx, w, E, ZV, pairs, ::Type{T}) where T
    K = size(V, 2)
    rows = Vector{Vector{T}}()
    rhs = T[]
    half = one(T) / T(2)

    # -- Loop 1: for each basis pair α,β we get an equation
    #   Σᵢⱼ V[i,α] V[j,β] Sᵢⱼ = Σⱼ ½ Hⱼ (V[j,α] Vₓ[j,β] - Vₓ[j,α] V[j,β])
    #   rearrange this into a flat system, where each equation is a pair (α,β)
    #   then s is a flattened vector of the entries Sᵢⱼ
    #   LHS (C) each row is for a pair (α,β), each column is for a pair (i,j)
    #           each column is V[i,α] V[j,β] for upper Sᵢⱼ, and - V[j,α] V[i,β] for lower Sᵢⱼ
    #   RHS (d) each entry is for a pair (α,β)
    #           each row is for a pair (α,β), = Σⱼ ½ Hⱼ (V[j,α] Vₓ[j,β] - Vₓ[j,α] V[j,β])
    #   Note that when α=β the row is all zeros, and the rhs is also zero, so ignore it.
    for alpha in 2:K # take only α > β
        a = V[:, alpha] # basis vector α
        for beta in 1:(alpha - 1) 
            b = V[:, beta]  # basis vector β
            row = _skew_bilinear_row(a, b, pairs, T) # calculate columns for a row of C
            rhs_value = half * (dot(a, w .* Vx[:, beta]) -
                                dot(Vx[:, alpha], w .* b)) # calculate entry of d
            _append_row!(rows, rhs, row, rhs_value) # stack system row by row
        end
    end

    # -- Loop 2: for each nullspace basis vector μ and basis vector β we get an equation
    #   Σᵢⱼ ZV[i,μ] V[j,β] Sᵢⱼ = Σⱼ Hⱼ Vₓ[j,β] ZV[j,μ] - Σᵢⱼ ½ E[i,j] V[j,β] ZV[i,μ]
    #   rearrange this into a flat system, where each equation is a pair (μ,β)
    #   then s is a flattened vector of the entries Sᵢⱼ
    #   LHS (C) each row is for a pair (μ,β), each column is for a pair (i,j)
    #           each column is ZV[i,μ] V[j,β] for upper Sᵢⱼ, and - ZV[k,μ] V[i,β] for lower Sᵢⱼ
    #   RHS (d) each entry is for a pair (μ,β)
    #           each row is for a pair (μ,β), = Σᵢ (Hᵢ Vₓ[i,β] - Σⱼ ½ E[i,j] V[j,β]) ZV[i,μ]
    for mu in 1:size(ZV, 2)
        a = ZV[:, mu]
        for beta in 1:K
            b = V[:, beta]
            row = _skew_bilinear_row(a, b, pairs, T)
            rhs_vec = w .* Vx[:, beta] - half * (E * b)
            _append_row!(rows, rhs, row, dot(a, rhs_vec))
        end
    end

    # build actual system matrix C and right-hand side d by unrapping the stacked vectors
    return _rows_to_matrix(rows, rhs, length(pairs), T)
end

function _unvec_skew(s, N::Int, pairs)
    T = eltype(s)
    S = zeros(T, N, N)
    for p in eachindex(pairs)
        i, j = pairs[p]
        S[i, j] = s[p]
        S[j, i] = -s[p]
    end
    return S
end

# ─────────────────────────────────────────────────────────────────────────────
# S optimization
# ─────────────────────────────────────────────────────────────────────────────

"""
    _build_derivative_tests(samples, ...) -> Vector{NamedTuple}

From precomputed H-orthogonal samples, normalize nodal values for the
derivative accuracy objective (near-zero tests are skipped).
"""
function _build_derivative_tests(samples, weights, w, scale_tol)
    tests = NamedTuple[]
    for m in eachindex(samples)
        gxp = samples[m].gx_perp
        gxp === nothing && continue
        # -- normalize the derivative errors by the H-norm of the H-orthogonal exact derivative
        alpha = sqrt(_weighted_norm2(gxp, w, :H))
        alpha <= scale_tol && continue
        gp = samples[m].g_perp
        # -- if large enough, save g, it's exact derivative, and the weight
        push!(tests, (g_hat = gp ./ alpha,
                      gx_hat = gxp ./ alpha,
                      omega = weights[m]))
    end
    return tests
end

function _derivative_objective(D, tests, w, norm_backend::Symbol)
    T = eltype(D)
    total = zero(T)
    for t in tests
        err = D * t.g_hat - t.gx_hat
        total += t.omega * _weighted_norm2(err, w, norm_backend)
    end
    return total
end

# compute the square Frobenius norm of a matrix
_full_jacobian_objective(A) = sum(abs2, A)

"""
    _optimize_S_parameters(...) -> Vector

Refine `S0` along the independent skew directions in one weighted least-squares step.
Columns of `ZS` (assembled upstream) span the free skew parameters; here we work with
the induced derivative and upwind-Jacobian bases

    D(y) = D0 + Σ_j y_j D_basis[j],    J(y) = J0 + Σ_j y_j D_basis[j]

(`tL`, `tR` are fixed during this stage, so `J = D + H⁻¹ tL tLᵀ` is affine in `y`.)

Objectives at the starting point (see `_derivative_objective`, `_full_jacobian_objective`):

    J_der(y)  = Σ_m ω_m ‖ D(y) ĝ_m - ĝ_{x,m} ‖_der^2
    J_norm(y) = ‖A(y)‖_F^2

Each block writes its affine residuals in `y` as rows `row` with rhs `constant`
(`row' y + constant`), stacks them, and solves `A y ≈ -b` for the exact
quadratic minimizer (direct least squares when the Frobenius block is active,
SVD pseudoinverse otherwise).
"""
function _optimize_S_parameters(D0, J0, D_basis, der_tests, w,
                                derivative_error_norm,
                                theta_S_acc, theta_S_norm, J_der_initial, J_norm_initial, obj_tol;
                                rank_tol)
    T = eltype(D0)
    nvars = length(D_basis)
    nvars == 0 && return zeros(T, 0)
    rows = Vector{Vector{T}}()
    rhs = T[]

    # ── Accuracy block (active if θ_acc > 0 and J_der_initial > obj_tol) ─────
    # For one normalized test (ω = t.omega), nodal derivative residual component i:
    #
    #   r_i(y) = s_i ( D(y) ĝ - ĝ_x )_i     where s_i is the norm weight (usually w_i)
    #          = base[i] + Σ_j cols[j][i] y_j, where
    #   base = s ⊙ (D0 ĝ - ĝ_x),   cols[j] = s ⊙ (D_basis[j] ĝ),   s = _residual_norm_scale(w, ·)
    #
    # Stack row' y + constant ≈ 0 with  global_scale = √(θ_acc / J_der_initial)
    #   row[j]    = √ω global_scale cols[j][i]
    #   constant  = √ω global_scale base[i]  (= √ω global_scale r_i(0) at y = 0)
    #
    # So row' y + constant = √ω global_scale r_i(y).  Minimize ‖global_scale √ω r(y)‖^2
    # over all (t, i); after stacking, A[i,:] = row_i, b[i] = constant_i and
    # min ‖A y + b‖_2^2 is exact quadratic minimization of the scaled objective.
    if theta_S_acc > zero(T) && J_der_initial > obj_tol && !isempty(der_tests)
        global_scale = sqrt(theta_S_acc / J_der_initial)
        norm_scale = _residual_norm_scale(w, derivative_error_norm)
        for t in der_tests
            base = norm_scale .* (D0 * t.g_hat - t.gx_hat)
            cols = [norm_scale .* (Dj * t.g_hat) for Dj in D_basis]
            row_scale = sqrt(t.omega) * global_scale
            for i in eachindex(base)
                row = zeros(T, nvars)
                for j in 1:nvars
                    row[j] = row_scale * cols[j][i]
                end
                _append_row!(rows, rhs, row, row_scale * base[i])
            end
        end
    end

    # ── Full-Jacobian Frobenius norm block (θ_norm > 0 and J_norm_initial > obj_tol) ─
    # J_norm(y) = ‖J(y)‖_F^2 with J(y) = J0 + Σ_k D_basis[k] y_k.  Each entry is affine:
    #
    #   J_{ij}(y) = D(y) + H⁻¹ tL tLᵀ
    #             = J0[i,j] + Σ_k D_basis[k][i,j] y_k.
    #
    # Stack row' y + constant with global_scale = √(θ_norm / J_norm_initial)
    #   row[k]    = global_scale D_basis[k][idx]
    #   constant  = global_scale J0[idx]  (= global_scale J_{ij}(0) at y = 0)
    #
    # So row' y + constant = global_scale J_{ij}(y).  Minimize ‖global_scale J(y)‖_F^2
    # entrywise; Stack these ontop of the accuracy rows.
    if theta_S_norm > zero(T) && J_norm_initial > obj_tol
        global_scale = sqrt(theta_S_norm / J_norm_initial)
        for idx in eachindex(J0)
            row = zeros(T, nvars)
            for j in 1:nvars
                row[j] = global_scale * D_basis[j][idx]
            end
            _append_row!(rows, rhs, row, global_scale * J0[idx])
        end
    end

    isempty(rows) && return zeros(T, nvars)
    A, b = _rows_to_matrix(rows, rhs, nvars, T)
    # With the Frobenius block, N² entry rows make A tall full column rank;
    # use QR/backslash when that block is present, SVD fallback otherwise.
    use_direct_lsq = theta_S_norm > zero(T) && J_norm_initial > obj_tol
    return -_least_squares_solve(A, b; rank_tol = rank_tol,
                                 prefer_direct = use_direct_lsq)
end

function _spectrum_diagnostics(D, J_up; rank_tol)
    rank_J = try
        _matrix_rank(J_up; rank_tol = rank_tol)
    catch err
        _stderr_warn("Unable to compute rank(J_up): $err")
        nothing
    end
    rank_D = try
        _matrix_rank(D; rank_tol = rank_tol)
    catch err
        _stderr_warn("Unable to compute rank(D): $err")
        nothing
    end

    eigvals_J_up = try
        eigvals(J_up)
    catch err
        _stderr_warn("Unable to compute eigvals(J_up): $err")
        nothing
    end

    spectral_radius_J_up = eigvals_J_up === nothing || isempty(eigvals_J_up) ?
        nothing : maximum(abs.(eigvals_J_up))

    return rank_D, rank_J, eigvals_J_up, spectral_radius_J_up
end
