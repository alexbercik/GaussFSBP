"""
    OptimizedOperatorBuilders.jl

Optimization-based construction of one-dimensional diagonal-norm FSBP
operators when the quadrature nodes and weights are already known.

The public entry point is [`optimize_fsbp_operator`](@ref).  It samples the
exactness basis at the supplied nodes, enforces the SBP/exactness constraints,
and then minimizes weighted extrapolation, derivative, and Frobenius/norm
surrogate objectives without changing the quadrature rule.
"""

"""
    optimize_fsbp_operator(x, w, xL, xR, basis, quad_basis; kwargs...) -> FSBPOperator
    optimize_fsbp_operator(x, w, xL, xR, basis_functions, basis_derivatives, quad_basis; kwargs...) -> FSBPOperator

Construct a 1-D diagonal-norm FSBP operator for known nodes `x`, positive
quadrature weights `w`, and interval endpoints `xL`, `xR`.

The exactness space can be supplied either as an `AbstractBasis` with analytic
derivatives, or as paired vectors of value and derivative callables.  The
builder samples `V`, `Vx`, `vL`, and `vR`, checks that `V` has full column rank,
and returns an [`FSBPOperator`](@ref).

# Optimization Paths
- `opt_method=:sequential` uses the original two-stage optimization.  It first
  builds exact boundary extrapolation parametrizations, optimizes `tL` and `tR`
  independently or with flip symmetry, forms `E = tR*tR' - tL*tL'`, then solves
  the reduced Marchildon-Zingg skew system for `S` and optimizes any remaining
  skew nullspace.
- `opt_method=:simultaneous` is the default.  It builds one reduced linear
  exactness system for the coupled unknowns `(S,tL,tR)`, computes a particular
  solution and nullspace, and then runs a nonlinear least-squares optimization
  over the coupled nullspace coordinates.  This keeps extrapolation and skew
  freedom coupled during the final objective minimization.  If both endpoints
  are fixed quadrature nodes, this path automatically falls back to the
  sequential routine.

Set `extrapolation_symmetry=:flip` to enforce `tR == reverse(tL)`.  This
requires reflection-symmetric nodes and weights and is supported by both
optimization paths.  The default `:none` treats the two boundaries separately.

# General Keyword Arguments
- `test_functions=Function[]` — additional functions used in extrapolation and
  derivative accuracy objectives.  May be a `FunctionBasis` or a collection of
  callables.
- `test_derivatives=Function[]` — derivatives of `test_functions`.  If
  `test_functions` is a `FunctionBasis` and this is empty, the basis derivatives
  are used.
- `test_weights=nothing` — non-negative weights for the test-function objectives;
  `nothing` uses unit weights.
- `extrapolation_objective_weights=(accuracy=1//2, norm=1//2)` — non-negative weights for
  extrapolation test accuracy and extrapolation norm objectives.
- `S_objective_weights=(accuracy=1//2, norm=1//2)` — non-negative weights for derivative
  test accuracy and upwind Jacobian Frobenius norm objectives.
- `extrapolation_norm::Symbol=:Hinv` — the weighted norm to use for tL, tR.
  Allowed values are `:Hinv`, `:H`, `:Euclidean`.
- `extrapolation_symmetry::Symbol=:none` — `:none` for independent boundary
  extrapolation; `:flip` for reflection-symmetric extrapolation.
- `derivative_error_norm::Symbol=:H` — the weighted norm to use for derivative
  accuracy tests.  Allowed values are `:Hinv`, `:H`, `:Euclidean`.
- `zero_boundary_scaling::Symbol=:fallback` — handling for zero boundary test
  scales: `:fallback` uses a nonzero reference scale; `:omit` drops that row.
- `rank_tol=nothing` — rank tolerance for Vandermonde, pseudoinverse, and
  nullspace computations.
- `compatibility_tol=nothing` — tolerance for the quadrature/SBP compatibility
  residual.  `nothing` selects a type-scaled default.
- `compatibility_action::Symbol=:warn` — action when compatibility exceeds the
  tolerance: `:warn`, `:error`, or `:ignore`.
- `extrapolation_scale_tol=nothing`, `derivative_scale_tol=nothing`,
  `objective_tol=nothing` — tolerances controlling objective block scaling and
  omission.  `nothing` selects defaults based on the working scalar type.
- `verbose::Bool=false` — print optimization diagnostics and final checks.

# Method Keyword Arguments
- `opt_method::Symbol=:simultaneous` — choose `:simultaneous` or `:sequential`.
- `simultaneous_nonlinear_solver::Symbol=:levenberg_marquardt` — local
  nonlinear least-squares backend for the simultaneous path.  Allowed values
  are `:levenberg_marquardt`, `:gauss_newton`, and `:auto`.
- `simultaneous_init::Symbol=:minimum_norm` — baseline initial point for the
  simultaneous path: `:minimum_norm` uses the coupled minimum-norm solution,
  while `:sequential` also projects the sequential solution into the coupled
  affine space.
- `simultaneous_num_starts::Int=10` — number of selected starts passed to the
  local nonlinear solver.  Use `1` for baseline-only behavior.
- `simultaneous_global_num_candidates=nothing` — candidate budget for the
  state-whitened global ray search; `nothing` selects a dimension-dependent
  default.
- `simultaneous_global_norm_growth_limit=3` — radial search limit expressed as
  allowed growth in the state/norm-objective scale.
- `simultaneous_global_radial_steps::Int=2` — number of shallow radial trial
  steps sampled along each state-whitened ray before local optimization.
- `simultaneous_local_min_tol=nothing` — tolerance for reporting distinct
  local minima or start sensitivity.
- `simultaneous_max_iter::Int=1000` — maximum local nonlinear-solver iterations.
- `simultaneous_step_tol=nothing`, `simultaneous_grad_tol=nothing`,
  `simultaneous_obj_tol=nothing` — local solver stopping tolerances.  `nothing`
  selects defaults based on the working scalar type.

# Implementation Layout
Both public methods share the same pipeline:
1. [`_optimize_fsbp_preamble`](@ref) validates inputs, keyword values, test
   functions, and the working scalar type.
2. The public method samples `V`, `Vx`, `vL`, and `vR` using either basis APIs
   or raw callables.
3. [`_optimize_fsbp_operator_core`](@ref) dispatches to the selected
   optimization path.
"""
# ── Public entry points (two ways to supply the exactness basis) ─────────────

"""
    optimize_fsbp_operator(x, w, xL, xR, basis::AbstractBasis, quad_basis; kwargs...)

`AbstractBasis` entry: sample `V`/`Vx`/`vL`/`vR` via `eval_basis_*` (same as the
exact construction path in `build_fsbp_operator`), then run the shared core.
"""
function optimize_fsbp_operator(x, w, xL, xR, basis::AbstractBasis, quad_basis::FunctionBasis;
                                kwargs...)
    basis isa FunctionBasis &&
        _require_function_basis_intervals_match(basis, quad_basis,
                                                "optimize_fsbp_operator")
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
    optimize_fsbp_operator(x, w, xL, xR, basis_functions, basis_derivatives, quad_basis; kwargs...)

Callable-vector entry: wraps `funcs`/`derivs` as evaluators, samples via
[`_sample_matrix`](@ref) / [`_sample_vector`](@ref), then runs the shared core.
"""
function optimize_fsbp_operator(x, w, xL, xR,
                                basis_functions,
                                basis_derivatives,
                                quad_basis::FunctionBasis;
                                kwargs...)
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
    _require_function_basis_intervals_match(op_basis, quad_basis,
                                          "optimize_fsbp_operator")
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
    extrapolation_symmetry = get(kwargs, :extrapolation_symmetry, :none)
    opt_method = get(kwargs, :opt_method, :simultaneous)
    simultaneous_nonlinear_solver = get(kwargs, :simultaneous_nonlinear_solver, :levenberg_marquardt)
    simultaneous_init = get(kwargs, :simultaneous_init, :minimum_norm)
    compatibility_action = get(kwargs, :compatibility_action, :warn)
    simultaneous_num_starts = get(kwargs, :simultaneous_num_starts, 10)
    simultaneous_max_iter = get(kwargs, :simultaneous_max_iter, 1000)
    simultaneous_global_num_candidates = get(kwargs, :simultaneous_global_num_candidates, nothing)
    simultaneous_global_norm_growth_limit = get(kwargs, :simultaneous_global_norm_growth_limit, 3)
    simultaneous_global_radial_steps = get(kwargs, :simultaneous_global_radial_steps, 2)

    K > 0 || throw(ArgumentError("The basis must contain at least one function."))
    _validate_norm_symbol(extrapolation_norm, (:Hinv, :H, :Euclidean, :Frobenius),
                          "extrapolation_norm")
    _validate_norm_symbol(derivative_error_norm, (:Hinv, :H, :Euclidean, :Frobenius),
                          "derivative_error_norm")
    zero_boundary_scaling in (:fallback, :omit) ||
        throw(ArgumentError("zero_boundary_scaling must be :fallback or :omit."))
    extrapolation_symmetry in (:none, :flip) ||
        throw(ArgumentError("extrapolation_symmetry must be :none or :flip."))
    opt_method in (:simultaneous, :sequential) ||
        throw(ArgumentError("opt_method must be :simultaneous or :sequential, got $opt_method."))
    simultaneous_nonlinear_solver in (:levenberg_marquardt, :gauss_newton, :auto) ||
        throw(ArgumentError("simultaneous_nonlinear_solver must be :levenberg_marquardt, :gauss_newton, or :auto."))
    simultaneous_init in (:minimum_norm, :sequential) ||
        throw(ArgumentError("simultaneous_init must be :minimum_norm or :sequential."))
    compatibility_action in (:warn, :error, :ignore) ||
        throw(ArgumentError("compatibility_action must be :warn, :error, or :ignore."))
    simultaneous_num_starts >= 1 ||
        throw(ArgumentError("simultaneous_num_starts must be at least 1."))
    simultaneous_max_iter >= 0 ||
        throw(ArgumentError("simultaneous_max_iter must be nonnegative."))
    (simultaneous_global_num_candidates === nothing || simultaneous_global_num_candidates >= 0) ||
        throw(ArgumentError("simultaneous_global_num_candidates must be nonnegative or nothing."))
    simultaneous_global_norm_growth_limit > 1 ||
        throw(ArgumentError("simultaneous_global_norm_growth_limit must be greater than 1."))
    simultaneous_global_radial_steps >= 1 ||
        throw(ArgumentError("simultaneous_global_radial_steps must be at least 1."))

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
                                      opt_method::Symbol = :simultaneous,
                                      simultaneous_nonlinear_solver::Symbol = :levenberg_marquardt,
                                      simultaneous_init::Symbol = :minimum_norm,
                                      simultaneous_num_starts::Int = 10,
                                      simultaneous_global_num_candidates = nothing,
                                      simultaneous_global_norm_growth_limit = 3,
                                      simultaneous_global_radial_steps::Int = 2,
                                      simultaneous_local_min_tol = nothing,
                                      simultaneous_max_iter::Int = 1000,
                                      simultaneous_step_tol = nothing,
                                      simultaneous_grad_tol = nothing,
                                      simultaneous_obj_tol = nothing,
                                      kwargs...)
    if opt_method === :sequential
        return _optimize_fsbp_operator_sequential_core(
            setup, V, Vx, vL, vR, op_basis, quad_basis; kwargs...)
    end

    return _optimize_fsbp_operator_simultaneous_core(
        setup, V, Vx, vL, vR, op_basis, quad_basis;
        simultaneous_nonlinear_solver = simultaneous_nonlinear_solver,
        simultaneous_init = simultaneous_init,
        simultaneous_num_starts = simultaneous_num_starts,
        simultaneous_global_num_candidates = simultaneous_global_num_candidates,
        simultaneous_global_norm_growth_limit = simultaneous_global_norm_growth_limit,
        simultaneous_global_radial_steps = simultaneous_global_radial_steps,
        simultaneous_local_min_tol = simultaneous_local_min_tol,
        simultaneous_max_iter = simultaneous_max_iter,
        simultaneous_step_tol = simultaneous_step_tol,
        simultaneous_grad_tol = simultaneous_grad_tol,
        simultaneous_obj_tol = simultaneous_obj_tol,
        kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Sequential decoupled optimization (first tL/tR, then S)
# ─────────────────────────────────────────────────────────────────────────────

function _optimize_fsbp_operator_sequential_core(setup, V, Vx, vL, vR, op_basis, quad_basis;
                                      test_weights = nothing,
                                      extrapolation_objective_weights = (accuracy = 1//2, norm = 1//2),
                                      S_objective_weights = (accuracy = 1//2, norm = 1//2),
                                      extrapolation_norm::Symbol = :Hinv,
                                      extrapolation_symmetry::Symbol = :none,
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
    theta_ext_acc, theta_ext_norm = _extract_objective_weights(
        T, extrapolation_objective_weights, "extrapolation_objective_weights")
    theta_S_acc, theta_S_norm = _extract_objective_weights(
        T, S_objective_weights, "S_objective_weights")

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
        println("  opt_method = :sequential")
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

    if extrapolation_symmetry === :flip
        _check_flip_symmetric_grid(x, w, xL, xR)
        tL0, tR0, Zflip = _build_flip_symmetric_extrapolation(
            V, w, vL, vR, x, xL, xR, left_endpoint_idx, right_endpoint_idx,
            extrapolation_norm; rank_tol = rank_tol)
        nL = size(Zflip, 2)
        nR = nL
        tL_has_free_parameters = nL > 0
        tR_has_free_parameters = tL_has_free_parameters
    else
        ZL = left_is_endpoint ? zeros(T, N, 0) : ZV
        ZR = right_is_endpoint ? zeros(T, N, 0) : ZV
        nL = size(ZL, 2)
        nR = size(ZR, 2)
        tL_has_free_parameters = nL > 0
        tR_has_free_parameters = nR > 0
        # -- Compute initial tL/tR vectors OR get the exact tL=e1, tR=eN
        tL0, tR0 = _build_extrapolation(V, x, w, vL, vR, xL, xR, extrapolation_norm)
    end

    if verbose
        println("  extrapolation symmetry = $extrapolation_symmetry")
        if extrapolation_symmetry === :flip
            println("  num free coupled tL/tR params = $nL")
        else
            println("  num free tL params = $nL")
            println("  num free tR params = $nR")
        end
        println("\n")
    end

    if tL_has_free_parameters || tR_has_free_parameters || verbose
        # -- Prepare the tests for the extrapolation optimization using orthogonalized test space
        ext_tests = _build_extrapolation_tests(test_samples, weights, w,
                                               zero_boundary_scaling, ext_scale_tol)
        J_ext_L_initial = _tL_extrapolation_objectives(tL0, ext_tests, w, extrapolation_norm)
        J_ext_R_initial = _tR_extrapolation_objectives(tR0, ext_tests, w, extrapolation_norm)
    else
        ext_tests = NamedTuple[]
    end

    if tL_has_free_parameters || tR_has_free_parameters
        if extrapolation_symmetry === :flip
            tL, tR = _optimize_flip_symmetric_extrapolation(
                tL0, Zflip, ext_tests, w, extrapolation_norm,
                theta_ext_acc, theta_ext_norm, J_ext_L_initial, J_ext_R_initial,
                obj_tol; rank_tol = rank_tol)
        else
            tL, tR = _optimize_extrapolation(tL0, tR0, ZL, ZR, ext_tests, w,
                                             extrapolation_norm,
                                             theta_ext_acc, theta_ext_norm,
                                             J_ext_L_initial, J_ext_R_initial,
                                             obj_tol;
                                             tL_has_free_parameters = tL_has_free_parameters,
                                             tR_has_free_parameters = tR_has_free_parameters,
                                             rank_tol = rank_tol)
        end
    else
        tL, tR = tL0, tR0
    end

    if verbose
        J_ext_L_final = if tL_has_free_parameters
            _tL_extrapolation_objectives(tL, ext_tests, w, extrapolation_norm)
        else
            J_ext_L_initial
        end
        J_ext_R_final = if tR_has_free_parameters
            _tR_extrapolation_objectives(tR, ext_tests, w, extrapolation_norm)
        else
            J_ext_R_initial
        end
        ext_exact_L = _euclidean_norm(V' * tL - vL)
        ext_exact_R = _euclidean_norm(V' * tR - vR)

        println("After extrapolation stage:")
        if extrapolation_symmetry === :flip
            println("  coupled tL/tR optimization active = $tL_has_free_parameters")
        else
            println("  tL optimization active = $tL_has_free_parameters")
            println("  tR optimization active = $tR_has_free_parameters")
        end
        println("  norm backend = $extrapolation_norm")
        println("  initial tL accuracy objective = $(J_ext_L_initial.accuracy)")
        println("    final tL accuracy objective = $(J_ext_L_final.accuracy)")
        println("            ratio final/initial = $(_objective_ratio(J_ext_L_final.accuracy, J_ext_L_initial.accuracy))")
        println("  initial tR accuracy objective = $(J_ext_R_initial.accuracy)")
        println("    final tR accuracy objective = $(J_ext_R_final.accuracy)")
        println("            ratio final/initial = $(_objective_ratio(J_ext_R_final.accuracy, J_ext_R_initial.accuracy))")
        println("  initial tL norm objective = $(J_ext_L_initial.norm)")
        println("    final tL norm objective = $(J_ext_L_final.norm)")
        println("        ratio final/initial = $(_objective_ratio(J_ext_L_final.norm, J_ext_L_initial.norm))")
        println("  initial tR norm objective = $(J_ext_R_initial.norm)")
        println("    final tR norm objective = $(J_ext_R_final.norm)")
        println("        ratio final/initial = $(_objective_ratio(J_ext_R_final.norm, J_ext_R_initial.norm))")
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
    #    so that S = S0 + ZC * a, and we solve the optimization problem for a.
    s0 = _pseudoinverse_solve(C, d; rank_tol = rank_tol)
    ZC = _nullspace_basis(C; rank_tol = rank_tol)
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

    nfreeS = size(ZC, 2)
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
        S_basis = [_unvec_skew(ZC[:, j], N, pairs) for j in 1:nfreeS]
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
        println("\n")
    end

    return FSBPOperator{T}(D, Diagonal(w), Q, S, E, tL, tR, x, w,
                         op_basis, quad_basis, (xL, xR), N, K)
end

# ─────────────────────────────────────────────────────────────────────────────
# Simultaneous coupled optimization (tL, tR, and S together)
# ─────────────────────────────────────────────────────────────────────────────

function _optimize_fsbp_operator_simultaneous_core(setup, V, Vx, vL, vR, op_basis, quad_basis;
                                                   test_weights = nothing,
                                                   extrapolation_objective_weights = (accuracy = 1//2, norm = 1//2),
                                                   S_objective_weights = (accuracy = 1//2, norm = 1//2),
                                                   extrapolation_norm::Symbol = :Hinv,
                                                   extrapolation_symmetry::Symbol = :none,
                                                   derivative_error_norm::Symbol = :H,
                                                   zero_boundary_scaling::Symbol = :fallback,
                                                   rank_tol = nothing,
                                                   compatibility_tol = nothing,
                                                   compatibility_action::Symbol = :warn,
                                                   extrapolation_scale_tol = nothing,
                                                   derivative_scale_tol = nothing,
                                                   objective_tol = nothing,
                                                   verbose::Bool = false,
                                                   simultaneous_nonlinear_solver::Symbol = :levenberg_marquardt,
                                                   simultaneous_init::Symbol = :minimum_norm,
                                                   simultaneous_num_starts::Int = 10,
                                                   simultaneous_global_num_candidates = nothing,
                                                   simultaneous_global_norm_growth_limit = 3,
                                                   simultaneous_global_radial_steps::Int = 2,
                                                   simultaneous_local_min_tol = nothing,
                                                   simultaneous_max_iter::Int = 1000,
                                                   simultaneous_step_tol = nothing,
                                                   simultaneous_grad_tol = nothing,
                                                   simultaneous_obj_tol = nothing)
    (; x, w, xL, xR, N, T, test_funcs, test_derivs, K) = setup
    sequential_kwargs = (;
        test_weights,
        extrapolation_objective_weights,
        S_objective_weights,
        extrapolation_norm,
        extrapolation_symmetry,
        derivative_error_norm,
        zero_boundary_scaling,
        rank_tol,
        compatibility_tol,
        compatibility_action,
        extrapolation_scale_tol,
        derivative_scale_tol,
        objective_tol,
        verbose,
    )

    # -- Determine if the left and right endpoints are endpoints of the interval
    left_endpoint_idx = _endpoint_node_index(x, xL)
    right_endpoint_idx = _endpoint_node_index(x, xR)
    left_is_endpoint = left_endpoint_idx !== nothing
    right_is_endpoint = right_endpoint_idx !== nothing
    if extrapolation_symmetry === :flip
        _check_flip_symmetric_grid(x, w, xL, xR)
    end
    if left_is_endpoint && right_is_endpoint
        verbose && println("# -- Both endpoints are fixed; falling back to sequential optimization.")
        return _optimize_fsbp_operator_sequential_core(
            setup, V, Vx, vL, vR, op_basis, quad_basis; sequential_kwargs...)
    end

    rankV = _check_vandermonde_rank(V, K, T; rank_tol,
                                              context = "optimize_fsbp_operator")

    # -- Extract the optimization weights
    #    _acc weights the accuracy on the test_funcs (either extrapolation or derivative)
    #    _norm weights the norm of the operators (tL/tR or A = D + H^(-1) tL tL^T)
    theta_ext_acc, theta_ext_norm = _extract_objective_weights(
        T, extrapolation_objective_weights, "extrapolation_objective_weights")
    theta_S_acc, theta_S_norm = _extract_objective_weights(
        T, S_objective_weights, "S_objective_weights")

    ext_scale_tol = extrapolation_scale_tol === nothing ? sqrt(eps(T)) : T(extrapolation_scale_tol)
    der_scale_tol = derivative_scale_tol === nothing ? sqrt(eps(T)) : T(derivative_scale_tol)
    obj_tol = objective_tol === nothing ? sqrt(eps(T)) : T(objective_tol)
    sim_obj_tol = simultaneous_obj_tol === nothing ? eps(T) : T(simultaneous_obj_tol)
    sim_step_tol = simultaneous_step_tol === nothing ? sqrt(eps(T)) : T(simultaneous_step_tol)
    sim_grad_tol = simultaneous_grad_tol === nothing ? sqrt(eps(T)) : T(simultaneous_grad_tol)
    sim_local_min_tol = simultaneous_local_min_tol === nothing ?
        10*sqrt(eps(T)) : T(simultaneous_local_min_tol)

    # -- Prepare the test functions (and their relative weights) by H-orthogonalizing w.r.t. V
    weights = _test_weights(T, length(test_funcs), test_weights)
    M = _exactness_gram_matrix(V, w) # modal mass matrix
    test_samples = _precompute_test_orthogonal_samples(
        test_funcs, test_derivs, x, xL, xR, V, Vx, w, vL, vR, M)

    # -- Basis of ker(V') when N > K (used for parametrizing the optimization).
    dim_null_V = N - K
    ZV = dim_null_V > 0 ? _nullspace_basis(V'; rank_tol = rank_tol) :
                           zeros(T, N, 0)

    if extrapolation_symmetry === :flip
        tL0, tR0, Zflip = _build_flip_symmetric_extrapolation(
            V, w, vL, vR, x, xL, xR, left_endpoint_idx, right_endpoint_idx,
            extrapolation_norm; rank_tol = rank_tol)
        nT = size(Zflip, 2)
        nL = nT
        nR = nT
        BL = Zflip
        BR = reverse(Zflip; dims = 1)
        tL_has_free_parameters = nT > 0
        tR_has_free_parameters = nT > 0
    else
        ZL = left_is_endpoint ? zeros(T, N, 0) : ZV
        ZR = right_is_endpoint ? zeros(T, N, 0) : ZV
        nL = size(ZL, 2)
        nR = size(ZR, 2)
        nT = nL + nR
        BL = zeros(T, N, nT)
        BR = zeros(T, N, nT)
        nL > 0 && (BL[:, 1:nL] .= ZL)
        nR > 0 && (BR[:, (nL + 1):(nL + nR)] .= ZR)
        # -- Compute initial tL/tR vectors OR get the exact tL=e1, tR=eN
        tL0, tR0 = _build_extrapolation(V, x, w, vL, vR, xL, xR, extrapolation_norm)
        tL_has_free_parameters = nL > 0
        tR_has_free_parameters = nR > 0
    end

    # -- Prepare the tests for the extrapolation optimization using orthogonalized test space
    ext_tests = _build_extrapolation_tests(test_samples, weights, w,
                                           zero_boundary_scaling, ext_scale_tol)
    J_ext_L_initial = _tL_extrapolation_objectives(tL0, ext_tests, w, extrapolation_norm)
    J_ext_R_initial = _tR_extrapolation_objectives(tR0, ext_tests, w, extrapolation_norm)

    if verbose
        println("\n")
        println("Optimization-based FSBP construction")
        println("  opt_method = :simultaneous")
        println("  num of nodes = $N")
        println("  dim of basis = $K")
        println("  num of test funcs = $(length(test_funcs))")
        println("  rank(V) = $rankV")
        println("  dim null(V^T) = $(N - K)")
        println("  extrapolation symmetry = $extrapolation_symmetry")
        if extrapolation_symmetry === :flip
            println("  num free coupled tL/tR params = $nT")
        else
            println("  num free tL params = $nL")
            println("  num free tR params = $nR")
        end
        println("\n")
    end

    # -- Get ready for optimization: check the SBP compatibility first
    compat_residual_matrix = _sbp_compatibility_residual(V, Vx, w, vL, vR)
    compat_residual = _frobenius_norm(compat_residual_matrix)
    compat_scale = max(one(T), _frobenius_norm(V' * _scale_rows(Vx, w)),
                         _frobenius_norm(vR * vR' - vL * vL'))
    compat_tol_eff = compatibility_tol === nothing ?
        T(100) * sqrt(eps(T)) * compat_scale : T(compatibility_tol)
    verbose && println("Quadrature/SBP compatibility residual = $compat_residual")
    if compat_residual > compat_tol_eff
        msg = "Quadrature/SBP compatibility residual $compat_residual exceeds tolerance $compat_tol_eff; exact construction of S may be impossible."
        if compatibility_action === :error
            error(msg)
        elseif compatibility_action === :warn
            _stderr_warn(msg)
        end
    end

    # -- Construct the linear system for optimization
    pairs = _skew_pairs(N) # vector of index pairs (i, j) with i < j for S[i,j]
    # assemble S V = H Vₓ - ½ E V into the coupled linear system C [s; a] = d
    C, d = _assemble_coupled_skew_extrapolation_system(
        V, Vx, w, ZV, BL, BR, tL0, tR0, vL, vR, pairs, T)
    # C should be size (nE, nS + nT), where
    #   nS = N * (N - 1) / 2  (number of independent entries in S)
    #   nT = number of free parameters in tL and tR
    #   nE = K * (K - 1) / 2 + dim_null_V * K  (number of independent equations)
    nS = length(pairs)
    nvars = nS + nT
    nE = length(d)
    expected_nS = N * (N - 1) ÷ 2
    expected_nE = K * (K - 1) ÷ 2 + dim_null_V * K
    if nS != expected_nS
        _stderr_warn("n_S = $nS but expected N(N-1)/2 = $expected_nS for N = $N.")
    end
    if nE != expected_nE
        _stderr_warn("n_E = $nE but expected K(K-1)/2 + dim_null_V * K = $expected_nE for K = $K and dim_null_V = $dim_null_V.")
    end
    size(C) == (expected_nE, nvars) ||
        _stderr_warn("size(C) = $(size(C)) but expected ($expected_nE, $nvars).")

    # -- Get the minimum-norm solution x0 = [s0; a0] = pinv(C) d
    #    We then parametrize all possible S matrices and tL/tR vectors through the nullspace basis ZC of C,
    #    so that x = x0 + ZC * y, and we solve the optimization problem for y.
    #    we then recover S = _unvec_skew(x[1:nS]) and tL = tL0 + BL x[nS+1:end] , tR = tR0 + BR x[nS+1:end]
    x0 = _pseudoinverse_solve(C, d; rank_tol = rank_tol)
    ZC = _nullspace_basis(C; rank_tol = rank_tol)
    nfree = size(ZC, 2)

    # -- Create a named tuple for the initial state, then construct the full initial states
    data0 = (;
        T, N, pairs, nS, nT, x0, ZC, tL0, tR0, BL, BR, w,
    )
    state0 = _simultaneous_state(zeros(T, nfree), data0)
    # -- Prepare the tests for the S optimization using orthogonalized test space
    der_tests = _build_derivative_tests(test_samples, weights, w, der_scale_tol)
    J_der_initial = _derivative_objective(state0.D, der_tests, w, derivative_error_norm)
    J_norm_initial = _full_jacobian_objective(state0.J_up)
    data = merge(data0, (;
                ext_tests,
                der_tests,
                theta_ext_acc,
                theta_ext_norm,
                theta_S_acc,
                theta_S_norm,
                J_ext_L_initial,
                J_ext_R_initial,
                J_der_initial,
                J_norm_initial,
                extrapolation_norm,
                derivative_error_norm,
                obj_tol,
                tL_has_free_parameters,
                tR_has_free_parameters,
            ))

    if verbose
        rankC = _matrix_rank(C; rank_tol = rank_tol)
        nullC = nS + nT - rankC
        half = one(T) / T(2)
        C_residual = _euclidean_norm(C * x0 - d)
        x0_accuracy_residual = _frobenius_norm(state0.S * V - (_scale_rows(Vx, w) - half * state0.E * V))

        println("Before simultaneous nonlinear optimization:")
        println("  # of independent S entries = $nS")
        println("  # of independent equations = $nE")
        println("  # of free tL/tR parameters = $nT")
        println("  # of actual free parameters = $nfree")
        println("  size(C) = $(size(C))")
        println("  rank(C) = $rankC")
        println("  null(C) = $nullC")
        println("  The following residuals test the exactness of the initial S0, tL0, tR0 solution:")
        println("  ||C x_0 - d|| = $C_residual")
        println("  ||S_0 V - (H V_x - (1/2) E_0 V)|| = $x0_accuracy_residual")
        println("\n")
    end

    y_final = zeros(T, nfree) # initialize the solution coefficients for the ZC nullspace basis
    results = NamedTuple[]
    if nfree == 0
        verbose && println("  coupled nullspace dimension is zero; skipping nonlinear optimization.")
    else
        # -- initialize the starting points for the nonlinear optimization
        #    first include the minimum norm solution, then optionally include the sequential solution
        base_starts = [zeros(T, nfree)]
        if simultaneous_init === :sequential
            seq_init = _simultaneous_sequential_initial_y(
                setup, V, Vx, vL, vR, op_basis, quad_basis, data;
                rank_tol = rank_tol,
                sequential_kwargs = merge(sequential_kwargs, (; verbose = false)))
            push!(base_starts, seq_init)
        end

        residual = y -> _simultaneous_residual(y, data)
        # -- generate and screen the starting points for the nonlinear optimization
        starts, start_info = _simultaneous_global_search_starts(
            base_starts, simultaneous_num_starts, T, data, residual;
            global_num_candidates = simultaneous_global_num_candidates,
            norm_growth_limit = T(simultaneous_global_norm_growth_limit),
            radial_steps = simultaneous_global_radial_steps)
        verbose && _print_simultaneous_start_summary(start_info)
        # -- run the least squares optimization for each start point
        for (idx, ystart) in enumerate(starts)
            result = try
                _solve_nonlinear_least_squares(
                    residual, ystart, T;
                    solver = simultaneous_nonlinear_solver,
                    max_iter = simultaneous_max_iter,
                    step_tol = sim_step_tol,
                    grad_tol = sim_grad_tol,
                    obj_tol = sim_obj_tol,
                    rank_tol = rank_tol)
            catch err
                err isa InterruptException && rethrow()
                _stderr_warn("simultaneous nonlinear start $idx failed: $err")
                (y = copy(ystart), objective = T(Inf), converged = false,
                 status = :failed, iterations = 0)
            end
            push!(results, result)
        end

        finite_results = [r for r in results if isfinite(r.objective)]
        if isempty(finite_results)
            _stderr_warn("all simultaneous nonlinear starts failed; falling back to sequential optimization.")
            return _optimize_fsbp_operator_sequential_core(
                setup, V, Vx, vL, vR, op_basis, quad_basis; sequential_kwargs...)
        else
            best_idx = argmin([r.objective for r in finite_results])
            y_final = finite_results[best_idx].y
        end

        if verbose
            _print_simultaneous_nonlinear_summary(results, sim_local_min_tol, T)
        end
    end

    final_state = _simultaneous_state(y_final, data)

    if verbose
        ext_exact_L = _euclidean_norm(V' * final_state.tL - vL)
        ext_exact_R = _euclidean_norm(V' * final_state.tR - vR)
        derivative_residual = _frobenius_norm(final_state.D * V - Vx)
        sbp_residual = _frobenius_norm(final_state.Q + final_state.Q' - final_state.E)

        rank_D, rank_J, eigvals_J, spectral_radius_J =
            _spectrum_diagnostics(final_state.D, final_state.J_up; rank_tol = rank_tol)
        J_ext_L_final = _tL_extrapolation_objectives(final_state.tL, ext_tests, w, extrapolation_norm)
        J_ext_R_final = _tR_extrapolation_objectives(final_state.tR, ext_tests, w, extrapolation_norm)
        J_der_final = _derivative_objective(final_state.D, der_tests, w, derivative_error_norm)
        J_norm_final = _full_jacobian_objective(final_state.J_up)

        println("After simultaneous optimization:")
        println("  initial tL accuracy objective = $(J_ext_L_initial.accuracy)")
        println("  final tL accuracy objective = $(J_ext_L_final.accuracy)")
        println("  tL accuracy ratio final/initial = $(_objective_ratio(J_ext_L_final.accuracy, J_ext_L_initial.accuracy))")
        println("  initial tR accuracy objective = $(J_ext_R_initial.accuracy)")
        println("  final tR accuracy objective = $(J_ext_R_final.accuracy)")
        println("  tR accuracy ratio final/initial = $(_objective_ratio(J_ext_R_final.accuracy, J_ext_R_initial.accuracy))")
        println("  initial tL norm objective = $(J_ext_L_initial.norm)")
        println("  final tL norm objective = $(J_ext_L_final.norm)")
        println("  tL norm ratio final/initial = $(_objective_ratio(J_ext_L_final.norm, J_ext_L_initial.norm))")
        println("  initial tR norm objective = $(J_ext_R_initial.norm)")
        println("  final tR norm objective = $(J_ext_R_final.norm)")
        println("  tR norm ratio final/initial = $(_objective_ratio(J_ext_R_final.norm, J_ext_R_initial.norm))")
        println("  initial derivative objective = $J_der_initial")
        println("  final derivative objective = $J_der_final")
        println("  derivative ratio final/initial = $(_objective_ratio(J_der_final, J_der_initial))")
        println("  initial Jacobian Frobenius objective = $J_norm_initial")
        println("  final Jacobian Frobenius objective = $J_norm_final")
        println("  Jacobian ratio final/initial = $(_objective_ratio(J_norm_final, J_norm_initial))")
        println("  ||V^T tL - v_L|| = $ext_exact_L")
        println("  ||V^T tR - v_R|| = $ext_exact_R")
        println("  ||D V - V_x|| = $derivative_residual")
        println("  ||Q + Q^T - E|| = $sbp_residual")
        println("  rank(D) = $rank_D")
        println("  rank(D + H⁻¹ tL tLᵀ) = $rank_J")
        println("  spectral radius (D + H⁻¹ tL tLᵀ) = $spectral_radius_J")
        if eigvals_J !== nothing && !isempty(eigvals_J)
            re_J = real.(eigvals_J)
            println("  Re(λ) range (D + H⁻¹ tL tLᵀ) = [$(minimum(re_J)), $(maximum(re_J))]")
        end
        println("\n")
    end

    return FSBPOperator{T}(final_state.D, Diagonal(w), final_state.Q, final_state.S,
                           final_state.E, final_state.tL, final_state.tR, x, w,
                           op_basis, quad_basis, (xL, xR), N, K)
end

"""
    _assemble_coupled_skew_extrapolation_system(V, Vx, w, ZV, BL, BR,
        tL0, tR0, vL, vR, pairs, T) -> (C, d)

Build the coupled linear system `C x = d` for simultaneous optimization of the
skew matrix `S` and boundary extrapolation vectors `(tL, tR)`.

`S` is stored by its upper-triangle entries `s` (index layout from `pairs`).
Boundary vectors are not optimized nodally; they are affine in a single
coefficient vector `a` (columns of `BL`, `BR`):

    tL = tL0 + BL*a,    tR = tR0 + BR*a,    E = tR tR' - tL tL'

`BL` and `BR` are `N × nT` with `nT = size(BL, 2) = size(BR, 2)` (built upstream
from `extrapolation_symmetry`):
- **`:none`** (`nT = nL + nR`): independent left/right parameters.  `BL[:, 1:nL]`
  spans free left directions (typically `ker(V')` when the left endpoint is not
  a node; otherwise `nL = 0` and `tL = tL0`).  `BR[:, nL+1:nL+nR]` is analogous
  on the right.  Perturbations satisfy `V' BL = 0` and `V' BR = 0`, so
  `V' tL = vL` and `V' tR = vR` are preserved.
- **`:flip`** (`nT = nL = nR`): one shared coefficient vector enforces
  `tR = reverse(tL)` via `BL = Zflip`, `BR = reverse(Zflip; dims = 1)`, with
  `Zflip` a basis of `ker([V'; reverse(V')])` (flip exactness on both boundaries).
  If both endpoints are quadrature nodes, `nT = 0`, `tL0`/`tR0` are fixed unit
  pulses at the paired endpoint indices, and `tR0 = reverse(tL0)`.

The unknown vector is `x = [s; a]` (`nvars = nS + nT`).  For nodal vectors `u`, `v`,
the bilinear form `u' S v` is linear in `s`; `_skew_bilinear_row` supplies the
corresponding columns of `C`.

Start from the full SBP relation (Marchildon-Zingg), with `E` depending on `(tL, tR)`:

    S V = H Vₓ - ½ E V,    E = tR tR' - tL tL'

Separate into two blocks (same structure as [`_assemble_independent_skew_system`](@ref),
but `E` is variable and `(tL, tR)` enter through `a`):

    Vᵀ S V  = ½(Vᵀ H Vₓ - Vₓᵀ H V)                       (compatibility; no `a`)
    ZVᵀ S V = ZVᵀ H Vₓ - ½ ZVᵀ E V                       (coupled; affine in `a`)
            = ZVᵀ H Vₓ - ½ ZVᵀ tR0 vRᵀ + ½ ZVᵀ tL0 vLᵀ

1. **Basis pairs** (`V[:,α]`, `V[:,β]`, `α > β`): same as the independent skew
   system — only `s` appears; the `E` term drops out of the projected compatibility
   equations on the exactness space.
2. **Nullspace × basis** (`ZV[:,μ]`, `V[:,β]`): `E` contributes through
   `½ z' E b` with `z = ZV[:,μ]`, `b = V[:,β]`.  Substituting `tL = tL0 + BL*a`,
   `tR = tR0 + BR*a` and moving the constant part to the rhs gives columns
   `½ (ZV' BR) vR[β] - ½ (ZV' BL) vL[β]` for `a` and
   `z' H Vₓ[:,β] - ½ (ZV' tR0) vR[β] + ½ (ZV' tL0) vL[β]` on the rhs.

The minimum-norm solution `x₀ = pinv(C) d` gives `(S₀, tL₀, tR₀)`; `ker(C)` parametrizes
the remaining free degrees of freedom optimized nonlinearly later.
"""
function _assemble_coupled_skew_extrapolation_system(V, Vx, w, ZV, BL, BR,
                                                     tL0, tR0, vL, vR, pairs,
                                                     ::Type{T}) where T
    K = size(V, 2)
    nS = length(pairs)
    nT = size(BL, 2)
    size(BR, 2) == nT ||
        throw(ArgumentError("left and right simultaneous extrapolation bases must have the same number of columns."))
    nvars = nS + nT
    rows = Vector{Vector{T}}()
    rhs = T[]
    half = one(T) / T(2)

    # -- Loop 1: for each basis pair α,β (same as _assemble_independent_skew_system)
    #   Σᵢⱼ V[i,α] V[j,β] Sᵢⱼ = Σⱼ ½ Hⱼ (V[j,α] Vₓ[j,β] - Vₓ[j,α] V[j,β])
    #   Only the skew block of x = [s; a] is nonzero; no boundary coefficients.
    for alpha in 2:K
        a = V[:, alpha]
        for beta in 1:(alpha - 1)
            b = V[:, beta]
            row = zeros(T, nvars)
            row[1:nS] .= _skew_bilinear_row(a, b, pairs, T) # leave last nT columns zero
            rhs_value = half * (dot(a, w .* Vx[:, beta]) -
                                dot(Vx[:, alpha], w .* b))
            _append_row!(rows, rhs, row, rhs_value)
        end
    end

    # -- Loop 2: for each nullspace vector μ and basis vector β
    #   Σᵢⱼ ZV[i,μ] V[j,β] Sᵢⱼ + ½ ZV[i,μ] tR(a)[i] vR[β] - ½ ZV[i,μ] tL(a)[i] vL[β] = Σᵢ ZV[i,μ] Hᵢ Vₓ[i,β]
    #   Σᵢⱼ ZV[i,μ] V[j,β] Sᵢⱼ + Σᵢ aᵢ (½ ZV[i,μ] BL[k,i] vR[β] - ½ ZV[i,μ] BR[k,i] vL[β]) = Σᵢ ZV[i,μ] Hᵢ Vₓ[i,β] - ½ ZV[i,μ] tR0[i] vR[β] + ½ ZV[i,μ] tL0[i] vL[β]
    #   LHS (C) each row is for a pair (μ,β), first nS columns is for a S pair (i,j), last nT columns is for a[i]
    #   RHS (d) each entry is for a pair (μ,β)
    ZVtBL = size(ZV, 2) == 0 || nT == 0 ? zeros(T, size(ZV, 2), nT) : ZV' * BL
    ZVtBR = size(ZV, 2) == 0 || nT == 0 ? zeros(T, size(ZV, 2), nT) : ZV' * BR
    ZVttL0 = size(ZV, 2) == 0 ? zeros(T, 0) : ZV' * tL0
    ZVttR0 = size(ZV, 2) == 0 ? zeros(T, 0) : ZV' * tR0
    for mu in 1:size(ZV, 2)
        z = ZV[:, mu]
        for beta in 1:K
            b = V[:, beta]
            row = zeros(T, nvars)
            row[1:nS] .= _skew_bilinear_row(z, b, pairs, T) # set first nS columns
            if nT > 0
                row[(nS + 1):(nS + nT)] .=                  # set last nT columns
                    half .* ZVtBR[mu, :] .* vR[beta] .-
                    half .* ZVtBL[mu, :] .* vL[beta]
            end
            rhs_value = dot(z, w .* Vx[:, beta]) -
                        half * ZVttR0[mu] * vR[beta] +
                        half * ZVttL0[mu] * vL[beta]
            _append_row!(rows, rhs, row, rhs_value)
        end
    end

    # build actual system matrix C and right-hand side d by unwrapping the stacked vectors
    return _rows_to_matrix(rows, rhs, nvars, T)
end

function _vec_skew(S, pairs, ::Type{T}) where T
    s = zeros(T, length(pairs))
    for p in eachindex(pairs)
        i, j = pairs[p]
        s[p] = T(S[i, j])
    end
    return s
end

function _simultaneous_state(y, data)
    # -- Construct the optimization nullspace vector x from the free parameters y
    xvec = data.x0 + data.ZC * y
    return _simultaneous_state_from_vector(xvec, data)
end

function _simultaneous_state_from_vector(xvec, data)
    # -- Unwrap the optimization nullspace vector xvec into the state variables
    #    S, tL, tR, E, Q, D, J_up
    T = data.T
    nS = data.nS
    nT = data.nT
    s = Vector{T}(xvec[1:nS])
    a = nT == 0 ? zeros(T, 0) : Vector{T}(xvec[(nS + 1):(nS + nT)])
    S = _unvec_skew(s, data.N, data.pairs)
    tL = copy(data.tL0)
    tR = copy(data.tR0)
    nT > 0 && (tL .+= data.BL * a)
    nT > 0 && (tR .+= data.BR * a)
    half = one(T) / T(2)
    E = tR * tR' - tL * tL'
    Q = S + half * E
    D = _divide_rows(Q, data.w)
    J_up = D + _divide_rows(tL * tL', data.w)
    return (; S, tL, tR, E, Q, D, J_up)
end

function _simultaneous_norm_residual_from_state(state, data, ::Type{T}) where T
    # Norm slice of the simultaneous residual (see _simultaneous_residual_from_state).
    residuals = T[]
    if data.theta_ext_norm > zero(T)
        norm_scale = _residual_norm_scale(data.w, data.extrapolation_norm)
        if data.tL_has_free_parameters && data.J_ext_L_initial.norm > data.obj_tol
            scale = sqrt(data.theta_ext_norm / data.J_ext_L_initial.norm)
            append!(residuals, scale .* norm_scale .* state.tL)
        end
        if data.tR_has_free_parameters && data.J_ext_R_initial.norm > data.obj_tol
            scale = sqrt(data.theta_ext_norm / data.J_ext_R_initial.norm)
            append!(residuals, scale .* norm_scale .* state.tR)
        end
    end
    if data.theta_S_norm > zero(T) && data.J_norm_initial > data.obj_tol
        scale = sqrt(data.theta_S_norm / data.J_norm_initial)
        append!(residuals, scale .* vec(state.J_up))
    end
    return residuals
end

function _simultaneous_state_norm_from_state(state, y, data, ::Type{T}) where T
    r_norm = _simultaneous_norm_residual_from_state(state, data, T)
    isempty(r_norm) || return sqrt(sum(abs2, r_norm))
    return max(one(T), _euclidean_norm(y))
end

function _simultaneous_residual_from_state(state, data)
    # Full simultaneous residual: extrapolation/derivative accuracy blocks, then the norm
    # slice appended via _simultaneous_norm_residual_from_state (same terms used for G ≈ J'J).
    T = data.T
    residuals = T[]

    if data.theta_ext_acc > zero(T)
        if data.tL_has_free_parameters && data.J_ext_L_initial.accuracy > data.obj_tol
            scale = sqrt(data.theta_ext_acc / data.J_ext_L_initial.accuracy)
            for t in data.ext_tests
                if t.activeL
                    push!(residuals, sqrt(t.omega) * scale *
                                     (dot(state.tL, t.g_perp) - t.gL_perp) / t.deltaL)
                end
            end
        end
        if data.tR_has_free_parameters && data.J_ext_R_initial.accuracy > data.obj_tol
            scale = sqrt(data.theta_ext_acc / data.J_ext_R_initial.accuracy)
            for t in data.ext_tests
                if t.activeR
                    push!(residuals, sqrt(t.omega) * scale *
                                     (dot(state.tR, t.g_perp) - t.gR_perp) / t.deltaR)
                end
            end
        end
    end

    # Norm block (tL/tR/J_up Frobenius terms): shared with the metric via r_norm(y).
    append!(residuals, _simultaneous_norm_residual_from_state(state, data, T))

    if data.theta_S_acc > zero(T) && data.J_der_initial > data.obj_tol &&
       !isempty(data.der_tests)
        scale = sqrt(data.theta_S_acc / data.J_der_initial)
        norm_scale = _residual_norm_scale(data.w, data.derivative_error_norm)
        for t in data.der_tests
            err = norm_scale .* (state.D * t.g_hat - t.gx_hat)
            append!(residuals, sqrt(t.omega) .* scale .* err)
        end
    end

    return residuals
end

function _simultaneous_residual(y, data)
    return _simultaneous_residual_from_state(_simultaneous_state(y, data), data)
end

function _simultaneous_trial_norm(y, data, ::Type{T}) where T
    if :ZC in keys(data)
        return _simultaneous_state_norm_from_state(_simultaneous_state(y, data), y, data, T)
    end
    return max(one(T), _euclidean_norm(y))
end

function _simultaneous_trial_metrics(y, data, residual, ::Type{T}) where T
    if :ZC in keys(data)
        state = _simultaneous_state(y, data)
        state_norm = _simultaneous_state_norm_from_state(state, y, data, T)
        R = _simultaneous_residual_from_state(state, data)
        objective = isempty(R) ? zero(T) : T(sum(abs2, R))
        return state_norm, objective
    end
    # Unit-test / stub data: keep the old ‖y‖ gate and evaluate the supplied residual only.
    R = residual(y)
    objective = isempty(R) ? zero(T) : T(sum(abs2, R))
    return _simultaneous_trial_norm(y, data, T), objective
end

function _simultaneous_sequential_initial_y(setup, V, Vx, vL, vR, op_basis, quad_basis,
                                            data; rank_tol, sequential_kwargs)
    seq = _optimize_fsbp_operator_sequential_core(
        setup, V, Vx, vL, vR, op_basis, quad_basis; sequential_kwargs...)
    xseq = zeros(data.T, data.nS + data.nT)
    xseq[1:data.nS] .= _vec_skew(seq.S, data.pairs, data.T)
    if data.nT > 0
        A = vcat(data.BL, data.BR)
        b = vcat(seq.tL - data.tL0, seq.tR - data.tR0)
        a = _pseudoinverse_solve(A, b; rank_tol = rank_tol)
        xseq[(data.nS + 1):(data.nS + data.nT)] .= a
    end
    size(data.ZC, 2) == 0 && return zeros(data.T, 0)
    return _pseudoinverse_solve(data.ZC, xseq - data.x0; rank_tol = rank_tol)
end

"""
    _simultaneous_global_search_starts(...) -> (starts, info)

Expand `base_starts` (minimum-norm `y=0`, optional sequential init) into a diverse
multistart set for the coupled nonlinear least-squares solve.

Pipeline:
1. Score baselines and mark them `protected` so they survive final screening.
2. Build a state metric from the norm residual blocks and normalize ray directions
   in this metric.
3. Run two rounds of shallow radial bracketing around elite centers, stopping
   each ray when the state norms grow beyond the configured limit.
4. From all screened candidates, pick `requested_starts` points: all protected
   baselines first, then lowest-objective unprotected points subject to a diversity radius.

# Keyword arguments
- `global_num_candidates` — cap on screened ray samples per search round (`nothing` → auto).
- `norm_growth_limit` — stop each ray when `‖·‖_G` along it exceeds this multiple of the center norm.
- `radial_steps` — number of trial radii sampled per ray before bracketing stops or the budget is spent.
"""
function _simultaneous_global_search_starts(base_starts, requested_starts::Int,
                                            ::Type{T}, data, residual;
                                            global_num_candidates,
                                            norm_growth_limit,
                                            radial_steps::Int) where T
    # -- Baselines: score ‖residual(y)‖² and tag protected=true (never dropped later)
    protected = [_simultaneous_scored_candidate(y, residual, T, true) for y in base_starts]
    starts = [copy(c.y) for c in protected]
    nvars = length(starts[1])
    # -- set inital sample radius 20% percent away from the starting point in the norm objectives.
    center_norm = _simultaneous_trial_norm(starts[1], data, T)
    initial_radius = max(sqrt(eps(T)), T(0.2) * center_norm)
    # -- If no free parameters or already have enough protected starts, return the protected starts
    if nvars == 0 || requested_starts <= length(protected)
        objectives = [c.objective for c in protected]
        info = _simultaneous_start_info(
            length(protected), length(starts), length(protected),
            initial_radius, zero(T), objectives, T)
        return starts, info
    end

    # -- Build the state metric G s.t. ‖Δy‖_G ≈ Δobjective
    #    where y explores the free parameter nullspace of the coupled system
    metric = _simultaneous_state_metric(starts[1], data, T)
    # note that ‖tL‖ and ‖tR‖ are linear in y, so this part of G is constant in y
    # however, ‖J_up‖_F is quadratic in y, so G does change in y. Still, the use
    # of a single constant G for all rays is `good enough` for estimating whitened distances

    # -- ad-hoc rule for the number of candidates to screen for in each round
    candidates_per_round = _simultaneous_global_candidate_count(
        requested_starts, nvars, global_num_candidates)
    candidates = copy(protected)   # initialize a pool of all y ever screened (baselines + search)
    centers = [copy(s) for s in starts]  # initialize the elite centers for the next round
    halton_index = 1
    search_rounds = 2 # number of rounds of shallow radial bracketing around elite centers
    global_elite_fraction = 1//4 # fraction of the candidates to promote to the next round

    # -- Global search rounds: bracket state-whitened rays, keep best diverse elites
    for round in 1:search_rounds
        round_candidates = NamedTuple[]
        # -- ad-hoc rule for the number of rays to sample for each round
        ray_count = _simultaneous_global_ray_count(
            candidates_per_round, nvars, radial_steps)
        # -- sample rays in the state space y (nullspace of the coupled system)
        directions, halton_index = _simultaneous_ray_directions(
            nvars, ray_count, halton_index, T)
        ray_candidate_sets = Vector{Vector{NamedTuple}}()
        for direction in directions
            for center in centers
                # -- around each center, for the given direction, get `radial_steps` sample points
                #    The initial step is taken to always be smaller than norm_growth_limit (in physical norm space)
                #    then the radius is expanded geometrically until the norm_growth_limit is reached.
                #    We are left with `radial_steps` candidate points for each ray, returned in ray_candidates.
                ray_candidates = _simultaneous_bracket_ray_candidates(
                    center, direction, residual, data, metric, T;
                    initial_radius = initial_radius,
                    norm_growth_limit = norm_growth_limit,
                    radial_steps = radial_steps,
                    remaining_budget = radial_steps)
                isempty(ray_candidates) || push!(ray_candidate_sets, ray_candidates)
            end
        end
        # Merge ray samples round-robin by radius index: all rays' 1st bracket point, then all
        # 2nd points, etc. (ray_candidates[k] is the k-th outward sample on that ray). Stops at
        # candidates_per_round so early rays do not exhaust the round budget alone.
        # i.e. ensures that all rays are sampled at least once.
        for step in 1:radial_steps
            for ray_candidates in ray_candidate_sets
                length(round_candidates) >= candidates_per_round && break
                length(ray_candidates) >= step &&
                    push!(round_candidates, ray_candidates[step])
            end
            length(round_candidates) >= candidates_per_round && break
        end
        isempty(round_candidates) && break
        append!(candidates, round_candidates)

        if round < search_rounds
            # Promote low-objective, spatially separated round candidates to next round's ray centers.
            sorted_round = sort(round_candidates; by = c -> c.objective)
            # How many elites to keep (default 1/4 of this round's pool, at least one).
            elite_count = _simultaneous_elite_count(length(sorted_round), global_elite_fraction)
            # Min ‖Δy‖_G between elites: pick a radius for new elite separations
            # so rays do not all restart on top of each other. Calculated
            # ad-hoc as ~half the lower quartile of pairwise round distances.
            elite_radius = _simultaneous_default_diversity_radius(
                sorted_round, initial_radius, T; metric = metric)
            # Greedy pick: lowest ‖residual‖² first, skip points within elite_radius of a chosen elite.
            centers = [copy(c.y) for c in _simultaneous_select_candidate_records(
                sorted_round, NamedTuple[], elite_count, elite_radius, T; metric = metric)]
            isempty(centers) && (centers = [copy(sorted_round[1].y)])
        end
    end

    # -- Final pick: keep all protected baselines, then best unprotected up to requested_starts
    diversity_radius = _simultaneous_default_diversity_radius(
        candidates, initial_radius, T; metric = metric)
    selected = _simultaneous_select_candidate_records(
        candidates, protected, requested_starts, diversity_radius, T; metric = metric)
    starts = [copy(c.y) for c in selected]
    selected_objectives = [c.objective for c in selected]
    info = _simultaneous_start_info(
        length(candidates), length(starts), length(protected),
        initial_radius, diversity_radius, selected_objectives, T)
    return starts, info
end

function _simultaneous_objective(residual, y, ::Type{T}) where T
    R = residual(y)
    isempty(R) && return zero(T)
    return T(sum(abs2, R))
end

function _simultaneous_scored_candidate(y, residual, ::Type{T}, protected::Bool) where T
    objective = _simultaneous_objective(residual, y, T)
    return (y = copy(y), objective = objective, protected = protected)
end

"""
    _simultaneous_state_metric(yref, data, T) -> G

Build a symmetric positive (semi)definite metric `G` on the coupled nullspace
coordinates `y`.  Global-search diversity uses `‖Δy‖_G = sqrt(Δy' G Δy)` instead
of Euclidean distance so "nearby" starts are measured in a scale tied to the
norm objectives (tL/tR/Jacobian Frobenius blocks), not raw `y` components.

We want a distance Δy to translate to a distance in the objective space, i.e.
‖Δy‖_G = Δobjective = ‖residual(yref + Δy) - residual(yref)‖. Let f(y) = residual(y).
Then the Taylor expansion of f(yref + Δy) around yref is:
f(yref + Δy) ≈ f(yref) + Df(yref) Δy + O(‖Δy‖²), Df(y) = [∂f/∂y1, ∂f/∂y2, ..., ∂f/∂yn]
Then ‖Δy‖_G ≈ ‖Df(yref) Δy‖ = √(Δy' Df(yref)' Df(yref) Δy) = √(Δy' G Δy) = ‖Δy‖_G,
where G = Df(yref)' Df(yref)

When no norm residuals are active, fall back to the `ZC'ZC` nullspace geometry
or the identity.
"""
function _simultaneous_state_metric(yref, data, ::Type{T}) where T
    nvars = length(yref)
    G = zeros(T, nvars, nvars)
    nvars == 0 && return G
    r_norm0 = _simultaneous_norm_residual(yref, data, T)
    if isempty(r_norm0)
        # No norm block in the objective: use coupled-system nullspace metric or I
        if :ZC in keys(data) && size(data.ZC, 2) == nvars
            G .= data.ZC' * data.ZC
        else
            for j in 1:nvars
                G[j, j] = one(T)
            end
        end
        return _simultaneous_regularize_metric(G, T)
    end

    # Central-difference Jacobian of the norm residual slice f(y)=r_norm(y) at yref
    hscale = eps(T)^(one(T) / T(3))
    yp = copy(yref)
    ym = copy(yref)
    J = zeros(T, length(r_norm0), nvars)
    for j in 1:nvars
        h = hscale * max(one(T), abs(yref[j]))
        yp[j] = yref[j] + h
        ym[j] = yref[j] - h
        r_norm_p = _simultaneous_norm_residual(yp, data, T)
        r_norm_m = _simultaneous_norm_residual(ym, data, T)
        J[:, j] .= (r_norm_p .- r_norm_m) ./ (T(2) * h)
        yp[j] = yref[j]
        ym[j] = yref[j]
    end
    # Gauss–Newton metric G ≈ J'J for the norm sub-objective
    G .= J' * J
    return _simultaneous_regularize_metric(G, T)
end

"""
    _simultaneous_norm_residual(y, data, T) -> Vector

Norm-only slice of [`_simultaneous_residual`](@ref): (no accuracy objectives).
Omits `tL`/`tR` rows when the corresponding boundary is fixed
   i.e. (`tL_has_free_parameters` / `tR_has_free_parameters` = false).
Used to build the state metric `G` via `‖r_norm(y)‖₂`.
finite differences of this vector define how strongly each `y` component affects
‖tL‖, ‖tR‖, and ‖J_up‖_F in the weighted least-squares objective.

Returns an empty vector when norm objectives are inactive or `data` lacks the
fields needed to evaluate them (then the metric falls back to `ZC'ZC` or `I`).
"""
function _simultaneous_norm_residual(y, data, ::Type{T}) where T
    required = (:theta_ext_norm, :theta_S_norm, :J_ext_L_initial,
                :J_ext_R_initial, :J_norm_initial, :obj_tol,
                :extrapolation_norm, :w,
                :tL_has_free_parameters, :tR_has_free_parameters)
    all(k -> k in keys(data), required) || return T[]
    return _simultaneous_norm_residual_from_state(_simultaneous_state(y, data), data, T)
end

function _simultaneous_state_norm(y, data, ::Type{T}) where T
    state = _simultaneous_state(y, data)
    return _simultaneous_state_norm_from_state(state, y, data, T)
end

# Tikhonov ridge on the metric:  G ← G + λ I,  λ = √ε · max(1, (1/n) Σⱼ |Gⱼⱼ|),
# so ‖·‖_G is well-defined when G ≈ J'J is rank-deficient.
function _simultaneous_regularize_metric(G, ::Type{T}) where T
    n = size(G, 1)
    n == 0 && return G
    scale = zero(T)
    for j in 1:n
        scale += abs(G[j, j])
    end
    scale = max(one(T), scale / T(n))
    ridge = sqrt(eps(T)) * scale
    for j in 1:n
        G[j, j] += ridge
    end
    return G
end

function _simultaneous_metric_norm(v, metric, ::Type{T}) where T
    isempty(v) && return zero(T)
    q = dot(v, metric * v)
    return sqrt(max(zero(T), T(q)))
end

function _simultaneous_global_ray_count(candidates_per_round::Int, nvars::Int,
                                        radial_steps::Int)
    candidates_per_round <= 0 && return 0
    return max(2 * nvars, cld(candidates_per_round, radial_steps))
end

function _simultaneous_global_candidate_count(requested_starts::Int, nvars::Int,
                                              global_num_candidates)
    global_num_candidates !== nothing && return max(0, Int(global_num_candidates))
    return max(3 * requested_starts, 4 * nvars, 24)
end

function _simultaneous_elite_count(candidate_count::Int, elite_fraction)
    candidate_count == 0 && return 0
    return max(1, min(candidate_count, ceil(Int, candidate_count * elite_fraction)))
end

# Build up to `ray_count` unit search directions: first ±e_j along each coordinate,
# then using a Halton sequence on the sphere (each paired with its negative) until full.
function _simultaneous_ray_directions(nvars::Int, ray_count::Int,
                                      halton_index::Int, ::Type{T}) where T
    directions = Vector{Vector{T}}()
    for j in 1:nvars
        length(directions) >= ray_count && break
        direction = zeros(T, nvars)
        direction[j] = one(T)
        push!(directions, direction)
        length(directions) >= ray_count && break
        direction = zeros(T, nvars)
        direction[j] = -one(T)
        push!(directions, direction)
    end
    while length(directions) < ray_count
        direction = _simultaneous_halton_direction(halton_index, nvars, T)
        push!(directions, direction)
        if length(directions) < ray_count
            push!(directions, -direction)
        end
        halton_index += 1
    end
    return directions, halton_index
end

function _simultaneous_bracket_ray_candidates(center, direction, residual, data,
                                              metric, ::Type{T};
                                              initial_radius,
                                              norm_growth_limit,
                                              radial_steps::Int,
                                              remaining_budget::Int) where T
    remaining_budget <= 0 && return NamedTuple[]
    dir_norm = _simultaneous_metric_norm(direction, metric, T)
    dir_norm <= sqrt(eps(T)) && return NamedTuple[]
    unit_direction = direction ./ dir_norm
    center_norm = _simultaneous_trial_norm(center, data, T)
    max_norm = T(norm_growth_limit) * max(center_norm, sqrt(eps(T)))
    radius = T(initial_radius)

    candidates = NamedTuple[]
    trial_y = similar(center)
    trial_norm = T(Inf)
    trial_objective = zero(T)

    # One state evaluation per trial: backoff to at least norm_growth_limit radius
    # for the initial evaluation, then march outward.
    trial_y .= center .+ radius .* unit_direction
    trial_norm, trial_objective = _simultaneous_trial_metrics(trial_y, data, residual, T)
    for _ in 1:radial_steps
        trial_norm <= max_norm && break
        radius /= T(2)
        trial_y .= center .+ radius .* unit_direction
        trial_norm, trial_objective = _simultaneous_trial_metrics(trial_y, data, residual, T)
    end

    # Then march outward until the budget is spent or the norm exceeds max_norm
    for _ in 1:radial_steps
        length(candidates) >= remaining_budget && break
        trial_norm > max_norm && break
        push!(candidates, (y = copy(trial_y), objective = trial_objective, protected = false))
        radius *= T(2)
        trial_y .= center .+ radius .* unit_direction
        trial_norm, trial_objective = _simultaneous_trial_metrics(trial_y, data, residual, T)
    end
    return candidates
end

function _simultaneous_select_candidate_records(candidates, protected,
                                                requested_starts::Int,
                                                diversity_radius, ::Type{T};
                                                metric = nothing) where T
    selected = NamedTuple[]
    scale = _simultaneous_candidate_scale(candidates, protected, T; metric = metric)
    duplicate_tol = sqrt(eps(T)) * scale
    for candidate in protected
        if !_simultaneous_has_nearby_candidate(selected, candidate.y, duplicate_tol, T;
                                               metric = metric)
            push!(selected, candidate)
        end
    end

    target = max(requested_starts, length(selected))
    remaining = sort([c for c in candidates if !c.protected && isfinite(c.objective)];
                     by = c -> c.objective)
    radius = max(T(diversity_radius), duplicate_tol)
    min_radius = duplicate_tol
    while length(selected) < target && radius >= min_radius
        added = false
        for candidate in remaining
            length(selected) >= target && break
            if !_simultaneous_has_nearby_candidate(selected, candidate.y, duplicate_tol, T;
                                                   metric = metric) &&
               !_simultaneous_has_nearby_candidate(selected, candidate.y, radius, T;
                                                   metric = metric)
                push!(selected, candidate)
                added = true
            end
        end
        added || (radius /= T(2))
        radius == zero(T) && break
    end

    if length(selected) < target
        for candidate in remaining
            length(selected) >= target && break
            if !_simultaneous_has_nearby_candidate(selected, candidate.y, duplicate_tol, T;
                                                   metric = metric)
                push!(selected, candidate)
            end
        end
    end
    return selected
end

function _simultaneous_has_nearby_candidate(candidates, y, radius, ::Type{T};
                                            metric = nothing) where T
    isempty(candidates) && return false
    if metric === nothing
        return any(_euclidean_norm(c.y - y) <= radius for c in candidates)
    end
    return any(_simultaneous_metric_norm(c.y - y, metric, T) <= radius
               for c in candidates)
end

function _simultaneous_candidate_scale(candidates, protected, ::Type{T};
                                       metric = nothing) where T
    scale = one(T)
    for candidate in candidates
        nrm = metric === nothing ? _euclidean_norm(candidate.y) :
              _simultaneous_metric_norm(candidate.y, metric, T)
        scale = max(scale, nrm)
    end
    for candidate in protected
        nrm = metric === nothing ? _euclidean_norm(candidate.y) :
              _simultaneous_metric_norm(candidate.y, metric, T)
        scale = max(scale, nrm)
    end
    return scale
end

function _simultaneous_default_diversity_radius(candidates, scale, ::Type{T};
                                               metric = nothing) where T
    ys = [c.y for c in candidates if isfinite(c.objective)]
    length(ys) < 2 && return sqrt(eps(T)) * max(one(T), T(scale))
    distances = T[]
    for i in 1:(length(ys) - 1)
        for j in (i + 1):length(ys)
            d = metric === nothing ? _euclidean_norm(ys[i] - ys[j]) :
                _simultaneous_metric_norm(ys[i] - ys[j], metric, T)
            d > zero(T) && push!(distances, d)
        end
    end
    isempty(distances) && return sqrt(eps(T)) * max(one(T), T(scale))
    sort!(distances)
    qidx = max(1, cld(length(distances), 4))
    floor = sqrt(eps(T)) * max(one(T), T(scale))
    return max(floor, distances[qidx] / T(2))
end

function _simultaneous_start_info(candidate_count::Int,
                                  selected_count::Int, protected_count::Int,
                                  search_scale, diversity_radius,
                                  selected_objectives,
                                  ::Type{T}) where T
    best_selected = isempty(selected_objectives) ? T(Inf) : minimum(selected_objectives)
    worst_selected = isempty(selected_objectives) ? T(Inf) : maximum(selected_objectives)
    return (;
        candidate_count,
        selected_count,
        protected_count,
        search_scale = T(search_scale),
        diversity_radius = T(diversity_radius),
        best_selected_objective = T(best_selected),
        worst_selected_objective = T(worst_selected),
    )
end

function _print_simultaneous_start_summary(info)
    println("Simultaneous multi-start point selection:")
    println("  # of starts screened = $(info.candidate_count)")
    println("  # of starts selected = $(info.selected_count)")
    println("  initial search radius (scaled by estimated Δ norm-objective) = $(info.search_scale)")
    println("  start diversity radius (scaled by estimated Δ norm-objective) = $(info.diversity_radius)")
    println("  best start point inital objective = $(info.best_selected_objective)")
    println("  worst start point initial objective = $(info.worst_selected_objective)")
    println("\n")
    return nothing
end

function _simultaneous_halton_direction(index::Int, nvars::Int, ::Type{T}) where T
    primes = _first_primes(nvars)
    direction = zeros(T, nvars)
    for j in 1:nvars
        direction[j] = T(2) * _radical_inverse(index, primes[j], T) - one(T)
    end
    if _euclidean_norm(direction) == zero(T)
        direction[1] = one(T)
    end
    return direction
end

function _radical_inverse(index::Int, base::Int, ::Type{T}) where T
    value = zero(T)
    factor = inv(T(base))
    n = index
    while n > 0
        digit = mod(n, base)
        value += T(digit) * factor
        n = div(n, base)
        factor /= T(base)
    end
    return value
end

function _first_primes(n::Int)
    primes = Int[]
    candidate = 2
    while length(primes) < n
        isprime = true
        limit = floor(Int, sqrt(candidate))
        for p in primes
            p > limit && break
            if candidate % p == 0
                isprime = false
                break
            end
        end
        isprime && push!(primes, candidate)
        candidate += candidate == 2 ? 1 : 2
    end
    return primes
end

"""
    _solve_nonlinear_least_squares(residual, y0, T; kwargs...) -> NamedTuple

Minimize `‖residual(y)‖₂²` over `y` with a small in-house Gauss–Newton /
Levenberg–Marquardt loop.  Used once per multistart after
[`_simultaneous_global_search_starts`](@ref).

# Arguments
- `residual(y)` — vector residual (simultaneous FSBP least-squares stack).
- `y0` — initial free parameters in the coupled nullspace.

# Keyword arguments
- `solver` — `:levenberg_marquardt` (default via `:auto`), `:gauss_newton`, or `:auto`.
- `max_iter`, `step_tol`, `grad_tol`, `obj_tol` — standard stopping criteria.
- `rank_tol` — passed to [`_least_squares_step`](@ref) for rank-deficient Jacobians.

# Returns
Named tuple `(y, objective, converged, status, iterations)` where `status` is one of
`:objective_tol`, `:gradient_tol`, `:step_tol`, `:no_decrease`, `:max_iter`,
`:nonfinite_initial`, or `:no_free_parameters`.

# Notes
- `J` is rebuilt each iteration by central finite differences (`2n` residual evaluations).
- LM adds `√λ I` to the Gauss–Newton normal equations; `λ` is increased/decreased heuristically
  until a trial step lowers the objective (no line-search on `λ` beyond that inner loop).
- Gauss–Newton uses backtracking on the step length `α` only.
"""
function _solve_nonlinear_least_squares(residual, y0, ::Type{T};
                                        solver::Symbol,
                                        max_iter::Int,
                                        step_tol,
                                        grad_tol,
                                        obj_tol,
                                        rank_tol) where T
    solver === :auto && (solver = :levenberg_marquardt)
    y = copy(y0)
    R = residual(y)
    obj = sum(abs2, R)
    !isfinite(obj) && return (y = y, objective = obj, converged = false,
                              status = :nonfinite_initial, iterations = 0)
    obj <= obj_tol && return (y = y, objective = obj, converged = true,
                              status = :objective_tol, iterations = 0)
    nvars = length(y)
    nvars == 0 && return (y = y, objective = obj, converged = true,
                          status = :no_free_parameters, iterations = 0)

    # LM damping: solve (J'J + λI) δ = -J'R; λ↑ when trial rejected, λ↓ on acceptance.
    lambda = one(T) / T(1000)
    lambda_min = eps(T)
    lambda_max = inv(eps(T))
    for iter in 1:max_iter
        J = _finite_difference_jacobian(residual, y, R, T)
        g = J' * R
        # First-order optimality: ‖J'R‖ small relative to objective scale.
        if _euclidean_norm(g) <= grad_tol * max(one(T), sqrt(obj))
            return (y = y, objective = obj, converged = true,
                    status = :gradient_tol, iterations = iter - 1)
        end

        if solver === :gauss_newton
            # Pure GN: damped normal equations only via backtracking on α.
            step = _least_squares_step(J, R, zero(T); rank_tol = rank_tol)
            accepted = false
            alpha = one(T)
            for _ in 1:25
                ytrial = y + alpha .* step
                Rtrial = residual(ytrial)
                objtrial = sum(abs2, Rtrial)
                if isfinite(objtrial) && objtrial <= obj
                    y, R, obj = ytrial, Rtrial, objtrial
                    accepted = true
                    break
                end
                alpha /= T(2)
            end
            if !accepted
                return (y = y, objective = obj, converged = false,
                        status = :no_decrease, iterations = iter - 1)
            end
            if _euclidean_norm(alpha .* step) <= step_tol * max(one(T), _euclidean_norm(y))
                return (y = y, objective = obj, converged = true,
                        status = :step_tol, iterations = iter)
            end
        else
            # Levenberg–Marquardt: inner loop increases λ until objective decreases.
            accepted = false
            step = zeros(T, nvars)
            for _ in 1:25
                step = _least_squares_step(J, R, lambda; rank_tol = rank_tol)
                ytrial = y + step
                Rtrial = residual(ytrial)
                objtrial = sum(abs2, Rtrial)
                if isfinite(objtrial) && objtrial < obj
                    y, R, obj = ytrial, Rtrial, objtrial
                    lambda = max(lambda / T(10), lambda_min)
                    accepted = true
                    break
                end
                lambda *= T(10)
                lambda > lambda_max && break
            end
            if !accepted
                return (y = y, objective = obj, converged = false,
                        status = :no_decrease, iterations = iter - 1)
            end
            if _euclidean_norm(step) <= step_tol * max(one(T), _euclidean_norm(y))
                return (y = y, objective = obj, converged = true,
                        status = :step_tol, iterations = iter)
            end
        end
        obj <= obj_tol && return (y = y, objective = obj, converged = true,
                                  status = :objective_tol, iterations = iter)
    end

    return (y = y, objective = obj, converged = false,
            status = :max_iter, iterations = max_iter)
end

function _finite_difference_jacobian(residual, y, R0, ::Type{T}) where T
    m = length(R0)
    n = length(y)
    J = zeros(T, m, n)
    n == 0 && return J
    hscale = eps(T)^(one(T) / T(3))
    yp = copy(y)
    ym = copy(y)
    for j in 1:n
        h = hscale * max(one(T), abs(y[j]))
        yp[j] = y[j] + h
        ym[j] = y[j] - h
        Rp = residual(yp)
        Rm = residual(ym)
        J[:, j] .= (Rp .- Rm) ./ (T(2) * h)
        yp[j] = y[j]
        ym[j] = y[j]
    end
    return J
end

function _least_squares_step(J, R, lambda; rank_tol)
    T = promote_type(eltype(J), eltype(R))
    n = size(J, 2)
    if lambda > zero(T)
        A = vcat(Matrix{T}(J), sqrt(lambda) .* Matrix{T}(I, n, n))
        b = vcat(Vector{T}(R), zeros(T, n))
        return -_least_squares_solve(A, b; rank_tol = rank_tol, prefer_direct = true)
    end
    return -_least_squares_solve(J, R; rank_tol = rank_tol, prefer_direct = true)
end

function _print_simultaneous_nonlinear_summary(results, local_min_tol, ::Type{T}) where T
    finite_results = [r for r in results if isfinite(r.objective)]
    println("Simultaneous nonlinear solve summary:")
    println("  number of nonlinear starts = $(length(results))")
    if isempty(finite_results)
        println("  no finite nonlinear objectives")
        println("\n")
        return nothing
    end
    objectives = sort([r.objective for r in finite_results])
    best = first(objectives)
    worst = last(objectives)
    println("  converged starts = $(count(r -> r.converged, finite_results))")
    println("  best (converged) objective = $best")
    println("  worst (converged) objective = $worst")
    threshold = local_min_tol * max(one(T), best)
    n_same = count(r -> abs(r.objective - best) <= threshold, finite_results)
    println("  objective match tolerance = $threshold (local_min_tol = $local_min_tol)")
    println("  # of starts within tolerance of best = $n_same / $(length(finite_results))")
    println("\n")
    return nothing
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
    any(omega .< zero(T)) && throw(ArgumentError("All test_weights must be non-negative."))
    return omega
end

function _validate_objective_weights(weights, name::AbstractString)
    vals = if weights isa NamedTuple
        collect(values(weights))
    elseif weights isa Tuple
        collect(weights)
    else
        throw(ArgumentError("$name must be a NamedTuple or Tuple, got $(typeof(weights))."))
    end
    any(v -> v < 0, vals) &&
        throw(ArgumentError("All $name must be non-negative."))
    return nothing
end

function _extract_objective_weights(::Type{T}, weights, name::AbstractString) where T
    _validate_objective_weights(weights, name)
    return (T(_objective_weight(weights, :accuracy, 1, 1//2)),
            T(_objective_weight(weights, :norm, 2, 1//2)))
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

function _symmetry_tolerance(scale, ::Type{T}) where T
    return T(100) * sqrt(eps(T)) * max(one(T), scale)
end

function _check_flip_symmetric_grid(x, w, xL, xR)
    T = eltype(x)
    node_scale = max(one(T), maximum(abs.(x)), abs(xL), abs(xR))
    node_tol = _symmetry_tolerance(node_scale, T)
    center = xL + xR
    node_err = maximum(abs.(x .+ reverse(x) .- center))
    node_err <= node_tol || throw(ArgumentError(
        "extrapolation_symmetry=:flip requires reflection-paired nodes; " *
        "max |x[i] + x[N+1-i] - (xL+xR)| = $node_err exceeds $node_tol."))

    weight_scale = max(one(T), maximum(abs.(w)))
    weight_tol = _symmetry_tolerance(weight_scale, T)
    weight_err = maximum(abs.(w .- reverse(w)))
    weight_err <= weight_tol || throw(ArgumentError(
        "extrapolation_symmetry=:flip requires reflection-paired weights; " *
        "max |w[i] - w[N+1-i]| = $weight_err exceeds $weight_tol."))
    return nothing
end

function _minimum_extrapolation_constraint_solution(C, w, d, norm_backend::Symbol;
                                                    rank_tol)
    Ct = Matrix(transpose(C))
    if norm_backend === :Hinv
        MinvCt = _scale_rows(Ct, w)
    elseif norm_backend === :H
        MinvCt = _divide_rows(Ct, w)
    elseif norm_backend in (:Euclidean, :Frobenius)
        MinvCt = Ct
    else
        throw(ArgumentError("Unsupported extrapolation norm $norm_backend."))
    end
    return MinvCt * _pseudoinverse_solve(C * MinvCt, d; rank_tol = rank_tol)
end

function _check_constraint_residual(C, t, d, context::AbstractString)
    T = eltype(t)
    residual = _euclidean_norm(C * t - d)
    scale = max(one(T), _frobenius_norm(C) * _euclidean_norm(t), _euclidean_norm(d))
    tol = T(1000) * sqrt(eps(T)) * scale
    residual <= tol || throw(ArgumentError(
        "$context: exact flip-symmetric extrapolation constraints are inconsistent; " *
        "residual $residual exceeds $tol."))
    return nothing
end

function _build_flip_symmetric_extrapolation(V, w, vL, vR, x, xL, xR,
                                             left_endpoint_idx,
                                             right_endpoint_idx,
                                             norm_backend::Symbol; rank_tol)
    T = eltype(w)
    N = length(w)
    if left_endpoint_idx !== nothing || right_endpoint_idx !== nothing
        if left_endpoint_idx === nothing || right_endpoint_idx === nothing
            throw(ArgumentError(
                "extrapolation_symmetry=:flip requires both boundary endpoints to be nodes, or neither."))
        end
        expected_right_idx = N + 1 - left_endpoint_idx
        right_endpoint_idx == expected_right_idx || throw(ArgumentError(
            "extrapolation_symmetry=:flip requires endpoint nodes to be reverse pairs; " *
            "left endpoint index $left_endpoint_idx maps to $expected_right_idx, " *
            "but right endpoint index is $right_endpoint_idx."))
        tL0 = zeros(T, N)
        tL0[left_endpoint_idx] = one(T)
        tR0 = reverse(tL0)
        return tL0, tR0, zeros(T, N, 0)
    end

    C = vcat(transpose(V), transpose(reverse(V; dims = 1)))
    d = vcat(vL, vR)
    tL0 = _minimum_extrapolation_constraint_solution(C, w, d, norm_backend;
                                                     rank_tol = rank_tol)
    _check_constraint_residual(C, tL0, d, "extrapolation_symmetry=:flip")
    Zflip = _nullspace_basis(C; rank_tol = rank_tol)
    return tL0, reverse(tL0), Zflip
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
    # -- Deprecated, use _tL_extrapolation_objectives / _tR_extrapolation_objectives
    return _extrapolation_accuracy_objective_boundary(tL, tests, :left) +
           _extrapolation_accuracy_objective_boundary(tR, tests, :right)
end

function _extrapolation_accuracy_objective_boundary(t, tests, side::Symbol)
    T = eltype(t)
    total = zero(T)
    if side === :left
        for test in tests
            if test.activeL
                err = (dot(t, test.g_perp) - test.gL_perp) / test.deltaL
                total += test.omega * err * err
            end
        end
    elseif side === :right
        for test in tests
            if test.activeR
                err = (dot(t, test.g_perp) - test.gR_perp) / test.deltaR
                total += test.omega * err * err
            end
        end
    else
        throw(ArgumentError("extrapolation boundary must be :left or :right, got $side."))
    end
    return total
end

function _extrapolation_norm_objective(tL, tR, w, norm_backend::Symbol)
    # -- Deprecated, use _tL_extrapolation_objectives / _tR_extrapolation_objectives
    return _weighted_norm2(tL, w, norm_backend) +
           _weighted_norm2(tR, w, norm_backend)
end

function _extrapolation_norm_objective_boundary(t, w, norm_backend::Symbol)
    return _weighted_norm2(t, w, norm_backend)
end

function _tL_extrapolation_objectives(tL, tests, w, norm_backend::Symbol)
    return (accuracy = _extrapolation_accuracy_objective_boundary(tL, tests, :left),
            norm = _extrapolation_norm_objective_boundary(tL, w, norm_backend))
end

function _tR_extrapolation_objectives(tR, tests, w, norm_backend::Symbol)
    return (accuracy = _extrapolation_accuracy_objective_boundary(tR, tests, :right),
            norm = _extrapolation_norm_objective_boundary(tR, w, norm_backend))
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

Parameterization, solved independently at each boundary:

    tL = tL0 + ZL*aL,    tR = tR0 + ZR*aR

Objectives at the starting point are normalized separately for each boundary:

    J_acc,L(tL)  = Σ_m ω_m ( ⟨tL,g_m^⊥⟩ - g_{L,m}^⊥ )^2 / δ_{L,m}^2
    J_acc,R(tR)  = Σ_m ω_m ( ⟨tR,g_m^⊥⟩ - g_{R,m}^⊥ )^2 / δ_{R,m}^2
    J_norm,L(tL) = ‖tL‖_norm^2,    J_norm,R(tR) = ‖tR‖_norm^2

For each boundary, the helper writes affine residuals in `a` as rows `row`
with rhs `constant` (`row' a + constant`), stacks them, solves `A a ≈ -b`
for the exact quadratic minimizer, and updates the boundary vector.
"""
function _optimize_extrapolation(tL0, tR0, ZL, ZR, tests, w, norm_backend,
                                 theta_acc, theta_norm, J_L, J_R, obj_tol;
                                 tL_has_free_parameters::Bool,
                                 tR_has_free_parameters::Bool,
                                 rank_tol)
    tL = if tL_has_free_parameters
        _optimize_extrapolation_boundary(tL0, ZL, tests, w, norm_backend,
                                         theta_acc, theta_norm, obj_tol;
                                         side = :left,
                                         J_acc0 = J_L.accuracy,
                                         J_norm0 = J_L.norm,
                                         rank_tol = rank_tol)
    else
        tL0
    end
    tR = if tR_has_free_parameters
        _optimize_extrapolation_boundary(tR0, ZR, tests, w, norm_backend,
                                         theta_acc, theta_norm, obj_tol;
                                         side = :right,
                                         J_acc0 = J_R.accuracy,
                                         J_norm0 = J_R.norm,
                                         rank_tol = rank_tol)
    else
        tR0
    end
    return tL, tR
end

function _append_extrapolation_boundary_rows!(rows, rhs, t0, Z, tests, w,
                                              norm_backend, theta_acc,
                                              theta_norm, J, obj_tol;
                                              side::Symbol)
    T = eltype(t0)
    nvars = size(Z, 2)
    J_acc0 = J.accuracy
    J_norm0 = J.norm

    # ── Accuracy block (active if θ_acc > 0 and J_acc0 > obj_tol) ───────────
    # Boundary residual for one test (ω = t.omega, δ = boundary scale):
    #
    #   r(a) = (√ω / δ) ( ⟨t, g^⊥⟩ - g_boundary^⊥ )
    #        = (√ω / δ) ( ⟨t0, g^⊥⟩ - g_boundary^⊥ + a' (Z' g^⊥) )
    #
    # Now stack row' a + constant ≈ 0 with
    #   row       = (√ω / δ) (Z' g^⊥) * global_scale,   global_scale = √(θ_acc / J_acc0)
    #   constant  = (√ω / δ) ( ⟨t0, g^⊥⟩ - g_boundary^⊥ ) * global_scale  (= r(0))
    #
    # So row' a + constant = global_scale * r(a).  Minimize ‖global_scale * r(a)‖^2
    # After stacking all rows (accuracy + norm), we get A[i,:] = row_i,  b[i] = constant_i,
    # so minimizing ||A a + b||_2^2 drives the objective function toward zero.
    if theta_acc > zero(T) && J_acc0 > obj_tol
        global_scale = sqrt(theta_acc / J_acc0)
        for t in tests
            sqrtomega = sqrt(t.omega)
            if side === :left && t.activeL
                row = zeros(T, nvars)
                row .= (Z' * t.g_perp) .* (sqrtomega * global_scale / t.deltaL)
                constant = (dot(t0, t.g_perp) - t.gL_perp) *
                           sqrtomega * global_scale / t.deltaL
                _append_row!(rows, rhs, row, constant)
            end
            if side === :right && t.activeR
                row = zeros(T, nvars)
                row .= (Z' * t.g_perp) .* (sqrtomega * global_scale / t.deltaR)
                constant = (dot(t0, t.g_perp) - t.gR_perp) *
                           sqrtomega * global_scale / t.deltaR
                _append_row!(rows, rhs, row, constant)
            end
        end
    end

    # ── Norm block (active if θ_norm > 0 and J_norm0 > obj_tol) ──────────────
    # J_norm(t) = ‖t‖_norm^2.  With s = norm_scale (so ‖t‖_norm^2 = ‖s⊙t‖_2^2),
    # and t = t0 + Z*a, nodal component i is affine in a:
    #
    #   u_i(a) = s_i t_i(a) = s_i t0_i + Σ_j Zscaled[i,j] a_j
    #          = base[i] + Zscaled[i,:]' a,    base = s ⊙ t0,  Zscaled[i,j] = s_i Z[i,j]
    #
    # Stack row' a + constant ≈ 0 with  global_scale = √(θ_norm / J_norm0)
    #   row         = global_scale * Zscaled[i, :]
    #   constant    = global_scale * base[i]  (= global_scale * u_i(0) at a = 0)
    #
    # So row' a + constant = global_scale * u_i(a).  We minimize ‖global_scale * u_i(a)‖^2
    # Together with the accuracy rows, A[i,:] = row_i, b[i] = constant_i and
    # min ||A a + b||_2^2 is exact quadratic minimization.
    use_direct_lsq = theta_norm > zero(T) && J_norm0 > obj_tol
    if use_direct_lsq
        global_scale = sqrt(theta_norm / J_norm0)
        norm_scale = _residual_norm_scale(w, norm_backend)
        base = norm_scale .* t0
        Zscaled = Z .* reshape(norm_scale, :, 1)
        for i in eachindex(base)
            row = zeros(T, nvars)
            row .= global_scale .* Zscaled[i, :]
            _append_row!(rows, rhs, row, global_scale * base[i])
        end
    end
    return use_direct_lsq
end

function _optimize_extrapolation_boundary(t0, Z, tests, w, norm_backend,
                                          theta_acc, theta_norm, obj_tol;
                                          side::Symbol, J_acc0, J_norm0,
                                          rank_tol)
    T = eltype(t0)
    nvars = size(Z, 2)
    nvars == 0 && return t0

    rows = Vector{Vector{T}}()
    rhs = T[]
    J = (accuracy = J_acc0, norm = J_norm0)
    use_direct_lsq = _append_extrapolation_boundary_rows!(
        rows, rhs, t0, Z, tests, w, norm_backend, theta_acc, theta_norm, J, obj_tol;
        side = side)

    isempty(rows) && return t0
    A, b = _rows_to_matrix(rows, rhs, nvars, T)
    # When the norm block is present, its positive diagonal scaling of the
    # full-rank nullspace bases makes A full column rank. Use the faster
    # direct least-squares solve, with SVD fallback for degenerate cases.
    a = -_least_squares_solve(A, b; rank_tol = rank_tol,
                              prefer_direct = use_direct_lsq)
    return t0 + Z * a
end

function _optimize_flip_symmetric_extrapolation(tL0, Zflip, tests, w, norm_backend,
                                                theta_acc, theta_norm, J_L, J_R,
                                                obj_tol; rank_tol)
    T = eltype(tL0)
    nvars = size(Zflip, 2)
    tR0 = reverse(tL0)
    nvars == 0 && return tL0, tR0

    rows = Vector{Vector{T}}()
    rhs = T[]
    use_direct_lsq = _append_extrapolation_boundary_rows!(
        rows, rhs, tL0, Zflip, tests, w, norm_backend, theta_acc, theta_norm,
        J_L, obj_tol; side = :left)
    use_direct_lsq = _append_extrapolation_boundary_rows!(
        rows, rhs, tR0, reverse(Zflip; dims = 1), tests, w, norm_backend,
        theta_acc, theta_norm, J_R, obj_tol; side = :right) || use_direct_lsq

    isempty(rows) && return tL0, tR0
    A, b = _rows_to_matrix(rows, rhs, nvars, T)
    a = -_least_squares_solve(A, b; rank_tol = rank_tol,
                              prefer_direct = use_direct_lsq)
    tL = tL0 + Zflip * a
    return tL, reverse(tL)
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
        z = ZV[:, mu]
        for beta in 1:K
            b = V[:, beta]
            row = _skew_bilinear_row(z, b, pairs, T)
            rhs_vec = w .* Vx[:, beta] - half * (E * b)
            _append_row!(rows, rhs, row, dot(z, rhs_vec))
        end
    end

    # build actual system matrix C and right-hand side d by unwrapping the stacked vectors
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
Columns of `D_basis` are taken from `ZC`, (assembled upstream) which span the free skew
parameters; here we work with the induced derivative and upwind-Jacobian bases

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
