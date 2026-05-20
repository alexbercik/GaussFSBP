"""
    OperatorVerification.jl

Implements `check_fsbp_operator`, which runs a comprehensive suite of
verification checks on an `FSBPOperator`.

Checks performed (all tolerances default to precision-dependent values):
1. Derivative exactness:  D fⱼ ≈ fⱼ'  for all fⱼ ∈ F.
2. Quadrature exactness:  Σ wᵢ g(xᵢ) ≈ ∫g  for all g ∈ G.
3. SBP property:          Q + Qᵀ ≈ E.
4. Boundary decomposition: E ≈ tR tRᵀ - tL tLᵀ.
5. Extrapolation exactness: tLᵀ fⱼ ≈ fⱼ(xL), tRᵀ fⱼ ≈ fⱼ(xR).
6. Positive weights:       all diag(H) > 0.
7. Weight sum:             1ᵀ H 1 ≈ xR - xL.
8. Nullspace consistency:  rank(D) = nn - 1.
9. Skew-symmetry of S:    S + Sᵀ ≈ 0.
10. SBP compatibility:     Vᵀ H Vₓ + Vₓᵀ H V ≈ vR vRᵀ - vL vLᵀ.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Report struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    FSBPOperatorReport

Result returned by `check_fsbp_operator`.

# Fields
- `passed::Bool` — `true` iff all checks passed.
- `checks::Dict{String,NamedTuple}` — per-check results.  Each entry has
  fields `passed::Bool`, `error` (or relevant metric), and
  `detail::String`.
"""
struct FSBPOperatorReport
    passed::Bool
    checks::Dict{String,NamedTuple}
end

function Base.show(io::IO, r::FSBPOperatorReport)
    status = r.passed ? "ALL PASSED" : "SOME FAILED"
    println(io, "FSBPOperatorReport [$status]")
    # Sort keys for deterministic output
    for key in sort(collect(keys(r.checks)))
        c = r.checks[key]
        mark = c.passed ? "✓" : "✗"
        println(io, "  $mark $key: $(c.detail)")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_fsbp_operator(op::FSBPOperator;
                        atol=nothing, rtol=nothing,
                        rank_tol=nothing,
                        max_ref_order=4096) -> FSBPOperatorReport

Run a comprehensive verification suite on the FSBP operator `op`.

Tolerances default to precision-dependent values based on the element type
of the operator (e.g. `eps(Float64)` ≈ 1e-16, `eps(BigFloat)` depends on
the current precision setting).

# Keyword arguments
- `atol` — absolute tolerance for most checks.
- `rtol` — relative tolerance for most checks.
- `rank_tol` — tolerance for singular value ratio in rank check.
- `max_ref_order` — maximum Gauss-Legendre order for reference integrals
  in the quadrature exactness check.

# Returns
An `FSBPOperatorReport` summarising all check outcomes.
"""
function check_fsbp_operator(op::FSBPOperator{T};
                              atol = nothing,
                              rtol = nothing,
                              rank_tol = nothing,
                              max_ref_order::Int = 4096) where T
    # Default tolerances: ~100 * machine epsilon
    _atol = atol !== nothing ? T(atol) : T(100) * eps(T)
    _rtol = rtol !== nothing ? T(rtol) : T(100) * eps(T)
    _rank_tol = rank_tol !== nothing ? T(rank_tol) : T(1000) * eps(T)

    checks = Dict{String,NamedTuple}()

    # 1. Derivative exactness
    checks["Derivative exactness"] = _check_derivative_exactness(op, _atol, _rtol)

    # 2. Quadrature exactness
    checks["Quadrature exactness"] = _check_quadrature_exactness(op, _atol, _rtol, max_ref_order)

    # 3. SBP property: Q + Qᵀ ≈ E
    checks["SBP property"] = _check_sbp_property(op, _atol)

    # 4. Boundary decomposition: E ≈ tR tRᵀ - tL tLᵀ
    checks["Boundary decomposition"] = _check_boundary_decomposition(op, _atol)

    # 5. Extrapolation exactness
    checks["Extrapolation exactness"] = _check_extrapolation_exactness(op, _atol, _rtol)

    # 6. Positive weights
    checks["Positive weights"] = _check_positive_weights(op)

    # 7. Weight sum
    checks["Weight sum"] = _check_weight_sum(op, _atol)

    # 8. Nullspace consistency
    checks["Nullspace consistency"] = _check_nullspace_consistency(op, _rank_tol)

    # 9. Skew-symmetry of S
    checks["Skew-symmetry"] = _check_skew_symmetry(op, _atol)

    # 10. Quadrature/SBP compatibility for exact construction
    checks["SBP compatibility"] = _check_sbp_compatibility(op, _atol, _rtol)

    all_passed = all(c.passed for c in values(checks))
    return FSBPOperatorReport(all_passed, checks)
end

# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

"""
Check 1: D fⱼ ≈ fⱼ' for all basis functions fⱼ ∈ F.
"""
function _check_derivative_exactness(op::FSBPOperator{T}, atol, rtol) where T
    V  = eval_basis_matrix(op.op_basis, op.x)
    Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

    # Promote to working precision
    V  = Matrix{T}(V)
    Vx = Matrix{T}(Vx)

    # D * V should equal Vx (each column is one basis function)
    DV = op.D * V
    errors = [maximum(abs.(DV[:, j] - Vx[:, j])) for j in 1:op.nb]
    max_err = maximum(errors)

    tol = atol + rtol * maximum(abs.(Vx))
    passed = max_err <= tol

    n_failed = count(e -> e > tol, errors)
    detail = if passed
        "max error = $(Printf.@sprintf("%.2e", Float64(max_err))) ($(op.nb)/$(op.nb) exact)"
    else
        "max error = $(Printf.@sprintf("%.2e", Float64(max_err))) ($n_failed/$(op.nb) failed)"
    end

    return (passed=passed, error=max_err, detail=detail)
end

"""
Check 2: Quadrature exactness — reuse existing check_quadrature_exactness.
"""
function _check_quadrature_exactness(op::FSBPOperator{T}, atol, rtol, max_ref_order) where T
    report = check_quadrature_exactness(op.quad_basis, op.x, op.w;
                                        interval=op.interval,
                                        atol=atol, rtol=rtol,
                                        max_ref_order=max_ref_order)

    detail = if report.passed
        "max error = $(Printf.@sprintf("%.2e", Float64(report.max_error))) (all $(length(report.errors)) exact)"
    else
        n_failed = count(
            k -> report.errors[k] > atol + rtol * abs(report._reference_integrals[k]),
            1:length(report.errors))
        "max error = $(Printf.@sprintf("%.2e", Float64(report.max_error))) ($n_failed/$(length(report.errors)) failed)"
    end

    return (passed=report.passed, error=report.max_error, detail=detail)
end

"""
Check 3: SBP property — Q + Qᵀ ≈ E.
"""
function _check_sbp_property(op::FSBPOperator, atol)
    residual = op.Q + op.Q' - op.E
    max_err = maximum(abs.(residual))
    passed = max_err <= atol

    detail = "‖Q + Qᵀ - E‖∞ = $(Printf.@sprintf("%.2e", Float64(max_err)))"
    return (passed=passed, error=max_err, detail=detail)
end

"""
Check 4: Boundary decomposition — E ≈ tR tRᵀ - tL tLᵀ.
"""
function _check_boundary_decomposition(op::FSBPOperator, atol)
    E_from_t = op.tR * op.tR' - op.tL * op.tL'
    residual = op.E - E_from_t
    max_err = maximum(abs.(residual))
    passed = max_err <= atol

    detail = "‖E - (tR tRᵀ - tL tLᵀ)‖∞ = $(Printf.@sprintf("%.2e", Float64(max_err)))"
    return (passed=passed, error=max_err, detail=detail)
end

"""
Check 5: Extrapolation exactness — tLᵀ fⱼ ≈ fⱼ(xL), tRᵀ fⱼ ≈ fⱼ(xR).
"""
function _check_extrapolation_exactness(op::FSBPOperator{T}, atol, rtol) where T
    a, b = op.interval
    V = Matrix{T}(eval_basis_matrix(op.op_basis, op.x))

    # tLᵀ * f  should give f(a) for each basis function
    # tRᵀ * f  should give f(b) for each basis function
    vL = eval_basis_vector(op.op_basis, a)
    vR = eval_basis_vector(op.op_basis, b)

    errors_L = [abs(dot(op.tL, V[:, j]) - vL[j]) for j in 1:op.nb]
    errors_R = [abs(dot(op.tR, V[:, j]) - vR[j]) for j in 1:op.nb]

    max_err_L = maximum(errors_L)
    max_err_R = maximum(errors_R)
    max_err = max(max_err_L, max_err_R)

    tol = atol + rtol * max(maximum(abs.(vL)), maximum(abs.(vR)))
    passed = max_err <= tol

    detail = "max error: tL=$(Printf.@sprintf("%.2e", Float64(max_err_L))), tR=$(Printf.@sprintf("%.2e", Float64(max_err_R)))"
    return (passed=passed, error=max_err, detail=detail)
end

"""
Check 6: All quadrature weights are positive.
"""
function _check_positive_weights(op::FSBPOperator)
    min_w = minimum(op.w)
    passed = min_w > 0

    detail = "min weight = $(Printf.@sprintf("%.4e", Float64(min_w)))"
    return (passed=passed, error=-min_w, detail=detail)
end

"""
Check 7: Weight sum — 1ᵀ H 1 ≈ xR - xL  (exactness for constants).
"""
function _check_weight_sum(op::FSBPOperator, atol)
    a, b = op.interval
    expected = b - a
    actual = sum(op.w)
    err = abs(actual - expected)
    passed = err <= atol

    detail = "Σw = $(Printf.@sprintf("%.4e", Float64(actual))), expected $(Printf.@sprintf("%.4e", Float64(expected))), err = $(Printf.@sprintf("%.2e", Float64(err)))"
    return (passed=passed, error=err, detail=detail)
end

"""
Check 8: Nullspace consistency — rank(D) = nn - 1.

A nullspace-consistent SBP operator has null(D) = span{1}, which means
rank(D) = nn - 1.  (Glaubitz et al. 2026, Lemma 4.1.)
"""
function _check_nullspace_consistency(op::FSBPOperator, rank_tol)
    # SVD: convert to Float64 for LAPACK-based svdvals when T is BigFloat,
    # since Julia's generic SVD for BigFloat can be very slow for large matrices.
    # Because the SVD is done in Float64, the rank tolerance must also be
    # Float64-scale — using a BigFloat-derived tolerance (e.g. 1e-74) would
    # be far below Float64 machine epsilon and defeat the rank test.
    D_f64 = Matrix{Float64}(op.D)
    sv = svdvals(D_f64)
    # Count singular values above tolerance (relative to largest)
    f64_rank_tol = max(Float64(rank_tol), 1000 * eps(Float64))
    threshold = f64_rank_tol * sv[1]
    numerical_rank = count(s -> s > threshold, sv)

    expected_rank = op.nn - 1
    passed = numerical_rank == expected_rank

    detail = "rank(D) = $numerical_rank, expected $expected_rank"
    return (passed=passed, error=Float64(abs(numerical_rank - expected_rank)), detail=detail)
end

"""
Check 9: Skew-symmetry of S — S + Sᵀ ≈ 0.
"""
function _check_skew_symmetry(op::FSBPOperator, atol)
    residual = op.S + op.S'
    max_err = maximum(abs.(residual))
    passed = max_err <= atol

    detail = "‖S + Sᵀ‖∞ = $(Printf.@sprintf("%.2e", Float64(max_err)))"
    return (passed=passed, error=max_err, detail=detail)
end

"""
Check 10: Quadrature/SBP compatibility for exact diagonal-norm construction.
"""
function _check_sbp_compatibility(op::FSBPOperator{T}, atol, rtol) where T
    a, b = op.interval
    V = Matrix{T}(eval_basis_matrix(op.op_basis, op.x))
    Vx = Matrix{T}(eval_basis_derivative_matrix(op.op_basis, op.x))
    vL = eval_basis_vector(op.op_basis, a)
    vR = eval_basis_vector(op.op_basis, b)

    residual = _sbp_compatibility_residual(V, Vx, op.w, vL, vR)
    max_err = maximum(abs.(residual))
    scale = max(one(T), maximum(abs.(vR * vR' - vL * vL')))
    tol = atol + rtol * scale
    passed = max_err <= tol

    detail = "‖VᵀHVₓ + VₓᵀHV - (vR vRᵀ - vL vLᵀ)‖∞ = $(Printf.@sprintf("%.2e", Float64(max_err)))"
    return (passed=passed, error=max_err, detail=detail)
end
