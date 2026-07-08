"""
    build_operator.jl

Driver demonstrating end-to-end construction and verification of FSBP
operators using `build_fsbp_operator` and `check_fsbp_operator`.

Several examples are shown, including polynomial, square-root, and exponential
bases.  The square-root example demonstrates passing explicit quadrature
moments for endpoint-singular functions.

Usage:
    julia --project=. drivers/build_operator.jl
"""

using GaussFSBP


# ═════════════════════════════════════════════════════════════════════════════
# Example 1: Polynomial basis — degree 3
#   F = {1, x, x², x³}                  (nb = 4)
#   G = {1, x, x², x³, x⁴, x⁵}  (6 functions → GLL → 4 nodes)
# ═════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Example 1: Polynomial Gauss-Legendre-Lobatto p=3 operator")
println("=" ^ 70)

p = 3
funcs  = [let k=k; x -> x^k end for k in 0:p]
derivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:p]
op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

qfuncs  = [let k=k; x -> x^k end for k in 0:(2p - 1)]
qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:(2p - 1)]
quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

println("Approximation basis: degree-$p polynomials, $(nbasis(op_basis)) functions")
println("Quadrature basis: $(nbasis(quad_basis)) functions")

fsbp = build_fsbp_operator(op_basis, quad_basis; orthogonalize=true, principal=:upper)
println("\nConstructed operator:")
println(fsbp)

println("\nNodes:   $(fsbp.x)")
println("Weights: $(fsbp.w)")
println("\nD = ")
display(round.(fsbp.D; digits=6))
println("\ntL = ")
display(round.(fsbp.tL'; digits=6))
println()

report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
println("\nVerification:")
println(report)

print_fsbp_operator_python(fsbp; num_digits=16)


# ═════════════════════════════════════════════════════════════════════════════
# Example 2: Polynomial basis — degree 3
#   F = {1, x, x², x³}                  (nb = 4)
#   G = {1, x, x², x³, x⁴, x⁵, x⁶, x⁷}  (8 functions → GL → 4 nodes)
# ═════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Example 2: Polynomial Gauss-Legendre p=3 operator")
println("=" ^ 70)

p = 3
#funcs  = [let k=k; x -> x^k end for k in 0:p]
#derivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:p]
funcs, derivs = legendre_functions(p+1, -1.0, 1.0)
op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

#qfuncs  = [let k=k; x -> x^k end for k in 0:(2p + 1)]
#qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:(2p + 1)]
qfuncs, qderivs = legendre_functions(2p+2, -1.0, 1.0)
quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

println("Approximation basis: degree-$p polynomials, $(nbasis(op_basis)) functions")
println("Quadrature basis: $(nbasis(quad_basis)) functions")

fsbp = build_fsbp_operator(op_basis, quad_basis; orthogonalize=false, principal=:lower, verbose=false)
println("\nConstructed operator:")
println(fsbp)

println("\nNodes:   $(fsbp.x)")
println("Weights: $(fsbp.w)")
println("\nD = ")
display(round.(fsbp.D; digits=6))
println("\ntL = ")
display(round.(fsbp.tL'; digits=6))
println()

report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
println("\nVerification:")
println(report)


# ═════════════════════════════════════════════════════════════════════════════
# Example 3: Square-root basis with endpoint-singular quadrature moments
#   F = {1, sqrt(1-x), x, x²}
#   G = {1, x, x², x³, 1/sqrt(1-x), x/sqrt(1-x), x²/sqrt(1-x)}
#   The singular q-basis moments are supplied explicitly to avoid adaptive
#   reference-integration error near x = 1.
# ═════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Example 3: Square-root basis with explicit moments")
println("=" ^ 70)

sqrt_quad_kwargs = (lost_digits = 5, add_endpoint = :left)

setprecision(BigFloat, 32; base=10) do
    ref = BigFloat(-1), BigFloat(1)

    funcs = [
        x -> BigFloat(1),
        x -> sqrt(BigFloat(1) - x),
        x -> x,
        x -> x^2,
    ]
    derivs = [
        x -> BigFloat(0),
        x -> -BigFloat(1) / (BigFloat(2) * sqrt(BigFloat(1) - x)),
        x -> BigFloat(1),
        x -> BigFloat(2) * x,
    ]
    op_basis = FunctionBasis(funcs; derivs=derivs, interval=ref)

    qfuncs = [
        x -> BigFloat(1),
        x -> x,
        x -> x^2,
        x -> x^3,
        x -> BigFloat(1) / sqrt(BigFloat(1) - x),
        x -> x / sqrt(BigFloat(1) - x),
        x -> x^2 / sqrt(BigFloat(1) - x),
    ]
    qderivs = [
        x -> BigFloat(0),
        x -> BigFloat(1),
        x -> BigFloat(2) * x,
        x -> BigFloat(3) * x^2,
        x -> BigFloat(1) / (BigFloat(2) * (BigFloat(1) - x) *
                            sqrt(BigFloat(1) - x)),
        x -> (BigFloat(2) - x) /
             (BigFloat(2) * (BigFloat(1) - x) * sqrt(BigFloat(1) - x)),
        x -> x * (BigFloat(4) - BigFloat(3) * x) /
             (BigFloat(2) * (BigFloat(1) - x) * sqrt(BigFloat(1) - x)),
    ]
    quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=ref)

    # Exact moments for the q-basis above, in qfuncs order.
    quad_moments = BigFloat[
        2,
        0,
        BigFloat(2) / 3,
        0,
        2 * sqrt(BigFloat(2)),
        2 * sqrt(BigFloat(2)) / 3,
        14 * sqrt(BigFloat(2)) / 15,
    ]

    println("Approximation basis: square-root basis, $(nbasis(op_basis)) functions")
    println("Quadrature basis: $(nbasis(quad_basis)) functions")

    fsbp = build_fsbp_operator(op_basis, quad_basis; orthogonalize=true,
        principal=:lower, use_optimization=false, verbose=false,
        quad_moments=quad_moments,
        quad_kwargs=sqrt_quad_kwargs)
    println("\nConstructed operator:")
    println(fsbp)

    println("\nNodes:   $(fsbp.x)")
    println("Weights: $(fsbp.w)")
    println("\nD = ")
    display(round.(fsbp.D; digits=6))
    println("\ntL = ")
    display(round.(fsbp.tL'; digits=6))
    println()

    report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10,
                                 quad_moments=quad_moments)
    println("\nVerification:")
    println(report)

    print_fsbp_operator_python(fsbp; num_digits=16)
end

# ═════════════════════════════════════════════════════════════════════════════
# Example 4: Exponential basis (no optimization) — degree 2+1
#   F = {1, x, x², e^x}                  (nb = 4)
#   G = {1, x, x², x³, e^x, xe^x, x²e^x, e^2x}  (8 functions → GL → 4 nodes)
# ═════════════════════════════════════════════════════════════════════════════
println("=" ^ 70)
println("Example 4: Exponential-GL p=2+1 operator")
println("=" ^ 70)
setprecision(BigFloat, 20; base=10) do
    ref = BigFloat(-1), BigFloat(1)
    p = 2
    funcs_poly  = [let k=k; x -> x^k end for k in 0:p]
    derivs_poly = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:p]
    func_exp = exp
    deriv_exp = exp
    funcs = vcat(funcs_poly, func_exp)
    derivs = vcat(derivs_poly, deriv_exp)
    op_basis = FunctionBasis(funcs; derivs=derivs, interval=ref)

    qfuncs_poly  = [let k=k; x -> x^k end for k in 0:(2p - 1)]
    qderivs_poly = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:(2p - 1)]
    qfuncs_exp = [x -> x^i * exp(x) for i in 0:p]
    qderivs_exp = vcat(exp,
                    [x -> (i * x^(i - 1) + x^i) * exp(x) for i in 1:p])
    qfuncs_exp2 = x -> exp(2 * x)
    qderivs_exp2 = x -> 2 * exp(2 * x)
    qfuncs = vcat(qfuncs_poly, qfuncs_exp, qfuncs_exp2)
    qderivs = vcat(qderivs_poly, qderivs_exp, qderivs_exp2)
    quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=ref)

    println("Approximation basis: degree-$p polynomials + e^x, $(nbasis(op_basis)) functions")
    println("Quadrature basis: $(nbasis(quad_basis)) functions")

    fsbp = build_fsbp_operator(op_basis, quad_basis; orthogonalize=true,
        principal=:lower, use_optimization=false, verbose=false,
        quad_kwargs=(add_endpoint=:left, lost_digits=8))
    println("\nConstructed operator:")
    println(fsbp)

    println("\nNodes:   $(fsbp.x)")
    println("Weights: $(fsbp.w)")
    println("\nD = ")
    display(round.(fsbp.D; digits=6))
    println("\ntL = ")
    display(round.(fsbp.tL'; digits=6))
    println()

    report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
    println("\nVerification:")
    println(report)
end

# ═════════════════════════════════════════════════════════════════════════════
# Example 5: Exponential basis (needs optimization) — degree 2+1
#   F = {1, x, x², x³, e^x}                  (nb = 5)
#   G = {1, x, x², x³, x⁴, x⁵, e^x, xe^x, x²e^x, x³e^x, e^2x, x⁶}  (12 functions → GLL → 6 nodes)
# ═════════════════════════════════════════════════════════════════════════════
# This ill-conditioned basis needs a per-call lost-digit allowance in the
# GeneralizedGauss nonlinear solves.
quad_kwargs = (lost_digits = 12, add_endpoint = :left)

println("=" ^ 70)
println("Example 5: Exponential-GLL p=3+1 operator")
println("=" ^ 70)
setprecision(BigFloat, 24; base=10) do
    ref = BigFloat(-1), BigFloat(1)
    p = 3
    #funcs_poly  = [let k=k; x -> x^k end for k in 0:p]
    #derivs_poly = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:p]
    funcs_poly, derivs_poly = legendre_functions(p+1, ref)
    func_exp = exp
    deriv_exp = exp
    funcs = vcat(funcs_poly, func_exp)
    derivs = vcat(derivs_poly, deriv_exp)
    op_basis = FunctionBasis(funcs; derivs=derivs, interval=ref)

    #qfuncs_poly  = [let k=k; x -> x^k end for k in 0:(2p )]#- 1)]
    #qderivs_poly = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:(2p )]#- 1)]
    qfuncs_poly, qderivs_poly = legendre_functions(2p+1, ref)
    qfuncs_exp = [x -> x^i * exp(x) for i in 0:p]
    qderivs_exp = vcat(exp,
                    [x -> (i * x^(i - 1) + x^i) * exp(x) for i in 1:p])
    qfuncs_exp2 = x -> exp(2 * x)
    qderivs_exp2 = x -> 2 * exp(2 * x)
    qfuncs = vcat(qfuncs_poly, qfuncs_exp, qfuncs_exp2)
    qderivs = vcat(qderivs_poly, qderivs_exp, qderivs_exp2)
    quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=ref)

    println("Approximation basis: degree-$p polynomials + e^x, $(nbasis(op_basis)) functions")
    println("Quadrature basis: $(nbasis(quad_basis)) functions")

    opt_funcs = [x -> x^i * exp(x) for i in p+1:p+2]
    opt_derivs = [x -> (i * x^(i - 1) + x^i) * exp(x) for i in p+1:p+2]
    test_weights = ones(length(opt_funcs))
    extrap_weights = (accuracy = 1//2, norm = 1//2)
    S_weights = (accuracy = 1//2, norm = 1//2)

    fsbp = build_fsbp_operator(op_basis, quad_basis; orthogonalize=true,
        principal=:lower, use_optimization=true, opt_method=:simultaneous,
        simultaneous_num_starts=30,
        verbose=true, quad_kwargs=(; quad_kwargs..., verbose=true),
        test_functions=opt_funcs, test_derivatives=opt_derivs, test_weights=test_weights,
        extrapolation_objective_weights=extrap_weights, S_objective_weights=S_weights)
    println("\nConstructed operator:")
    println(fsbp)

    println("\nNodes:   $(fsbp.x)")
    println("Weights: $(fsbp.w)")
    println("\nD = ")
    display(round.(fsbp.D; digits=6))
    println("\ntL = ")
    display(round.(fsbp.tL'; digits=6))
    println()

    report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
    println("\nVerification:")
    println(report)
end
