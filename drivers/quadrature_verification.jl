"""
    quadrature_verification.jl

Driver demonstrating how to use `check_quadrature_exactness` to verify that a
candidate quadrature rule integrates a given basis exactly.

This driver shows three approaches:
1. Using hard-coded quadrature nodes/weights.
2. Using GeneralizedGauss.jl to compute the quadrature rule automatically.
3. Passing exact moments directly to the verification routine.

Usage:
    julia --project=. drivers/quadrature_verification.jl
"""

using GaussFSBP

# ═════════════════════════════════════════════════════════════════════════════
# Example 1: Hand-coded 2-point Gauss-Legendre rule
# ═════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Example 1: Verify a hand-coded 2-point Gauss-Legendre rule")
println("=" ^ 70)

# 1. Define a quadrature basis
quad_funcs = [x -> 1.0, x -> x, x -> x^2, x -> x^3]

# 2. Provide the candidate rule (2-point GL, exact for degree ≤ 3)
x_candidate = [-1.0/sqrt(3.0), 1.0/sqrt(3.0)]
w_candidate = [1.0, 1.0]

# 3. Run the exactness check with adaptive reference integrals
report = check_quadrature_exactness(quad_funcs, x_candidate, w_candidate;
                                    interval = (-1.0, 1.0),
                                    atol = 1e-12,
                                    rtol = 1e-12)
println(report)

# 4. Run the same check with exact moments in quad_funcs order
quad_moments = [2.0, 0.0, 2.0 / 3.0, 0.0]
report_exact = check_quadrature_exactness(quad_funcs, x_candidate, w_candidate;
                                          interval = (-1.0, 1.0),
                                          quad_moments = quad_moments,
                                          atol = 1e-12,
                                          rtol = 1e-12)
println("\nUsing exact moments:")
println(report_exact)

# ═════════════════════════════════════════════════════════════════════════════
# Example 2: Compute the rule via GeneralizedGauss and then verify
# ═════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Example 2: Compute a Gauss-Legendre rule via GeneralizedGauss.jl")
println("=" ^ 70)

# 1. Build a GeneralizedGauss basis (degree-p monomials on [-1,1])
p = 5
funcs  = [let k = k; x -> x^k end for k in 0:p]
derivs = [let k = k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:p]
gg_basis = quadbasis(funcs, derivs, -1.0, 1.0)

# 2. Compute the quadrature rule
w_gg, x_gg = compute_gauss_rule(gg_basis)
println("Computed $(length(x_gg))-point rule:")
println("  Nodes:   $x_gg")
println("  Weights: $w_gg")

# 3. Verify it integrates products up to degree p
product_funcs = [let k = k; x -> x^k end for k in 0:p]
report2 = check_quadrature_exactness(product_funcs, x_gg, w_gg;
                                     interval = (-1.0, 1.0))
println("\nQuadrature exactness for product basis (degree 0:$(p)):")
println(report2)

# 4. Also verify the same product basis through a FunctionBasis wrapper and exact moments
basis_obj = FunctionBasis(product_funcs)
product_moments = [iseven(k) ? 2.0 / (k + 1) : 0.0 for k in 0:p]
report3 = check_quadrature_exactness(basis_obj, x_gg, w_gg;
                                     quad_moments=product_moments)
println("\nUsing FunctionBasis wrapper with exact moments (degree 0:$(p)):")
println(report3)
