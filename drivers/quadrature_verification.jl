"""
    quadrature_verification_driver.jl

Driver demonstrating how to use `check_quadrature_exactness` to verify that a
candidate quadrature rule integrates a given basis exactly.

This driver shows two approaches:
1. Using hard-coded quadrature nodes/weights.
2. Using GeneralizedGauss.jl to compute the quadrature rule automatically.

Usage:
    julia --project=. drivers/quadrature_verification_driver.jl
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

# 3. Run the exactness check
report = check_quadrature_exactness(quad_funcs, x_candidate, w_candidate;
                                    interval = (-1.0, 1.0),
                                    atol = 1e-12,
                                    rtol = 1e-12)
println(report)

# ═════════════════════════════════════════════════════════════════════════════
# Example 2: Compute the rule via GeneralizedGauss and then verify
# ═════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Example 2: Compute a Gauss-Legendre rule via GeneralizedGauss.jl")
println("=" ^ 70)

# 1. Build a GeneralizedGauss basis (degree-3 monomials on [-1,1])
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

# 4. Also verify using a FunctionBasis wrapper
basis_obj = FunctionBasis(quad_funcs)
report3   = check_quadrature_exactness(basis_obj, x_gg, w_gg)
println("\nUsing FunctionBasis wrapper (degree 0:$p):")
println(report3)
