"""
    quadrature_verification_driver.jl

Driver demonstrating how to use `check_quadrature_exactness` to verify that a
candidate quadrature rule integrates a given basis exactly.

Usage:
    julia --project=. drivers/quadrature_verification_driver.jl
"""

using GaussFSBP

# ─────────────────────────────────────────────────────────────────────────────
# 1. Define a quadrature basis
#    Here we use a simple polynomial basis {1, x, x^2, x^3}.
# ─────────────────────────────────────────────────────────────────────────────

quadbasis = [x -> 1.0, x -> x, x -> x^2, x -> x^3]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Provide a candidate quadrature rule
#    Using the 2-point Gauss-Legendre rule, which is exact for degree ≤ 3.
# ─────────────────────────────────────────────────────────────────────────────

x_candidate = [-1.0/sqrt(3.0), 1.0/sqrt(3.0)]
w_candidate = [1.0, 1.0]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run the exactness check
# ─────────────────────────────────────────────────────────────────────────────

report = check_quadrature_exactness(quadbasis, x_candidate, w_candidate;
                                    interval = (-1.0, 1.0),
                                    atol = 1e-12,
                                    rtol = 1e-12)

println(report)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Optionally wrap the basis in a FunctionBasis for richer introspection
# ─────────────────────────────────────────────────────────────────────────────

basis_obj = FunctionBasis(quadbasis)
report2   = check_quadrature_exactness(basis_obj, x_candidate, w_candidate)

println("\nUsing FunctionBasis wrapper:")
println(report2)
