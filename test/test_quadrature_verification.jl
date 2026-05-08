using Test
using GaussFSBP

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded quadrature rules for testing
# ─────────────────────────────────────────────────────────────────────────────

# 2-point Gauss-Legendre rule on [-1, 1]
# Integrates polynomials of degree ≤ 3 exactly.
const GL2_nodes   = [-1.0/sqrt(3.0),  1.0/sqrt(3.0)]
const GL2_weights = [ 1.0,            1.0           ]

# 3-point Gauss-Legendre rule on [-1, 1]
# Integrates polynomials of degree ≤ 5 exactly.
const GL3_nodes   = [-sqrt(3.0/5.0), 0.0, sqrt(3.0/5.0)]
const GL3_weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]

# 1-point midpoint rule on [-1, 1]
# Integrates only degree ≤ 1 exactly.
const MP1_nodes   = [0.0]
const MP1_weights = [2.0]

# Polynomial callable basis used across tests
const poly_funcs = [x -> 1.0, x -> x, x -> x^2, x -> x^3]

@testset "reference_integral_gausslegendre" begin
    # ∫_{-1}^{1} 1 dx = 2
    I1, _ = reference_integral_gausslegendre(x -> 1.0, (-1.0, 1.0))
    @test I1 ≈ 2.0

    # ∫_{-1}^{1} x^2 dx = 2/3
    I2, _ = reference_integral_gausslegendre(x -> x^2, (-1.0, 1.0))
    @test I2 ≈ 2.0/3.0

    # ∫_{0}^{1} exp(x) dx = e - 1
    I3, _ = reference_integral_gausslegendre(exp, (0.0, 1.0))
    @test I3 ≈ exp(1) - 1  atol=1e-12
end

@testset "check_quadrature_exactness — 2-point GL passes for deg ≤ 3" begin
    report = check_quadrature_exactness(poly_funcs, GL2_nodes, GL2_weights)

    @test report isa QuadratureExactnessReport
    @test report.passed
    @test report.max_error < 1e-12
    @test length(report.errors) == 4
    @test report.min_weight ≈ 1.0
end

@testset "check_quadrature_exactness — 3-point GL passes for deg ≤ 5" begin
    funcs6 = [x -> 1.0, x -> x, x -> x^2, x -> x^3, x -> x^4, x -> x^5]
    report = check_quadrature_exactness(funcs6, GL3_nodes, GL3_weights)

    @test report.passed
    @test report.max_error < 1e-12
end

@testset "check_quadrature_exactness — midpoint rule fails for deg ≥ 2" begin
    # The 1-point midpoint rule is exact only for polynomials of degree ≤ 1.
    # Using [1, x, x^2, x^3] it should fail.
    report = check_quadrature_exactness(poly_funcs, MP1_nodes, MP1_weights)

    @test !report.passed
    @test report.max_error > 1e-10   # large error on x^2, x^3 terms
end

@testset "check_quadrature_exactness — FunctionBasis input" begin
    funcs = [x -> 1.0, x -> x]
    basis = FunctionBasis(funcs)
    # 1-point rule is exact for degree ≤ 1
    report = check_quadrature_exactness(basis, MP1_nodes, MP1_weights)

    @test report.passed
end

@testset "QuadratureExactnessReport show" begin
    report = check_quadrature_exactness(poly_funcs, GL2_nodes, GL2_weights)
    buf = IOBuffer()
    show(buf, report)
    str = String(take!(buf))
    @test occursin("PASSED", str)
    @test occursin("max_error", str)
end
