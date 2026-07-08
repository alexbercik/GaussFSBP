using Test
using GaussFSBP

# Standard reference rules on [-1, 1].
const GL2_NODES = [-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)]
const GL2_WEIGHTS = [1.0, 1.0]
const GL3_NODES = [-sqrt(3.0 / 5.0), 0.0, sqrt(3.0 / 5.0)]
const GL3_WEIGHTS = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
const MIDPOINT_NODES = [0.0]
const MIDPOINT_WEIGHTS = [2.0]

const QUAD_POLYS_DEG3 = [x -> 1.0, x -> x, x -> x^2, x -> x^3]

@testset "quadrature type consistency errors" begin
    funcs = [x -> 1.0, x -> x]

    @test_throws ArgumentError check_quadrature_exactness(
        funcs, [-1.0, 1.0], [1.0, 1.0]; interval = (BigFloat(-1), BigFloat(1)))
    @test_throws ArgumentError check_quadrature_exactness(
        funcs, BigFloat[-1, 1], BigFloat[1, 1]; interval = (-1.0, 1.0))
    @test_throws ArgumentError reference_integral_gausslegendre(
        x -> 1.0, (-1.0, 1.0); T = BigFloat)
end

@testset "reference_integral_gausslegendre" begin
    constant_integral, _ = reference_integral_gausslegendre(x -> 1.0, (-1.0, 1.0))
    quadratic_integral, _ = reference_integral_gausslegendre(x -> x^2, (-1.0, 1.0))
    exponential_integral, _ = reference_integral_gausslegendre(exp, (0.0, 1.0))

    @test constant_integral ≈ 2.0
    @test quadratic_integral ≈ 2.0 / 3.0
    @test exponential_integral ≈ exp(1) - 1 atol = 1e-12
end

@testset "check_quadrature_exactness for standard rules" begin
    gl2_report = check_quadrature_exactness(QUAD_POLYS_DEG3, GL2_NODES, GL2_WEIGHTS)
    @test gl2_report isa QuadratureExactnessReport
    @test gl2_report.passed
    @test gl2_report.max_error < 1e-12
    @test gl2_report.min_weight ≈ 1.0

    funcs6 = [x -> 1.0, x -> x, x -> x^2, x -> x^3, x -> x^4, x -> x^5]
    gl3_report = check_quadrature_exactness(funcs6, GL3_NODES, GL3_WEIGHTS)
    @test gl3_report.passed
    @test gl3_report.max_error < 1e-12

    # The midpoint rule is exact through degree 1, but not for x^2.
    midpoint_report = check_quadrature_exactness(QUAD_POLYS_DEG3,
                                                 MIDPOINT_NODES, MIDPOINT_WEIGHTS)
    @test !midpoint_report.passed
    @test midpoint_report.max_error > 1e-10
end

@testset "check_quadrature_exactness inputs and exact moments" begin
    linear_basis = FunctionBasis([x -> 1.0, x -> x])
    basis_report = check_quadrature_exactness(linear_basis,
                                              MIDPOINT_NODES, MIDPOINT_WEIGHTS)
    @test basis_report.passed

    moments = [2.0, 0.0, 2.0 / 3.0, 0.0]
    gl2_report = check_quadrature_exactness(QUAD_POLYS_DEG3, GL2_NODES, GL2_WEIGHTS;
                                            quad_moments = moments)
    @test gl2_report.passed
    @test gl2_report.max_error < 1e-12
    @test gl2_report.reference_orders == zeros(Int, length(QUAD_POLYS_DEG3))
    @test gl2_report._reference_integrals ≈ moments

    midpoint_report = check_quadrature_exactness(QUAD_POLYS_DEG3,
                                                 MIDPOINT_NODES, MIDPOINT_WEIGHTS;
                                                 quad_moments = moments)
    @test !midpoint_report.passed

    @test_throws ArgumentError check_quadrature_exactness(
        QUAD_POLYS_DEG3, GL2_NODES, GL2_WEIGHTS; quad_moments = [2.0, 0.0])
end
