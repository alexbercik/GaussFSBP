using Test
using LinearAlgebra
using GaussFSBP

if !isdefined(@__MODULE__, :polynomial_basis)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "BigFloat precision" begin
    @testset "p=3 GLL polynomial operator" begin
        a, b = BigFloat(-1), BigFloat(1)
        expected_x, expected_w = gll4_nodes_weights(BigFloat)

        op = build_fsbp_operator(polynomial_basis(3; interval = (a, b)),
                                 polynomial_basis(5; interval = (a, b));
                                 orthogonalize = true,
                                 principal = :upper)

        @test op isa FSBPOperator{BigFloat}
        @test eltype(op.x) == BigFloat
        @test eltype(op.w) == BigFloat
        @test eltype(op.D) == BigFloat
        @test op.x ≈ expected_x atol = BigFloat("1e-40") rtol = BigFloat("1e-40")
        @test op.w ≈ expected_w atol = BigFloat("1e-40") rtol = BigFloat("1e-40")
        @test op.D ≈ lagrange_derivative_matrix(expected_x) atol = BigFloat("1e-40") rtol = BigFloat("1e-40")

        report = check_fsbp_operator(op; quad_moments = polynomial_moments(5, BigFloat))
        test_standard_fsbp_report(report)
        @test report.checks["Derivative exactness"].error < BigFloat("1e-30")
        @test report.checks["Weight sum"].error < BigFloat("1e-30")
    end

    @testset "Rectangular least-squares path" begin
        a, b = BigFloat(-1), BigFloat(1)

        op = build_fsbp_operator(polynomial_basis(1; interval = (a, b)),
                                 polynomial_basis(3; interval = (a, b));
                                 orthogonalize = true,
                                 principal = :upper)

        @test op isa FSBPOperator{BigFloat}
        @test op.nn == 3
        @test op.nb == 2
        @test all(op.w .> 0)
        @test op.tL == BigFloat[1, 0, 0]
        @test op.tR == BigFloat[0, 0, 1]

        report = check_fsbp_operator(op; quad_moments = polynomial_moments(3, BigFloat))
        test_standard_fsbp_report(report)
    end
end
