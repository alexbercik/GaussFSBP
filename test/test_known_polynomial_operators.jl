using Test
using LinearAlgebra
using GaussFSBP

if !isdefined(@__MODULE__, :polynomial_basis)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Known polynomial FSBP operators" begin
    @testset "p=1 GLL operator" begin
        expected_x = [-1.0, 1.0]
        expected_w = [1.0, 1.0]

        op = build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                 principal = :upper)

        @test op.x ≈ expected_x atol = 1e-14 rtol = 1e-14
        @test op.w ≈ expected_w atol = 1e-14 rtol = 1e-14
        @test diag(op.H) ≈ expected_w atol = 1e-14 rtol = 1e-14
        @test op.tL ≈ [1.0, 0.0] atol = 1e-14 rtol = 1e-14
        @test op.tR ≈ [0.0, 1.0] atol = 1e-14 rtol = 1e-14
        @test op.D ≈ lagrange_derivative_matrix(expected_x) atol = 1e-14 rtol = 1e-14
        @test op.Q ≈ op.H * op.D atol = 1e-14 rtol = 1e-14
        @test op.Q + op.Q' ≈ op.E atol = 1e-14 rtol = 1e-14
        @test op.E ≈ op.tR * op.tR' - op.tL * op.tL' atol = 1e-14 rtol = 1e-14

        test_standard_fsbp_report(check_fsbp_operator(op; atol = 1e-12, rtol = 1e-12))
    end

    @testset "p=3 GLL operator" begin
        expected_x, expected_w = gll4_nodes_weights(Float64)

        op = build_fsbp_operator(polynomial_basis(3), polynomial_basis(5);
                                 principal = :upper)

        @test op.x ≈ expected_x atol = 1e-12 rtol = 1e-12
        @test op.w ≈ expected_w atol = 1e-12 rtol = 1e-12
        @test diag(op.H) ≈ expected_w atol = 1e-12 rtol = 1e-12
        @test op.tL ≈ [1.0, 0.0, 0.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test op.tR ≈ [0.0, 0.0, 0.0, 1.0] atol = 1e-12 rtol = 1e-12
        @test op.D ≈ lagrange_derivative_matrix(expected_x) atol = 1e-12 rtol = 1e-12
        @test op.Q + op.Q' ≈ op.E atol = 1e-12 rtol = 1e-12
        @test op.E ≈ op.tR * op.tR' - op.tL * op.tL' atol = 1e-12 rtol = 1e-12

        test_standard_fsbp_report(check_fsbp_operator(op; atol = 1e-10, rtol = 1e-10))
    end

    @testset "p=3 GL operator" begin
        expected_x, expected_w = gl4_nodes_weights(Float64)
        expected_tL = lagrange_endpoint_vector(expected_x, -1.0)
        expected_tR = lagrange_endpoint_vector(expected_x, 1.0)

        op = build_fsbp_operator(polynomial_basis(3), polynomial_basis(7);
                                 principal = :lower)

        @test op.x ≈ expected_x atol = 1e-12 rtol = 1e-12
        @test op.w ≈ expected_w atol = 1e-12 rtol = 1e-12
        @test diag(op.H) ≈ expected_w atol = 1e-12 rtol = 1e-12
        @test op.tL ≈ expected_tL atol = 1e-12 rtol = 1e-12
        @test op.tR ≈ expected_tR atol = 1e-12 rtol = 1e-12
        @test op.D ≈ lagrange_derivative_matrix(expected_x) atol = 1e-12 rtol = 1e-12
        @test op.Q + op.Q' ≈ op.E atol = 1e-12 rtol = 1e-12
        @test op.E ≈ op.tR * op.tR' - op.tL * op.tL' atol = 1e-12 rtol = 1e-12

        test_standard_fsbp_report(check_fsbp_operator(op; atol = 1e-10, rtol = 1e-10))
    end
end
