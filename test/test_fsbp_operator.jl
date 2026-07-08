using Test
using LinearAlgebra
using GaussFSBP

if !isdefined(@__MODULE__, :polynomial_basis)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "FSBP operator builder behavior" begin
    @testset "Rectangular GLL least-squares construction" begin
        op = build_fsbp_operator(polynomial_basis(1), polynomial_basis(3);
                                 orthogonalize = true,
                                 principal = :upper)

        @test op.nn == 3
        @test op.nb == 2
        @test op.nn > op.nb
        @test all(op.w .> 0)
        @test op.tL ≈ [1.0, 0.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test op.tR ≈ [0.0, 0.0, 1.0] atol = 1e-12 rtol = 1e-12

        test_standard_fsbp_report(check_fsbp_operator(op; atol = 1e-10, rtol = 1e-10))
    end

    @testset "Rectangular GL extrapolation boundary" begin
        op = build_fsbp_operator(polynomial_basis(1), polynomial_basis(5);
                                 orthogonalize = true,
                                 principal = :lower)

        @test op.nn == 3
        @test op.nb == 2
        @test all(abs.(op.x .+ 1.0) .> 1e-8)
        @test all(abs.(op.x .- 1.0) .> 1e-8)

        # Interior-node rectangular operators must use the extrapolation
        # boundary matrix, not an endpoint-diagonal shortcut.
        @test op.E ≈ op.tR * op.tR' - op.tL * op.tL' atol = 1e-12 rtol = 1e-12

        report = check_fsbp_operator(op; atol = 1e-10, rtol = 1e-10)
        @test report.checks["Derivative exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["Boundary decomposition"].passed
        @test report.checks["Extrapolation exactness"].passed
        @test report.checks["Skew-symmetry"].passed
    end

    @testset "Explicit quadrature moments and orthogonalization" begin
        op_basis = polynomial_basis(1)
        quad_basis = polynomial_basis(3)
        quad_moments = polynomial_moments(3, Float64)

        # Exact moments are supplied for the original quadrature basis.  The
        # builder must transform them when orthogonalization is enabled.
        op_orth = build_fsbp_operator(op_basis, quad_basis;
                                      orthogonalize = true,
                                      principal = :lower,
                                      quad_moments = quad_moments)
        op_raw = build_fsbp_operator(op_basis, quad_basis;
                                     orthogonalize = false,
                                     principal = :lower,
                                     quad_moments = quad_moments)

        @test op_orth.nn == op_raw.nn == 2
        @test op_orth.x ≈ op_raw.x atol = 1e-12 rtol = 1e-12
        @test op_orth.w ≈ op_raw.w atol = 1e-12 rtol = 1e-12

        report = check_fsbp_operator(op_orth; atol = 1e-10, rtol = 1e-10,
                                     quad_moments = quad_moments)
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["SBP property"].passed

        @test_throws ArgumentError check_fsbp_operator(op_orth; quad_moments = [2.0])
        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal = :lower,
                                                       quad_moments = [2.0, 0.0])
        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal = :lower,
                                                       quad_kwargs = (moments = quad_moments,))
    end

    @testset "SBP construction check action" begin
        op_basis = polynomial_basis(2)
        incompatible_quad_basis = polynomial_basis_from_degrees([0, 1, 3, 4])

        @test_throws ErrorException build_fsbp_operator(op_basis, incompatible_quad_basis;
                                                        principal = :upper,
                                                        sbp_check_action = :error)

        ignored = build_fsbp_operator(op_basis, incompatible_quad_basis;
                                      principal = :upper,
                                      sbp_check_action = :ignore)
        @test ignored.nn == ignored.nb == 3
        @test !check_fsbp_operator(ignored; atol = 1e-10, rtol = 1e-10).passed

        @test_throws ArgumentError build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                                       sbp_check_action = :invalid)
    end

    @testset "Optimization hook builds operator" begin
        op = build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                 principal = :upper,
                                 use_optimization = true,
                                 sbp_check_action = :error,
                                 test_functions = Function[x -> exp(x)],
                                 test_derivatives = Function[x -> exp(x)],
                                 test_weights = [2.0],
                                 extrapolation_objective_weights = (accuracy = 1 // 2, norm = 1 // 2),
                                 S_objective_weights = (accuracy = 1 // 2, norm = 1 // 2),
                                 derivative_error_norm = :H,
                                 opt_method = :sequential,
                                 extrapolation_symmetry = :none)

        @test op.nn == 2
        @test op.tL ≈ [1.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test op.tR ≈ [0.0, 1.0] atol = 1e-12 rtol = 1e-12

        report = check_fsbp_operator(op; atol = 1e-10, rtol = 1e-10)
        @test report.checks["Derivative exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["SBP compatibility"].passed
    end

    @testset "Public input validation" begin
        op_basis = FunctionBasis([x -> one(x), x -> x];
                                 derivs = [x -> zero(x), x -> one(x)],
                                 interval = (BigFloat(-1), BigFloat(1)))
        quad_basis = polynomial_basis(3)

        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal = :upper)

        @test_throws Exception build_fsbp_operator(FunctionBasis([x -> 1.0, x -> x]),
                                                   polynomial_basis(1))

        @test_throws ArgumentError build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                                       principal = :upper,
                                                       quad_kwargs = (solver_tolerance = -1.0,))
        @test_throws ArgumentError build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                                       quad_kwargs = (principal = :upper,))
        @test_throws ArgumentError build_fsbp_operator(polynomial_basis(1), polynomial_basis(1);
                                                       add_endpoint = :right,
                                                       quad_kwargs = (add_endpoint = :left,))
    end
end
