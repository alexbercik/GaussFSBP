using Test
using GaussFSBP

@testset "FunctionBasis basics" begin
    funcs = [x -> 1.0, x -> x, x -> x^2]
    derivs = [x -> 0.0, x -> 1.0, x -> 2x]
    basis = FunctionBasis(funcs; derivs = derivs)

    @test nbasis(basis) == 3
    @test basis_functions(basis) === funcs
    @test basis.interval == (-1.0, 1.0)

    @test eval_basis(basis, 0.5) ≈ [1.0, 0.5, 0.25]
    @test eval_basis_derivative(basis, 0.5) ≈ [0.0, 1.0, 1.0]

    nodes = [-1.0, 0.0, 1.0]
    V = eval_basis_matrix(basis, nodes)
    Vx = eval_basis_derivative_matrix(basis, nodes)

    # Matrix convention: rows are nodes and columns are basis functions.
    @test size(V) == (3, 3)
    @test V ≈ [1.0 -1.0 1.0;
               1.0  0.0 0.0;
               1.0  1.0 1.0]
    @test size(Vx) == (3, 3)
    @test Vx ≈ [0.0 1.0 -2.0;
                0.0 1.0  0.0;
                0.0 1.0  2.0]
end

@testset "FunctionBasis derivative and constructor errors" begin
    @test_throws ErrorException eval_basis_derivative(FunctionBasis([x -> 1.0, x -> x]), 0.0)
    @test_throws ArgumentError FunctionBasis([x -> 1.0, x -> x]; derivs = [x -> 0.0])
end

@testset "FunctionBasis interval type consistency" begin
    @test_throws ArgumentError FunctionBasis([x -> x]; interval = (BigFloat(-1), 1.0))
    @test_throws ArgumentError FunctionBasis([x -> x]; interval = (-1.0, BigFloat(1)))

    basis = FunctionBasis([x -> one(x), x -> x];
                          interval = (BigFloat(-1), BigFloat(1)))

    @test eltype(basis) == BigFloat
    @test_throws ArgumentError eval_basis_matrix(basis, [-1.0, 1.0])

    V = eval_basis_matrix(basis, BigFloat[-1, 1])
    @test eltype(V) == BigFloat
    @test V == BigFloat[1 -1; 1 1]
end

@testset "GeneralizedGauss interop on FunctionBasis" begin
    funcs = [x -> 1.0, x -> x]
    derivs = [x -> 0.0, x -> 1.0]
    basis = FunctionBasis(funcs; derivs = derivs)

    @test length(basis) == 2

    moments = compute_moments(basis)
    @test moments[1] ≈ 2.0
    @test moments[2] ≈ 0.0 atol = 1e-12

    t_report = check_T_system(basis; num_tuples = 32, verbose = false)
    @test t_report.sampled_pass

    w, x = compute_gauss_rule(basis; principal = :upper)
    @test length(w) == 2
    @test length(x) == 2
    @test sum(w) ≈ 2.0 atol = 1e-12

    # A one-function ECT check exercises the FunctionBasis interop path without
    # requiring higher derivatives than FunctionBasis stores.
    ect_basis = FunctionBasis([x -> 1.0]; derivs = [x -> 0.0])
    ect_report = check_ECT_system(ect_basis; n_points = 32, verbose = false)
    @test ect_report.sampled_constant_sign
end
