using Test
using GaussFSBP

@testset "FunctionBasis construction" begin
    funcs = [x -> 1.0, x -> x, x -> x^2]
    basis = FunctionBasis(funcs)

    @test nbasis(basis) == 3
    @test basis.interval == (-1.0, 1.0)
    @test basis.derivs === nothing
end

@testset "FunctionBasis with derivatives" begin
    funcs  = [x -> 1.0, x -> x, x -> x^2, x -> x^3]
    derivs = [x -> 0.0, x -> 1.0, x -> 2x, x -> 3x^2]
    basis  = FunctionBasis(funcs; derivs = derivs)

    @test nbasis(basis) == 4
    @test basis.derivs !== nothing
end

@testset "eval_basis scalar" begin
    funcs = [x -> 1.0, x -> x, x -> x^2]
    basis = FunctionBasis(funcs)

    vals = eval_basis(basis, 0.5)
    @test length(vals) == 3
    @test vals[1] ≈ 1.0
    @test vals[2] ≈ 0.5
    @test vals[3] ≈ 0.25
end

@testset "eval_basis_derivative scalar" begin
    funcs  = [x -> 1.0, x -> x, x -> x^2]
    derivs = [x -> 0.0, x -> 1.0, x -> 2x]
    basis  = FunctionBasis(funcs; derivs = derivs)

    dvals = eval_basis_derivative(basis, 0.5)
    @test length(dvals) == 3
    @test dvals[1] ≈ 0.0
    @test dvals[2] ≈ 1.0
    @test dvals[3] ≈ 1.0
end

@testset "eval_basis_derivative error without derivs" begin
    funcs = [x -> 1.0, x -> x]
    basis = FunctionBasis(funcs)

    @test_throws ErrorException eval_basis_derivative(basis, 0.0)
end

@testset "eval_basis_matrix" begin
    funcs = [x -> 1.0, x -> x, x -> x^2]
    basis = FunctionBasis(funcs)
    xnodes = [-1.0, 0.0, 1.0]

    V = eval_basis_matrix(basis, xnodes)
    @test size(V) == (3, 3)

    # V[i, j] = basis j at node i
    @test V[1, 1] ≈ 1.0   # f1(-1) = 1
    @test V[1, 2] ≈ -1.0  # f2(-1) = -1
    @test V[1, 3] ≈ 1.0   # f3(-1) = 1
    @test V[2, 2] ≈ 0.0   # f2(0)  = 0
    @test V[3, 3] ≈ 1.0   # f3(1)  = 1
end

@testset "eval_basis_derivative_matrix" begin
    funcs  = [x -> 1.0, x -> x, x -> x^2]
    derivs = [x -> 0.0, x -> 1.0, x -> 2x]
    basis  = FunctionBasis(funcs; derivs = derivs)
    xnodes = [-1.0, 0.0, 1.0]

    Vx = eval_basis_derivative_matrix(basis, xnodes)
    @test size(Vx) == (3, 3)

    # Vx[i, j] = derivative of basis j at node i
    @test Vx[1, 3] ≈ -2.0  # d/dx(x^2) at x=-1 => 2*(-1) = -2
    @test Vx[2, 3] ≈  0.0  # d/dx(x^2) at x=0  => 0
    @test Vx[3, 3] ≈  2.0  # d/dx(x^2) at x=1  => 2
end

@testset "AbstractBasis fallback errors" begin
    # A minimal concrete type that does NOT implement any interface
    struct BareTestBasis <: AbstractBasis end
    b = BareTestBasis()

    @test_throws ErrorException nbasis(b)
    @test_throws ErrorException basis_functions(b)
    @test_throws ErrorException eval_basis(b, 0.0)
    @test_throws ErrorException eval_basis_derivative(b, 0.0)
    @test_throws ErrorException eval_basis_matrix(b, [0.0])
    @test_throws ErrorException eval_basis_derivative_matrix(b, [0.0])
end

@testset "FunctionBasis interval type consistency" begin
    @test_throws ArgumentError FunctionBasis([x -> x]; interval = (BigFloat(-1), 1.0))
    @test_throws ArgumentError FunctionBasis([x -> x]; interval = (-1.0, BigFloat(1)))

    basis = FunctionBasis([x -> one(x), x -> x]; interval = (BigFloat(-1), BigFloat(1)))
    @test eltype(basis) == BigFloat

    @test_throws ArgumentError eval_basis_matrix(basis, [-1.0, 1.0])
    V = eval_basis_matrix(basis, BigFloat[-1, 1])
    @test eltype(V) == BigFloat
end

@testset "FunctionBasis mismatched derivs error" begin
    funcs  = [x -> 1.0, x -> x]
    derivs = [x -> 0.0]   # length mismatch
    @test_throws ArgumentError FunctionBasis(funcs; derivs = derivs)
end

@testset "GeneralizedGauss interop on FunctionBasis" begin
    funcs = [x -> 1.0, x -> x]
    basis = FunctionBasis(funcs)

    @test length(basis) == 2

    moments = compute_moments(basis)
    @test moments[1] ≈ 2.0
    @test moments[2] ≈ 0.0 atol=1e-12

    t_report = check_T_system(basis; num_tuples=32, verbose=false)
    @test t_report.sampled_pass
end

@testset "GeneralizedGauss rule construction on FunctionBasis" begin
    funcs  = [x -> 1.0, x -> x]
    derivs = [x -> 0.0, x -> 1.0]
    basis = FunctionBasis(funcs; derivs=derivs)

    w, x = compute_gauss_rule(basis; principal=:upper)

    @test length(w) == 2
    @test length(x) == 2
    @test sum(w) ≈ 2.0 atol=1e-12

    ect_report = check_ECT_system(basis; n_points=32, verbose=false)
    @test ect_report.sampled_constant_sign
end
