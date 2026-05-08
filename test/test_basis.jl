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

@testset "FunctionBasis mismatched derivs error" begin
    funcs  = [x -> 1.0, x -> x]
    derivs = [x -> 0.0]   # length mismatch
    @test_throws ArgumentError FunctionBasis(funcs; derivs = derivs)
end
