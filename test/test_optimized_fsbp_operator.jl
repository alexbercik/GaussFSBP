using Test
using GaussFSBP
using LinearAlgebra

@testset "Optimized FSBP Operator Construction" begin

    @testset "Known Simpson nodes and weights" begin
        x = [-1.0, 0.0, 1.0]
        w = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
        funcs = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        tests = [x -> x^2]
        test_derivs = [x -> 2.0 * x]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs;
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    compatibility_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

        @test op isa FSBPOperator{Float64}
        @test op.tL == [1.0, 0.0, 0.0]
        @test op.tR == [0.0, 0.0, 1.0]
        @test norm(op.D * V - Vx) < 1e-10
        @test norm(op.Q + op.Q' - op.E) < 1e-10
        @test norm(op.S + op.S') < 1e-10
    end

    @testset "Known GL nodes with free parameters and tests" begin
        a = sqrt(3.0 / 7.0 + 2.0 * sqrt(6.0 / 5.0) / 7.0)
        b = sqrt(3.0 / 7.0 - 2.0 * sqrt(6.0 / 5.0) / 7.0)
        w_outer = (18.0 - sqrt(30.0)) / 36.0
        w_inner = (18.0 + sqrt(30.0)) / 36.0
        x = [-a, -b, b, a]
        w = [w_outer, w_inner, w_inner, w_outer]

        funcs = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        tests = [x -> x^2, x -> x^3]
        test_derivs = [x -> 2.0 * x, x -> 3.0 * x^2]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs;
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    compatibility_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)
        vL = eval_basis_vector(op.op_basis, -1.0)
        vR = eval_basis_vector(op.op_basis, 1.0)

        @test norm(V' * op.tL - vL) < 1e-10
        @test norm(V' * op.tR - vR) < 1e-10
        @test norm(op.D * V - Vx) < 1e-10
        @test norm(op.Q + op.Q' - op.E) < 1e-10
    end

    @testset "type mismatch errors" begin
        x = [-1.0, 1.0]
        w = [1.0, 1.0]
        funcs = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]

        @test_throws ArgumentError optimize_fsbp_operator(x, w, BigFloat(-1), BigFloat(1),
                                                          funcs, derivs)
        @test_throws ArgumentError optimize_fsbp_operator(BigFloat[-1, 1], BigFloat[1, 1],
                                                          -1.0, 1.0, funcs, derivs)
    end

    @testset "BigFloat known endpoints" begin
        a, b = BigFloat(-1), BigFloat(1)
        x = BigFloat[-1, 1]
        w = BigFloat[1, 1]
        funcs = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]

        op = optimize_fsbp_operator(x, w, a, b, funcs, derivs;
                                    compatibility_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

        @test op isa FSBPOperator{BigFloat}
        @test op.tL == BigFloat[1, 0]
        @test op.tR == BigFloat[0, 1]
        @test norm(op.D * V - Vx) < BigFloat("1e-30")
        @test norm(op.Q + op.Q' - op.E) < BigFloat("1e-30")
    end
end
