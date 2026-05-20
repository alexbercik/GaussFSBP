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

        verbose_output = mktemp() do _, io
            redirect_stdout(io) do
                optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs;
                                       test_functions = tests,
                                       test_derivatives = test_derivs,
                                       compatibility_action = :error,
                                       verbose = true)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end
        @test occursin("extrapolation symmetry = none", verbose_output)
        @test occursin("num free tL params", verbose_output)
        @test occursin("num free tR params", verbose_output)
        @test occursin("tL optimization active", verbose_output)
        @test occursin("tR optimization active", verbose_output)
        @test occursin("initial tL accuracy objective", verbose_output)
        @test occursin("initial tR accuracy objective", verbose_output)
        @test !occursin("initial extrapolation accuracy objective", verbose_output)
    end

    @testset "flip-symmetric extrapolation" begin
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
                                    extrapolation_symmetry = :flip,
                                    compatibility_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        vL = eval_basis_vector(op.op_basis, -1.0)
        vR = eval_basis_vector(op.op_basis, 1.0)

        @test op.tR == reverse(op.tL)
        @test norm(V' * op.tL - vL) < 1e-10
        @test norm(V' * op.tR - vR) < 1e-10

        endpoint_op = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                             [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                             -1.0, 1.0, funcs, derivs;
                                             extrapolation_symmetry = :flip,
                                             compatibility_action = :error)
        @test endpoint_op.tL == [1.0, 0.0, 0.0]
        @test endpoint_op.tR == [0.0, 0.0, 1.0]

        verbose_output = mktemp() do _, io
            redirect_stdout(io) do
                optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs;
                                       test_functions = tests,
                                       test_derivatives = test_derivs,
                                       extrapolation_symmetry = :flip,
                                       compatibility_action = :error,
                                       verbose = true)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end
        @test occursin("extrapolation symmetry = flip", verbose_output)
        @test occursin("num free coupled tL/tR params", verbose_output)
        @test occursin("coupled tL/tR optimization active", verbose_output)

        @test_throws ArgumentError optimize_fsbp_operator([-0.9, 0.0, 1.0],
                                                          [1.0, 1.0, 1.0],
                                                          -1.0, 1.0, funcs, derivs;
                                                          extrapolation_symmetry = :flip)
    end

    @testset "split extrapolation normalization" begin
        theta_acc = 0.25
        theta_norm = 0.75
        obj_tol = 1e-14

        tL0 = [1.0, -0.25]
        tR0 = [-1.5, 0.75]
        ZL = reshape([0.5, -1.0], 2, 1)
        ZR = reshape([2.0, 0.25], 2, 1)
        w = [1.0, 1.0]
        ext_tests = [
            (g_perp = [2.0, -1.0], gL_perp = 10.0, gR_perp = -1.0,
             deltaL = 5.0, deltaR = 0.5, activeL = true, activeR = true,
             omega = 3.0),
            (g_perp = [-1.0, 4.0], gL_perp = -2.0, gR_perp = 7.0,
             deltaL = 1.5, deltaR = 2.0, activeL = true, activeR = true,
             omega = 0.25),
        ]

        function expected_boundary_solution(t0, Z, side, J_acc0, J_norm0)
            rows = Float64[]
            rhs = Float64[]
            if theta_acc > 0 && J_acc0 > obj_tol
                scale = sqrt(theta_acc / J_acc0)
                for t in ext_tests
                    if side === :left && t.activeL
                        factor = sqrt(t.omega) * scale / t.deltaL
                        push!(rows, dot(vec(Z), t.g_perp) * factor)
                        push!(rhs, (dot(t0, t.g_perp) - t.gL_perp) * factor)
                    elseif side === :right && t.activeR
                        factor = sqrt(t.omega) * scale / t.deltaR
                        push!(rows, dot(vec(Z), t.g_perp) * factor)
                        push!(rhs, (dot(t0, t.g_perp) - t.gR_perp) * factor)
                    end
                end
            end
            if theta_norm > 0 && J_norm0 > obj_tol
                scale = sqrt(theta_norm / J_norm0)
                for i in eachindex(t0)
                    push!(rows, scale * Z[i, 1])
                    push!(rhs, scale * t0[i])
                end
            end
            a = -dot(rows, rhs) / dot(rows, rows)
            return t0 + vec(Z) * a
        end

        J_L = GaussFSBP._tL_extrapolation_objectives(tL0, ext_tests, w, :Euclidean)
        J_R = GaussFSBP._tR_extrapolation_objectives(tR0, ext_tests, w, :Euclidean)

        tL, tR = GaussFSBP._optimize_extrapolation(
            tL0, tR0, ZL, ZR, ext_tests, w, :Euclidean,
            theta_acc, theta_norm, J_L, J_R, obj_tol;
            tL_has_free_parameters = true, tR_has_free_parameters = true,
            rank_tol = nothing)

        expectedL = expected_boundary_solution(tL0, ZL, :left, J_L.accuracy, J_L.norm)
        expectedR = expected_boundary_solution(tR0, ZR, :right, J_R.accuracy, J_R.norm)
        combinedL = expected_boundary_solution(tL0, ZL, :left, J_L.accuracy + J_R.accuracy,
                                               J_L.norm + J_R.norm)

        @test isapprox(tL, expectedL; atol = 1e-14, rtol = 0)
        @test isapprox(tR, expectedR; atol = 1e-14, rtol = 0)
        @test norm(tL - combinedL) > 1e-3
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
