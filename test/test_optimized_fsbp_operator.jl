using Test
using GaussFSBP
using LinearAlgebra

_optimize_test_quad_basis(funcs, derivs, xL = -1.0, xR = 1.0) =
    FunctionBasis(funcs; derivs = derivs, interval = (xL, xR))

@testset "Optimized FSBP Operator Construction" begin

    @testset "Known Simpson nodes and weights" begin
        x = [-1.0, 0.0, 1.0]
        w = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
        funcs = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        tests = [x -> x^2]
        test_derivs = [x -> 2.0 * x]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    sbp_check_action = :error)

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

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    sbp_check_action = :error)

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
                optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                       test_functions = tests,
                                       test_derivatives = test_derivs,
                                       sbp_check_action = :error,
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

    @testset "simultaneous optimization path" begin
        funcs = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]
        tests = [x -> x^2, x -> x^3]
        test_derivs = [x -> 2 * x, x -> 3 * x^2]

        function default_simultaneous_objective(op, initial_op)
            T = eltype(op.x)
            V = eval_basis_matrix(op.op_basis, op.x)
            Vx = eval_basis_derivative_matrix(op.op_basis, op.x)
            vL = eval_basis_vector(op.op_basis, op.interval[1])
            vR = eval_basis_vector(op.op_basis, op.interval[2])
            weights = ones(T, length(tests))
            M = GaussFSBP._exactness_gram_matrix(V, op.w)
            samples = GaussFSBP._precompute_test_orthogonal_samples(
                tests, test_derivs, op.x, op.interval[1], op.interval[2],
                V, Vx, op.w, vL, vR, M)
            ext_tests = GaussFSBP._build_extrapolation_tests(
                samples, weights, op.w, :fallback, sqrt(eps(T)))
            der_tests = GaussFSBP._build_derivative_tests(samples, weights, op.w, sqrt(eps(T)))
            obj_tol = sqrt(eps(T))
            theta = one(T) / T(2)

            JL0 = GaussFSBP._tL_extrapolation_objectives(initial_op.tL, ext_tests, op.w, :Hinv)
            JR0 = GaussFSBP._tR_extrapolation_objectives(initial_op.tR, ext_tests, op.w, :Hinv)
            JL = GaussFSBP._tL_extrapolation_objectives(op.tL, ext_tests, op.w, :Hinv)
            JR = GaussFSBP._tR_extrapolation_objectives(op.tR, ext_tests, op.w, :Hinv)
            Jup0 = initial_op.D + GaussFSBP._divide_rows(initial_op.tL * initial_op.tL',
                                                        initial_op.w)
            Jup = op.D + GaussFSBP._divide_rows(op.tL * op.tL', op.w)
            Jder0 = GaussFSBP._derivative_objective(initial_op.D, der_tests, op.w, :H)
            Jder = GaussFSBP._derivative_objective(op.D, der_tests, op.w, :H)
            Jnorm0 = GaussFSBP._full_jacobian_objective(Jup0)
            Jnorm = GaussFSBP._full_jacobian_objective(Jup)

            total = zero(T)
            JL0.accuracy > obj_tol && (total += theta * JL.accuracy / JL0.accuracy)
            JR0.accuracy > obj_tol && (total += theta * JR.accuracy / JR0.accuracy)
            JL0.norm > obj_tol && (total += theta * JL.norm / JL0.norm)
            JR0.norm > obj_tol && (total += theta * JR.norm / JR0.norm)
            Jder0 > obj_tol && (total += theta * Jder / Jder0)
            Jnorm0 > obj_tol && (total += theta * Jnorm / Jnorm0)
            return total
        end

        fixed_seq = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                           [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                           -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                           opt_method = :sequential,
                                           sbp_check_action = :error)
        fixed_sim = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                           [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                           -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                           opt_method = :simultaneous,
                                           sbp_check_action = :error)
        @test fixed_sim.tL == fixed_seq.tL
        @test fixed_sim.tR == fixed_seq.tR
        @test fixed_sim.S ≈ fixed_seq.S
        @test fixed_sim.D ≈ fixed_seq.D

        a2 = 1.0 / sqrt(3.0)
        nofree_output = mktemp() do _, io
            redirect_stdout(io) do
                optimize_fsbp_operator([-a2, a2], [1.0, 1.0],
                                       -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                       opt_method = :simultaneous,
                                       sbp_check_action = :error,
                                       verbose = true)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end
        nofree = optimize_fsbp_operator([-a2, a2], [1.0, 1.0],
                                        -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                        opt_method = :simultaneous,
                                        sbp_check_action = :error)
        V = eval_basis_matrix(nofree.op_basis, nofree.x)
        Vx = eval_basis_derivative_matrix(nofree.op_basis, nofree.x)
        @test occursin("coupled nullspace dimension is zero", nofree_output)
        @test norm(nofree.D * V - Vx) < 1e-10

        @testset "global start search diversity" begin
            base_starts = [zeros(2)]
            data = (; x0 = zeros(3))
            residual = y -> [y[1]^2 - 1, y[2]]
            starts, info = GaussFSBP._simultaneous_global_search_starts(
                base_starts, 3, Float64, data, residual;
                global_num_candidates = 4,
                norm_growth_limit = 3.0,
                radial_steps = 2)
            @test length(starts) == 3
            @test info.selected_count == 3
            @test any(y -> y[1] > 0.75 && abs(y[2]) < 1e-12, starts)
            @test any(y -> y[1] < -0.75 && abs(y[2]) < 1e-12, starts)

            big_base_starts = [zeros(BigFloat, 2)]
            big_data = (; x0 = zeros(BigFloat, 3))
            big_residual = y -> BigFloat[y[1]^2 - one(BigFloat), y[2]]
            big_starts, _ = GaussFSBP._simultaneous_global_search_starts(
                big_base_starts, 3, BigFloat, big_data, big_residual;
                global_num_candidates = 4,
                norm_growth_limit = BigFloat(3),
                radial_steps = 2)
            @test all(y -> eltype(y) === BigFloat, big_starts)
            @test any(y -> y[1] > BigFloat("0.75"), big_starts)
            @test any(y -> y[1] < BigFloat("-0.75"), big_starts)
        end

        onefixed = optimize_fsbp_operator([-1.0, 0.25, 0.75], [0.6, 0.9, 0.5],
                                          -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                          test_functions = [tests[1]],
                                          test_derivatives = [test_derivs[1]],
                                          opt_method = :simultaneous,
                                          sbp_check_action = :error,
                                          simultaneous_num_starts = 3)
        V = eval_basis_matrix(onefixed.op_basis, onefixed.x)
        Vx = eval_basis_derivative_matrix(onefixed.op_basis, onefixed.x)
        vR = eval_basis_vector(onefixed.op_basis, 1.0)
        @test onefixed.tL == [1.0, 0.0, 0.0]
        @test norm(V' * onefixed.tR - vR) < 1e-10
        @test norm(onefixed.D * V - Vx) < 1e-10

        a4 = sqrt(3.0 / 7.0 + 2.0 * sqrt(6.0 / 5.0) / 7.0)
        b4 = sqrt(3.0 / 7.0 - 2.0 * sqrt(6.0 / 5.0) / 7.0)
        w_outer = (18.0 - sqrt(30.0)) / 36.0
        w_inner = (18.0 + sqrt(30.0)) / 36.0
        x4 = [-a4, -b4, b4, a4]
        w4 = [w_outer, w_inner, w_inner, w_outer]
        min_op = optimize_fsbp_operator(x4, w4, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                        test_functions = tests,
                                        test_derivatives = test_derivs,
                                        opt_method = :simultaneous,
                                        simultaneous_num_starts = 1,
                                        simultaneous_max_iter = 0,
                                        sbp_check_action = :error)
        sim_op = optimize_fsbp_operator(x4, w4, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                        test_functions = tests,
                                        test_derivatives = test_derivs,
                                        opt_method = :simultaneous,
                                        simultaneous_num_starts = 4,
                                        sbp_check_action = :error)
        seq_op = optimize_fsbp_operator(x4, w4, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                        test_functions = tests,
                                        test_derivatives = test_derivs,
                                        opt_method = :sequential,
                                        sbp_check_action = :error)
        sim_from_seq = optimize_fsbp_operator(x4, w4, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                              test_functions = tests,
                                              test_derivatives = test_derivs,
                                              opt_method = :simultaneous,
                                              simultaneous_init = :sequential,
                                              simultaneous_num_starts = 1,
                                              sbp_check_action = :error)
        V = eval_basis_matrix(sim_op.op_basis, sim_op.x)
        Vx = eval_basis_derivative_matrix(sim_op.op_basis, sim_op.x)
        @test norm(sim_op.D * V - Vx) < 1e-10
        @test norm(sim_op.Q + sim_op.Q' - sim_op.E) < 1e-10
        @test default_simultaneous_objective(sim_op, min_op) <=
              default_simultaneous_objective(min_op, min_op) + 1e-10
        @test default_simultaneous_objective(sim_from_seq, min_op) <=
              default_simultaneous_objective(seq_op, min_op) + 1e-10

        big_a = inv(sqrt(BigFloat(3)))
        big_op = optimize_fsbp_operator(BigFloat[-big_a, big_a], BigFloat[1, 1],
                                        BigFloat(-1), BigFloat(1), funcs, derivs,
                                        _optimize_test_quad_basis(funcs, derivs, BigFloat(-1), BigFloat(1));
                                        opt_method = :simultaneous,
                                        sbp_check_action = :error)
        V = eval_basis_matrix(big_op.op_basis, big_op.x)
        Vx = eval_basis_derivative_matrix(big_op.op_basis, big_op.x)
        @test big_op isa FSBPOperator{BigFloat}
        @test eltype(big_op.D) === BigFloat
        @test norm(big_op.D * V - Vx) < BigFloat("1e-30")

        big_onefixed = optimize_fsbp_operator(
            BigFloat[-1, BigFloat("0.25"), BigFloat("0.75")],
            BigFloat[BigFloat("0.6"), BigFloat("0.9"), BigFloat("0.5")],
            BigFloat(-1), BigFloat(1), funcs, derivs,
            _optimize_test_quad_basis(funcs, derivs, BigFloat(-1), BigFloat(1));
            test_functions = [tests[1]],
            test_derivatives = [test_derivs[1]],
            opt_method = :simultaneous,
            simultaneous_num_starts = 1,
            sbp_check_action = :error)
        V = eval_basis_matrix(big_onefixed.op_basis, big_onefixed.x)
        Vx = eval_basis_derivative_matrix(big_onefixed.op_basis, big_onefixed.x)
        @test big_onefixed isa FSBPOperator{BigFloat}
        @test eltype(big_onefixed.tR) === BigFloat
        @test big_onefixed.tL == BigFloat[1, 0, 0]
        @test norm(big_onefixed.D * V - Vx) < BigFloat("1e-30")
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

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    extrapolation_symmetry = :flip,
                                    sbp_check_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        vL = eval_basis_vector(op.op_basis, -1.0)
        vR = eval_basis_vector(op.op_basis, 1.0)

        @test op.tR == reverse(op.tL)
        @test norm(V' * op.tL - vL) < 1e-10
        @test norm(V' * op.tR - vR) < 1e-10

        endpoint_op = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                             [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                             -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                             extrapolation_symmetry = :flip,
                                             sbp_check_action = :error)
        @test endpoint_op.tL == [1.0, 0.0, 0.0]
        @test endpoint_op.tR == [0.0, 0.0, 1.0]

        verbose_output = mktemp() do _, io
            redirect_stdout(io) do
                optimize_fsbp_operator(x, w, -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                       test_functions = tests,
                                       test_derivatives = test_derivs,
                                       extrapolation_symmetry = :flip,
                                       sbp_check_action = :error,
                                       verbose = true)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end
        @test occursin("extrapolation symmetry = flip", verbose_output)
        @test occursin("opt_method = :simultaneous", verbose_output)
        @test occursin("num free coupled tL/tR params", verbose_output)
        @test occursin("coupled tL/tR optimization active", verbose_output)
        @test !occursin("extrapolation_symmetry=:flip uses the existing sequential", verbose_output)

        @test_throws ErrorException optimize_fsbp_operator([-0.9, 0.0, 1.0],
                                                           [1.0, 1.0, 1.0],
                                                           -1.0, 1.0, funcs, derivs, _optimize_test_quad_basis(funcs, derivs);
                                                           extrapolation_symmetry = :flip,
                                                           sbp_check_action = :error)
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

        @test_throws ArgumentError optimize_fsbp_operator(
            x, w, BigFloat(-1), BigFloat(1), funcs, derivs,
            _optimize_test_quad_basis(funcs, derivs, BigFloat(-1), BigFloat(1)))
        @test_throws ArgumentError optimize_fsbp_operator(
            BigFloat[-1, 1], BigFloat[1, 1], -1.0, 1.0, funcs, derivs,
            _optimize_test_quad_basis(funcs, derivs))
    end

    @testset "BigFloat known endpoints" begin
        a, b = BigFloat(-1), BigFloat(1)
        x = BigFloat[-1, 1]
        w = BigFloat[1, 1]
        funcs = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]

        op = optimize_fsbp_operator(x, w, a, b, funcs, derivs,
                                    _optimize_test_quad_basis(funcs, derivs, a, b);
                                    sbp_check_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

        @test op isa FSBPOperator{BigFloat}
        @test op.tL == BigFloat[1, 0]
        @test op.tR == BigFloat[0, 1]
        @test norm(op.D * V - Vx) < BigFloat("1e-30")
        @test norm(op.Q + op.Q' - op.E) < BigFloat("1e-30")
    end
end
