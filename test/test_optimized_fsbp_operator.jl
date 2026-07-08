using Test
using LinearAlgebra
using GaussFSBP

if !isdefined(@__MODULE__, :polynomial_basis)
    include(joinpath(@__DIR__, "test_helpers.jl"))
end

@testset "Optimized FSBP operator construction" begin
    @testset "Known Simpson nodes and weights" begin
        x = [-1.0, 0.0, 1.0]
        w = [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
        basis = polynomial_basis(1)
        tests = Function[x -> x^2]
        test_derivs = Function[x -> 2.0 * x]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, basis, basis;
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    sbp_check_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

        @test op isa FSBPOperator{Float64}
        @test op.tL ≈ [1.0, 0.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test op.tR ≈ [0.0, 0.0, 1.0] atol = 1e-12 rtol = 1e-12
        @test norm(op.D * V - Vx) < 1e-10
        @test norm(op.Q + op.Q' - op.E) < 1e-10
        @test norm(op.S + op.S') < 1e-10
    end

    @testset "Known GL nodes with free parameters" begin
        x, w = gl4_nodes_weights(Float64)
        basis = polynomial_basis(1)
        tests = Function[x -> x^2, x -> x^3]
        test_derivs = Function[x -> 2.0 * x, x -> 3.0 * x^2]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, basis, basis;
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
    end

    @testset "Simultaneous optimization path" begin
        basis = polynomial_basis(1)
        tests = Function[x -> x^2, x -> x^3]
        test_derivs = Function[x -> 2.0 * x, x -> 3.0 * x^2]

        fixed_seq = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                           [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                           -1.0, 1.0, basis, basis;
                                           opt_method = :sequential,
                                           sbp_check_action = :error)
        fixed_sim = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                           [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                           -1.0, 1.0, basis, basis;
                                           opt_method = :simultaneous,
                                           sbp_check_action = :error)
        @test fixed_sim.tL ≈ fixed_seq.tL atol = 1e-12 rtol = 1e-12
        @test fixed_sim.tR ≈ fixed_seq.tR atol = 1e-12 rtol = 1e-12
        @test fixed_sim.S ≈ fixed_seq.S atol = 1e-12 rtol = 1e-12
        @test fixed_sim.D ≈ fixed_seq.D atol = 1e-12 rtol = 1e-12

        a = inv(sqrt(3.0))
        nofree = optimize_fsbp_operator([-a, a], [1.0, 1.0],
                                        -1.0, 1.0, basis, basis;
                                        opt_method = :simultaneous,
                                        sbp_check_action = :error)
        V = eval_basis_matrix(nofree.op_basis, nofree.x)
        Vx = eval_basis_derivative_matrix(nofree.op_basis, nofree.x)
        @test norm(nofree.D * V - Vx) < 1e-10
        @test norm(nofree.Q + nofree.Q' - nofree.E) < 1e-10

        onefixed = optimize_fsbp_operator([-1.0, 0.25, 0.75], [0.6, 0.9, 0.5],
                                          -1.0, 1.0, basis, basis;
                                          test_functions = Function[tests[1]],
                                          test_derivatives = Function[test_derivs[1]],
                                          opt_method = :simultaneous,
                                          simultaneous_num_starts = 1,
                                          sbp_check_action = :error)
        V = eval_basis_matrix(onefixed.op_basis, onefixed.x)
        Vx = eval_basis_derivative_matrix(onefixed.op_basis, onefixed.x)
        vR = eval_basis_vector(onefixed.op_basis, 1.0)
        @test onefixed.tL ≈ [1.0, 0.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test norm(V' * onefixed.tR - vR) < 1e-10
        @test norm(onefixed.D * V - Vx) < 1e-10

        x4, w4 = gl4_nodes_weights(Float64)
        sim_op = optimize_fsbp_operator(x4, w4, -1.0, 1.0, basis, basis;
                                        test_functions = tests,
                                        test_derivatives = test_derivs,
                                        opt_method = :simultaneous,
                                        simultaneous_num_starts = 2,
                                        sbp_check_action = :error)
        V = eval_basis_matrix(sim_op.op_basis, sim_op.x)
        Vx = eval_basis_derivative_matrix(sim_op.op_basis, sim_op.x)
        @test norm(sim_op.D * V - Vx) < 1e-10
        @test norm(sim_op.Q + sim_op.Q' - sim_op.E) < 1e-10
    end

    @testset "Flip-symmetric extrapolation" begin
        x, w = gl4_nodes_weights(Float64)
        basis = polynomial_basis(1)
        tests = Function[x -> x^2, x -> x^3]
        test_derivs = Function[x -> 2.0 * x, x -> 3.0 * x^2]

        op = optimize_fsbp_operator(x, w, -1.0, 1.0, basis, basis;
                                    test_functions = tests,
                                    test_derivatives = test_derivs,
                                    extrapolation_symmetry = :flip,
                                    sbp_check_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        vL = eval_basis_vector(op.op_basis, -1.0)
        vR = eval_basis_vector(op.op_basis, 1.0)

        @test op.tR ≈ reverse(op.tL) atol = 1e-12 rtol = 1e-12
        @test norm(V' * op.tL - vL) < 1e-10
        @test norm(V' * op.tR - vR) < 1e-10

        endpoint_op = optimize_fsbp_operator([-1.0, 0.0, 1.0],
                                             [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0],
                                             -1.0, 1.0, basis, basis;
                                             extrapolation_symmetry = :flip,
                                             sbp_check_action = :error)
        @test endpoint_op.tL ≈ [1.0, 0.0, 0.0] atol = 1e-12 rtol = 1e-12
        @test endpoint_op.tR ≈ [0.0, 0.0, 1.0] atol = 1e-12 rtol = 1e-12

        @test_throws ErrorException optimize_fsbp_operator([-0.9, 0.0, 1.0],
                                                           [1.0, 1.0, 1.0],
                                                           -1.0, 1.0, basis, basis;
                                                           extrapolation_symmetry = :flip,
                                                           sbp_check_action = :error)
    end

    @testset "BigFloat optimized endpoints" begin
        a, b = BigFloat(-1), BigFloat(1)
        basis = polynomial_basis(1; interval = (a, b))

        op = optimize_fsbp_operator(BigFloat[-1, 1], BigFloat[1, 1],
                                    a, b, basis, basis;
                                    sbp_check_action = :error)

        V = eval_basis_matrix(op.op_basis, op.x)
        Vx = eval_basis_derivative_matrix(op.op_basis, op.x)

        @test op isa FSBPOperator{BigFloat}
        @test op.tL == BigFloat[1, 0]
        @test op.tR == BigFloat[0, 1]
        @test norm(op.D * V - Vx) < BigFloat("1e-30")
        @test norm(op.Q + op.Q' - op.E) < BigFloat("1e-30")
    end

    @testset "Public input validation" begin
        basis = polynomial_basis(1)
        big_basis = polynomial_basis(1; interval = (BigFloat(-1), BigFloat(1)))

        @test_throws ArgumentError optimize_fsbp_operator(
            [-1.0, 1.0], [1.0, 1.0], BigFloat(-1), BigFloat(1),
            big_basis, big_basis)
        @test_throws ArgumentError optimize_fsbp_operator(
            BigFloat[-1, 1], BigFloat[1, 1], -1.0, 1.0,
            basis, basis)
    end
end
