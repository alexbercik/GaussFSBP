using Test
using GaussFSBP
using LinearAlgebra

@testset "BigFloat Precision" begin

    # ═════════════════════════════════════════════════════════════════════
    # Test 1: FunctionBasis with BigFloat interval
    # ═════════════════════════════════════════════════════════════════════

    @testset "FunctionBasis accepts BigFloat interval" begin
        a, b = BigFloat(-1), BigFloat(1)
        funcs  = [x -> 1.0, x -> x, x -> x^2]
        derivs = [x -> 0.0, x -> 1.0, x -> 2x]
        basis  = FunctionBasis(funcs; derivs=derivs, interval=(a, b))

        @test eltype(basis) == BigFloat
        @test basis.interval == (a, b)
        @test nbasis(basis) == 3
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 2: BigFloat FSBP operator — polynomial degree 3 (GLL)
    #   Same as the Float64 test but with BigFloat endpoints.
    #   Errors should be much smaller than Float64 eps.
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial degree 3 (GLL, BigFloat)" begin
        p = 3
        a, b = BigFloat(-1), BigFloat(1)

        funcs  = [let k=k; x -> x^k end for k in 0:p]
        derivs = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:p]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(a, b))

        qfuncs  = [let k=k; x -> x^k end for k in 0:(2p - 1)]
        qderivs = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:(2p - 1)]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(a, b))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        # Check that the operator is BigFloat-precision
        @test eltype(fsbp.x) == BigFloat
        @test eltype(fsbp.w) == BigFloat
        @test eltype(fsbp.D) == BigFloat
        @test fsbp isa FSBPOperator{BigFloat}

        @test fsbp.nn == fsbp.nb == 4
        @test all(fsbp.w .> 0)

        # Run verification — should pass with default (tight) tolerances
        report = check_fsbp_operator(fsbp)
        println("\nBigFloat polynomial degree 3 (GLL) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["Extrapolation exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed

        # Verify errors are much smaller than Float64 precision
        @test report.checks["Derivative exactness"].error < 1e-30
        @test report.checks["Weight sum"].error < 1e-30
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 3: BigFloat FSBP operator — polynomial degree 1 (GLL)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial degree 1 (GLL, BigFloat)" begin
        a, b = BigFloat(-1), BigFloat(1)

        funcs  = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(a, b))

        qfuncs  = [x -> one(x), x -> x]
        qderivs = [x -> zero(x), x -> one(x)]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(a, b))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp isa FSBPOperator{BigFloat}
        @test fsbp.nn == 2
        @test all(fsbp.w .> 0)

        report = check_fsbp_operator(fsbp)
        println("\nBigFloat polynomial degree 1 (GLL) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 4: BigFloat nn > nb (least-squares path)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial (nn > nb, least-squares, BigFloat)" begin
        a, b = BigFloat(-1), BigFloat(1)

        funcs  = [x -> one(x), x -> x]
        derivs = [x -> zero(x), x -> one(x)]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(a, b))

        qfuncs  = [let k=k; x -> x^k end for k in 0:3]
        qderivs = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:3]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(a, b))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp isa FSBPOperator{BigFloat}
        @test fsbp.nn > fsbp.nb
        @test all(fsbp.w .> 0)
        @test fsbp.tL == BigFloat[1, 0, 0]
        @test fsbp.tR == BigFloat[0, 0, 1]

        report = check_fsbp_operator(fsbp)
        println("\nBigFloat polynomial (nn > nb, least-squares) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 5: Mixed — Float64 basis functions still produce Float64 operator
    # ═════════════════════════════════════════════════════════════════════

    @testset "Float64 interval stays Float64" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp isa FSBPOperator{Float64}
        @test eltype(fsbp.x) == Float64
        @test eltype(fsbp.w) == Float64
        @test all(fsbp.w .> 0)

        report = check_fsbp_operator(fsbp)
        @test report.checks["Derivative exactness"].passed
        @test report.checks["Positive weights"].passed
    end

end
