using Test
using GaussFSBP
using LinearAlgebra

@testset "FSBP Operator Construction" begin

    # ═════════════════════════════════════════════════════════════════════
    # Test 1: Polynomial basis — degree 3 monomials on [-1, 1]
    #   F = {1, x, x², x³}  (nb = 4)
    #   G = {1, x, x², x³, x⁴, x⁵}  (2p = 6 functions, even)
    #   With principal=:upper → GLL-type rule → 4 nodes (nn == nb)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial degree 3 (GLL, nn == nb)" begin
        p = 3

        # Approximation basis F = {1, x, x², x³}
        funcs  = [let k=k; x -> x^k end for k in 0:p]
        derivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:p]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        # Quadrature basis: degrees 0 to 2p-1
        qfuncs  = [let k=k; x -> x^k end for k in 0:(2p - 1)]
        qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:(2p - 1)]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        # Use principal=:upper to get GLL (n+1 = 4 nodes for 2n=6 basis)
        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp.nn == fsbp.nb == 4
        @test all(fsbp.w .> 0)

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
        println("\nPolynomial degree 3 (GLL) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["Extrapolation exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 2: Polynomial degree 3, GL rule with enlarged quad basis
    #   F = {1, x, x², x³}  (nb = 4)
    #   G = {1, x, ..., x⁷}  (2(p+1) = 8 functions, even)
    #   With principal=:lower → GL-type rule → 4 interior nodes (nn == nb)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial degree 3 (GL, enlarged quad basis, nn == nb)" begin
        p = 3

        funcs  = [let k=k; x -> x^k end for k in 0:p]
        derivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:p]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        # Enlarged quadrature basis: degrees 0 to 2(p+1)-1 = 2p+1
        qfuncs  = [let k=k; x -> x^k end for k in 0:(2p + 1)]
        qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:(2p + 1)]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:lower)

        @test fsbp.nn == fsbp.nb == 4
        @test all(fsbp.w .> 0)

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
        println("\nPolynomial degree 3 (GL, enlarged) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 3: Polynomial degree 1 — simplest non-trivial case
    #   F = {1, x}  (nb = 2)
    #   G = {1, x}  (2 functions, even)
    #   principal=:upper → GLL → 2 nodes  (nn == nb)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial degree 1 (GLL, nn == nb)" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp.nn == 2
        @test all(fsbp.w .> 0)

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
        println("\nPolynomial degree 1 (GLL) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 4: nn > nb — least-squares path
    #   F = {1, x}         (nb = 2)
    #   G = {1, x, x², x³}  (4 functions, even)
    #   principal=:upper → GLL → 3 nodes (nn=3 > nb=2)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial (nn > nb, least-squares)" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        qfuncs  = [let k=k; x -> x^k end for k in 0:3]
        qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:3]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:upper)

        @test fsbp.nn > fsbp.nb
        @test all(fsbp.w .> 0)
        @test fsbp.tL == [1.0, 0.0, 0.0]
        @test fsbp.tR == [0.0, 0.0, 1.0]

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
        println("\nPolynomial (nn > nb, least-squares) report:")
        println(report)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["Positive weights"].passed
        @test report.checks["Weight sum"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 5: use_optimization=true should error (not yet implemented)
    # ═════════════════════════════════════════════════════════════════════

    @testset "Optimization hook raises error" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        @test_throws ErrorException build_fsbp_operator(op_basis, quad_basis;
                                                         use_optimization=true)
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 6: Basis without derivatives should error
    # ═════════════════════════════════════════════════════════════════════

    @testset "Missing derivatives raises error" begin
        funcs  = [x -> 1.0, x -> x]
        op_basis = FunctionBasis(funcs)  # no derivs

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        @test_throws Exception build_fsbp_operator(op_basis, quad_basis)
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 7: verbose keyword is forwarded to GeneralizedGauss construction
    # ═════════════════════════════════════════════════════════════════════

    @testset "Verbose construction keyword" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        fsbp = redirect_stdout(devnull) do
            build_fsbp_operator(op_basis, quad_basis;
                                principal=:upper, verbose=true)
        end

        @test fsbp.nn == 2
        @test all(fsbp.w .> 0)
    end

end
