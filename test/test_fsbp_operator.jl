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
    # Test 4b: nn > nb without endpoint nodes
    #   The rectangular construction must use E = tR tRᵀ - tL tLᵀ, not a
    #   hard-coded endpoint-diagonal boundary matrix.
    # ═════════════════════════════════════════════════════════════════════

    @testset "Polynomial (nn > nb, GL, extrapolation boundary)" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

        qfuncs  = [let k=k; x -> x^k end for k in 0:5]
        qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:5]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                    orthogonalize=true, principal=:lower)

        @test fsbp.nn == 3
        @test fsbp.nn > fsbp.nb
        @test all(abs.(fsbp.x .+ 1.0) .> 1e-8)
        @test all(abs.(fsbp.x .- 1.0) .> 1e-8)
        @test fsbp.E ≈ fsbp.tR * fsbp.tR' - fsbp.tL * fsbp.tL'

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)

        @test report.checks["Derivative exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["Boundary decomposition"].passed
        @test report.checks["Extrapolation exactness"].passed
        @test report.checks["Skew-symmetry"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 5: use_optimization=true builds an optimized operator
    # ═════════════════════════════════════════════════════════════════════

    @testset "Optimization hook builds operator" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                   principal=:upper,
                                   use_optimization=true,
                                   sbp_check_action=:error,
                                   test_functions=[x -> exp(x)],
                                   test_derivatives=[x -> exp(x)],
                                   test_weights=[2.0],
                                   extrapolation_objective_weights=(accuracy=1//2, norm=1//2),
                                   S_objective_weights=(accuracy=1//2, norm=1//2),
                                   derivative_error_norm=:H,
                                   opt_method=:sequential,
                                   extrapolation_symmetry=:none)

        @test fsbp.nn == 2
        @test fsbp.tL == [1.0, 0.0]
        @test fsbp.tR == [0.0, 1.0]

        report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
        @test report.checks["Derivative exactness"].passed
        @test report.checks["SBP property"].passed
        @test report.checks["SBP compatibility"].passed
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 6: Basis without derivatives should error
    # ═════════════════════════════════════════════════════════════════════

    @testset "Mismatched basis interval types error" begin
        op_funcs  = [x -> one(x), x -> x]
        op_derivs = [x -> zero(x), x -> one(x)]
        op_basis = FunctionBasis(op_funcs; derivs=op_derivs,
                               interval=(BigFloat(-1), BigFloat(1)))

        qfuncs  = [let k=k; x -> x^k end for k in 0:3]
        qderivs = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:3]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                     principal=:upper)
    end

    @testset "Missing derivatives raises error" begin
        funcs  = [x -> 1.0, x -> x]
        op_basis = FunctionBasis(funcs)  # no derivs

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        @test_throws Exception build_fsbp_operator(op_basis, quad_basis)
    end


    # ═════════════════════════════════════════════════════════════════════
    # Test 7: verbose keyword controls FSBP output; quad_kwargs can override
    # ═════════════════════════════════════════════════════════════════════

    @testset "Verbose construction keyword" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        verbose_output = mktemp() do _, io
            redirect_stdout(io) do
                build_fsbp_operator(op_basis, quad_basis;
                                    principal=:upper, verbose=true)
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end

        @test occursin("Direct FSBP construction", verbose_output)
        @test occursin("Computing two-point Lobatto rule", verbose_output)

        quiet_quad_output = mktemp() do _, io
            redirect_stdout(io) do
                build_fsbp_operator(op_basis, quad_basis;
                                    principal=:upper, verbose=true,
                                    quad_kwargs=(verbose=false,))
            end
            flush(io)
            seekstart(io)
            read(io, String)
        end

        @test occursin("Direct FSBP construction", quiet_quad_output)
        @test !occursin("Computing two-point Lobatto rule", quiet_quad_output)
    end

    @testset "Quadrature kwargs are forwarded" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [x -> 1.0, x -> x]
        qderivs = [x -> 0.0, x -> 1.0]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)

        # These keywords are consumed by GeneralizedGauss.compute_gauss_rule,
        # not by the FSBP builder itself.
        fsbp = build_fsbp_operator(op_basis, quad_basis;
                                   principal=:upper,
                                   quad_kwargs=(lost_digits=3,
                                                add_endpoint=:right,
                                                solver_tolerance=1e-12,
                                                intermediate_tolerance=1e-9))

        @test fsbp.nn == 2
        @test all(fsbp.w .> 0)

        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal=:upper,
                                                       quad_kwargs=(solver_tolerance=-1.0,))
        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       quad_kwargs=(principal=:upper,))
        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       add_endpoint=:right,
                                                       quad_kwargs=(add_endpoint=:left,))
    end

    @testset "Explicit quadrature moments" begin
        funcs  = [x -> 1.0, x -> x]
        derivs = [x -> 0.0, x -> 1.0]
        op_basis = FunctionBasis(funcs; derivs=derivs)

        qfuncs  = [let k=k; x -> x^k end for k in 0:3]
        qderivs = [let k=k; k == 0 ? (x -> 0.0) : (x -> k * x^(k-1)) end for k in 0:3]
        quad_basis = FunctionBasis(qfuncs; derivs=qderivs)
        quad_moments = [2.0, 0.0, 2.0 / 3.0, 0.0]

        # The supplied moments are for the original quadrature basis.  The
        # builder must transform them internally when orthogonalization is on.
        fsbp_orth = build_fsbp_operator(op_basis, quad_basis;
                                        orthogonalize=true,
                                        principal=:lower,
                                        quad_moments=quad_moments)
        fsbp_raw = build_fsbp_operator(op_basis, quad_basis;
                                       orthogonalize=false,
                                       principal=:lower,
                                       quad_moments=quad_moments)

        @test fsbp_orth.nn == fsbp_raw.nn == 2
        @test fsbp_orth.x ≈ fsbp_raw.x
        @test fsbp_orth.w ≈ fsbp_raw.w

        report = check_fsbp_operator(fsbp_orth; atol=1e-10, rtol=1e-10,
                                     quad_moments=quad_moments)
        @test report.checks["Quadrature exactness"].passed
        @test report.checks["SBP property"].passed
        @test_throws ArgumentError check_fsbp_operator(fsbp_orth;
                                                       quad_moments=[2.0])

        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal=:lower,
                                                       quad_moments=[2.0, 0.0])
        @test_throws ArgumentError build_fsbp_operator(op_basis, quad_basis;
                                                       principal=:lower,
                                                       quad_kwargs=(moments=quad_moments,))
    end

end
