using Test

@testset "GaussFSBP" begin
    include("test_basis.jl")
    include("test_quadrature_verification.jl")
    include("test_polynomial_sbp_placeholders.jl")
    include("test_fsbp_operator.jl")
    include("test_optimized_fsbp_operator.jl")
    include("test_bigfloat.jl")
end
