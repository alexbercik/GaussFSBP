using Test

@testset "GaussFSBP" begin
    include("test_helpers.jl")
    include("test_basis.jl")
    include("test_quadrature_verification.jl")
    include("test_known_polynomial_operators.jl")
    include("test_fsbp_operator.jl")
    include("test_optimized_fsbp_operator.jl")
    include("test_bigfloat.jl")
end
