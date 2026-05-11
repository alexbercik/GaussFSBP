using Test
using GaussFSBP

# ─────────────────────────────────────────────────────────────────────────────
# Placeholder tests for well-known polynomial SBP operators.
#
# These tests will eventually recreate and verify the following operators:
#   - Legendre-Gauss-Lobatto (LGL) SBP operators
#   - Legendre-Gauss (LG) SBP operators
#   - Other polynomial diagonal-norm SBP operators
#
# For each operator, the standard SBP verification checks are:
#   1. Accuracy: D * V = Vx  (D differentiates the basis functions)
#   2. SBP property: Q + Q' = B  where Q = H * D, H is the norm matrix,
#      and B = diag(-1, 0, ..., 0, 1) encodes boundary terms.
#   3. Interpolation/extrapolation to boundary nodes is exact.
# ─────────────────────────────────────────────────────────────────────────────

@testset "Polynomial SBP placeholders" begin

    @testset "LGL operators (TODO)" begin
        # TODO: Once OperatorBuilders is implemented, construct the degree-p
        # LGL operator and verify:
        #   1. D * V = Vx
        #   2. Q + Q' = B
        #   3. H * ones(n) = w  (LGL weights)
        #
        # Reference: Gassner & Kopriva (2011), Carpenter & Kennedy (1994).
        @test_broken false  # placeholder — not yet implemented
    end

    @testset "LG operators (TODO)" begin
        # TODO: Once OperatorBuilders is implemented, construct the degree-p
        # Legendre-Gauss (interior-node) operator and verify:
        #   1. D * V = Vx
        #   2. Q + Q' = B  (with extrapolation operators to boundary)
        #   3. H * ones(n) = w  (LG weights)
        #
        # Reference: Fernández et al. (2014), Montoya & Zingg (2020).
        @test_broken false  # placeholder — not yet implemented
    end

    @testset "SBP accuracy check stub (TODO)" begin
        # TODO: Implement a generic helper
        #   check_sbp_accuracy(basis, D, H) -> Bool
        # and test it here against known LGL/LG data.
        @test_broken false  # placeholder — not yet implemented
    end

    @testset "SBP property check stub (TODO)" begin
        # TODO: Implement a generic helper
        #   check_sbp_property(D, H, B) -> Bool
        # verifying Q + Q' = B where Q = H * D.
        @test_broken false  # placeholder — not yet implemented
    end

end
