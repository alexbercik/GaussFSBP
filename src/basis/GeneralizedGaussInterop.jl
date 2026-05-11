"""
    _to_gg_basis(basis::FunctionBasis; require_derivs=false)

Convert a `FunctionBasis` to the `GeneralizedGauss.quadbasis` representation.

When `require_derivs=true`, the conversion requires derivative data and throws
an informative error if `basis.derivs === nothing`.
"""
function _to_gg_basis(basis::FunctionBasis; require_derivs::Bool = false)
    if require_derivs && basis.derivs === nothing
        error("The basis must have derivatives supplied to convert to a " *
              "GeneralizedGauss basis. Pass `derivs` to the FunctionBasis " *
              "constructor.")
    end

    a, b = basis.interval
    return GeneralizedGauss.quadbasis(basis.funcs, basis.derivs, a, b)
end

Base.length(basis::FunctionBasis) = nbasis(basis)

# Extend the public GeneralizedGauss API so callers can pass FunctionBasis
# directly instead of manually constructing a quadbasis wrapper first.
GeneralizedGauss.compute_moments(basis::FunctionBasis; measure=nothing) =
    GeneralizedGauss.compute_moments(_to_gg_basis(basis); measure=measure)

GeneralizedGauss.compute_gauss_rule(basis::FunctionBasis, moments=nothing; kwargs...) =
    GeneralizedGauss.compute_gauss_rule(_to_gg_basis(basis), moments; kwargs...)

GeneralizedGauss.compute_gauss_rules(basis::FunctionBasis, moments=nothing; kwargs...) =
    GeneralizedGauss.compute_gauss_rules(_to_gg_basis(basis), moments; kwargs...)

GeneralizedGauss.orthogonalize_basis(basis::FunctionBasis; measure=nothing, quad_order=nothing) =
    GeneralizedGauss.orthogonalize_basis(_to_gg_basis(basis); measure=measure, quad_order=quad_order)

GeneralizedGauss.check_ECT_system(basis::FunctionBasis; n_points::Int = 200, verbose::Bool = true) =
    GeneralizedGauss.check_ECT_system(_to_gg_basis(basis); n_points=n_points, verbose=verbose)

function GeneralizedGauss.check_T_system(basis::FunctionBasis;
                                         num_tuples::Int = 5000,
                                         tuple_size::Int = length(basis),
                                         verbose::Bool = true,
                                         rng = GeneralizedGauss.Random.default_rng(),
                                         near_zero_rel_tol = nothing)
    GeneralizedGauss.check_T_system(_to_gg_basis(basis);
                                    num_tuples=num_tuples,
                                    tuple_size=tuple_size,
                                    verbose=verbose,
                                    rng=rng,
                                    near_zero_rel_tol=near_zero_rel_tol)
end
