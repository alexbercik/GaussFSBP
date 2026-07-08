using Test
using LinearAlgebra
using GaussFSBP

# Shared helpers for publication tests.  The polynomial fixture helpers use
# standard interpolation formulas, not GaussFSBP internals, so they provide an
# independent reference for known GL/GLL operators.

function polynomial_basis(max_degree; interval = (-1.0, 1.0))
    funcs = Function[
        let k = k
            x -> x^k
        end for k in 0:max_degree
    ]
    derivs = Function[
        let k = k
            if k == 0
                x -> zero(x)
            else
                x -> k * x^(k - 1)
            end
        end for k in 0:max_degree
    ]
    return FunctionBasis(funcs; derivs = derivs, interval = interval)
end

function polynomial_basis_from_degrees(degrees; interval = (-1.0, 1.0))
    funcs = Function[
        let k = k
            x -> x^k
        end for k in degrees
    ]
    derivs = Function[
        let k = k
            if k == 0
                x -> zero(x)
            else
                x -> k * x^(k - 1)
            end
        end for k in degrees
    ]
    return FunctionBasis(funcs; derivs = derivs, interval = interval)
end

function polynomial_moments(max_degree, ::Type{T} = Float64) where T
    return [iseven(k) ? T(2) / T(k + 1) : zero(T) for k in 0:max_degree]
end

function lagrange_derivative_matrix(x::AbstractVector{T}) where T
    n = length(x)
    bary_weights = Vector{T}(undef, n)

    for j in 1:n
        denom = one(T)
        for k in 1:n
            k == j && continue
            denom *= x[j] - x[k]
        end
        bary_weights[j] = inv(denom)
    end

    D = zeros(T, n, n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            D[i, j] = bary_weights[j] / bary_weights[i] / (x[i] - x[j])
        end
        D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
    end
    return D
end

function lagrange_endpoint_vector(x::AbstractVector{T}, endpoint) where T
    z = T(endpoint)
    n = length(x)
    ell = Vector{T}(undef, n)

    for j in 1:n
        value = one(T)
        for k in 1:n
            k == j && continue
            value *= (z - x[k]) / (x[j] - x[k])
        end
        ell[j] = value
    end
    return ell
end

function gll4_nodes_weights(::Type{T} = Float64) where T
    a = inv(sqrt(T(5)))
    return T[-1, -a, a, 1], T[T(1) / T(6), T(5) / T(6), T(5) / T(6), T(1) / T(6)]
end

function gl4_nodes_weights(::Type{T} = Float64) where T
    a = sqrt(T(3) / T(7) + T(2) * sqrt(T(6) / T(5)) / T(7))
    b = sqrt(T(3) / T(7) - T(2) * sqrt(T(6) / T(5)) / T(7))
    w_outer = (T(18) - sqrt(T(30))) / T(36)
    w_inner = (T(18) + sqrt(T(30))) / T(36)
    return T[-a, -b, b, a], T[w_outer, w_inner, w_inner, w_outer]
end

function test_standard_fsbp_report(report)
    @test report.passed
    for check_name in (
        "Derivative exactness",
        "Quadrature exactness",
        "SBP property",
        "Boundary decomposition",
        "Extrapolation exactness",
        "Positive weights",
        "Weight sum",
        "Skew-symmetry",
        "SBP compatibility",
    )
        @test report.checks[check_name].passed
    end
end
