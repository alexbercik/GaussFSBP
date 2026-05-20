"""
    BasisEvaluation.jl

Utility routines for evaluating all basis functions (and their derivatives)
at scalar points or vectors of points.

Matrix convention
-----------------
    V[i, j]  = basis function j evaluated at node xnodes[i]
    Vx[i, j] = derivative of basis function j evaluated at node xnodes[i]

So each *column* of `V` (or `Vx`) corresponds to one basis function sampled
at all nodes.
"""

"""
    _sample_vector(eval, K, x, ::Type{T}) -> Vector{T}

Sample a length-`K` basis vector from callable `eval` at scalar `x`.
"""
function _sample_vector(eval, K::Int, x, ::Type{T}; context::AbstractString = "Basis evaluator") where T
    vals = eval(x)
    length(vals) == K || throw(ArgumentError(
        "$context: returned $(length(vals)) values, expected $K."))
    out = Vector{T}(undef, K)
    for j in 1:K
        out[j] = vals[j]
    end
    return out
end

"""
    _sample_matrix(eval, K, xnodes, ::Type{T}) -> Matrix{T}

Sample basis values at all nodes into an `length(xnodes) × K` matrix.
"""
function _sample_matrix(eval, K::Int, xnodes, ::Type{T}; context::AbstractString = "Basis evaluator") where T
    M = Matrix{T}(undef, length(xnodes), K)
    for i in eachindex(xnodes)
        vals = eval(xnodes[i])
        length(vals) == K || throw(ArgumentError(
            "$context: returned $(length(vals)) values at node $i, expected $K."))
        for j in 1:K
            M[i, j] = vals[j]
        end
    end
    return M
end

"""
    eval_basis_vector(basis::AbstractBasis, x) -> Vector

Evaluate all basis functions in `basis` at scalar `x`.

Returns a vector `v` of length `nbasis(basis)` where `v[j]` is basis function
`j` evaluated at `x`.
"""
function eval_basis_vector(basis::FunctionBasis, x)
    T = eltype(basis)
    typeof(x) == T || throw(ArgumentError(
        "eval_basis_vector: point type ($(typeof(x))) must match " *
        "basis interval type ($T)."))
    return _sample_vector(z -> eval_basis(basis, z), nbasis(basis), x, T;
                          context = "eval_basis_vector")
end

function eval_basis_vector(basis::AbstractBasis, x)
    T = typeof(x)
    return _sample_vector(z -> eval_basis(basis, z), nbasis(basis), x, T;
                          context = "eval_basis_vector")
end

"""
    eval_basis_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate all basis functions in `basis` at each node in `xnodes`.

Returns a matrix `V` of size `(length(xnodes), nbasis(basis))` where

    V[i, j] = basis function j evaluated at xnodes[i].

For `FunctionBasis`, nodes must match `eltype(basis)`; the matrix element
type is `eltype(basis)`.
"""
function eval_basis_matrix(basis::FunctionBasis, xnodes)
    T = eltype(basis)
    _require_nodes_match_eltype(xnodes, T, "eval_basis_matrix nodes")
    return _sample_matrix(z -> eval_basis(basis, z), nbasis(basis), xnodes, T;
                          context = "eval_basis_matrix")
end

function eval_basis_matrix(basis::AbstractBasis, xnodes)
    T = _array_element_type(xnodes, "eval_basis_matrix nodes")
    return _sample_matrix(z -> eval_basis(basis, z), nbasis(basis), xnodes, T;
                          context = "eval_basis_matrix")
end

"""
    eval_basis_derivative_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate the derivatives of all basis functions in `basis` at each node in
`xnodes`.

Returns a matrix `Vx` of size `(length(xnodes), nbasis(basis))` where

    Vx[i, j] = derivative of basis function j evaluated at xnodes[i].
"""
function eval_basis_derivative_matrix(basis::FunctionBasis, xnodes)
    T = eltype(basis)
    _require_nodes_match_eltype(xnodes, T, "eval_basis_derivative_matrix nodes")
    return _sample_matrix(z -> eval_basis_derivative(basis, z), nbasis(basis), xnodes, T;
                          context = "eval_basis_derivative_matrix")
end

function eval_basis_derivative_matrix(basis::AbstractBasis, xnodes)
    T = _array_element_type(xnodes, "eval_basis_derivative_matrix nodes")
    return _sample_matrix(z -> eval_basis_derivative(basis, z), nbasis(basis), xnodes, T;
                          context = "eval_basis_derivative_matrix")
end
