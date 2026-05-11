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
    eval_basis_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate all basis functions in `basis` at each node in `xnodes`.

Returns a matrix `V` of size `(length(xnodes), nbasis(basis))` where

    V[i, j] = basis function j evaluated at xnodes[i].

The element type of the matrix is inferred from the evaluation results.
"""
function eval_basis_matrix(basis::AbstractBasis, xnodes)
    n  = length(xnodes)
    nb = nbasis(basis)
    # Evaluate at the first node to determine the element type
    vals1 = eval_basis(basis, xnodes[1])
    T = eltype(vals1)
    V = Matrix{T}(undef, n, nb)
    for j in 1:nb
        V[1, j] = vals1[j]
    end
    for i in 2:n
        vals = eval_basis(basis, xnodes[i])
        for j in 1:nb
            V[i, j] = vals[j]
        end
    end
    return V
end

"""
    eval_basis_derivative_matrix(basis::AbstractBasis, xnodes) -> Matrix

Evaluate the derivatives of all basis functions in `basis` at each node in
`xnodes`.

Returns a matrix `Vx` of size `(length(xnodes), nbasis(basis))` where

    Vx[i, j] = derivative of basis function j evaluated at xnodes[i].

The element type of the matrix is inferred from the evaluation results.
"""
function eval_basis_derivative_matrix(basis::AbstractBasis, xnodes)
    n  = length(xnodes)
    nb = nbasis(basis)
    # Evaluate at the first node to determine the element type
    dvals1 = eval_basis_derivative(basis, xnodes[1])
    T = eltype(dvals1)
    Vx = Matrix{T}(undef, n, nb)
    for j in 1:nb
        Vx[1, j] = dvals1[j]
    end
    for i in 2:n
        dvals = eval_basis_derivative(basis, xnodes[i])
        for j in 1:nb
            Vx[i, j] = dvals[j]
        end
    end
    return Vx
end
