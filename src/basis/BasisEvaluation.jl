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
    eval_basis_matrix(basis::AbstractBasis, xnodes) -> Matrix{Float64}

Evaluate all basis functions in `basis` at each node in `xnodes`.

Returns a matrix `V` of size `(length(xnodes), nbasis(basis))` where

    V[i, j] = basis function j evaluated at xnodes[i].
"""
function eval_basis_matrix(basis::AbstractBasis, xnodes)
    n  = length(xnodes)
    nb = nbasis(basis)
    V  = Matrix{Float64}(undef, n, nb)
    for (i, xi) in enumerate(xnodes)
        vals = eval_basis(basis, xi)
        for j in 1:nb
            V[i, j] = vals[j]
        end
    end
    return V
end

"""
    eval_basis_derivative_matrix(basis::AbstractBasis, xnodes) -> Matrix{Float64}

Evaluate the derivatives of all basis functions in `basis` at each node in
`xnodes`.

Returns a matrix `Vx` of size `(length(xnodes), nbasis(basis))` where

    Vx[i, j] = derivative of basis function j evaluated at xnodes[i].
"""
function eval_basis_derivative_matrix(basis::AbstractBasis, xnodes)
    n  = length(xnodes)
    nb = nbasis(basis)
    Vx = Matrix{Float64}(undef, n, nb)
    for (i, xi) in enumerate(xnodes)
        dvals = eval_basis_derivative(basis, xi)
        for j in 1:nb
            Vx[i, j] = dvals[j]
        end
    end
    return Vx
end
