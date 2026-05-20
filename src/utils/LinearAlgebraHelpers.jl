"""
    LinearAlgebraHelpers.jl

SVD-based rank, nullspace, and rank-truncated least-squares helpers shared by
exact and optimization-based FSBP builders.
"""

function _svd_factor(A; full::Bool = false)
    return svd(A; full = full)
end

function _rank_from_singular_values(s, dims, ::Type{T}, rank_tol) where T
    isempty(s) && return 0
    sigma_max = maximum(s)
    sigma_max == zero(T) && return 0
    tol = rank_tol === nothing ? max(dims...) * sqrt(eps(T)) * sigma_max : T(rank_tol)
    return count(sigma -> sigma > tol, s)
end

function _matrix_rank(A; rank_tol = nothing)
    T = eltype(A)
    min(size(A)...) == 0 && return 0
    F = _svd_factor(A; full = false)
    return _rank_from_singular_values(F.S, size(A), T, rank_tol)
end

"""
    _check_vandermonde_rank(V, K, T; rank_tol, context)

Require full column rank `K` of the exactness Vandermonde matrix `V` (`size(V,2) == K`).
"""
function _check_vandermonde_rank(V, K::Int, ::Type{T};
                                         rank_tol = nothing,
                                         context::AbstractString = "FSBP construction") where T
    _rank_tol = rank_tol === nothing ? nothing : T(rank_tol)
    rankV = _matrix_rank(V; rank_tol = _rank_tol)
    rankV == K || throw(ArgumentError(
        "$context: the sampled exactness matrix V has rank $rankV, expected $K."))
    return rankV
end

function _pseudoinverse_solve(A, b; rank_tol = nothing)
    T = promote_type(eltype(A), eltype(b))
    m, n = size(A)
    if n == 0
        return zeros(T, 0)
    elseif m == 0
        return zeros(T, n)
    end
    F = _svd_factor(Matrix{T}(A); full = false)
    r = _rank_from_singular_values(F.S, size(A), T, rank_tol)
    r == 0 && return zeros(T, n)
    U = F.U[:, 1:r]
    V = F.V[:, 1:r]
    sigma = F.S[1:r]
    rhs = Vector{T}(b)
    return V * ((U' * rhs) ./ sigma)
end

function _least_squares_solve(A, b; rank_tol = nothing, prefer_direct::Bool = false)
    if prefer_direct
        T = promote_type(eltype(A), eltype(b))
        try
            lhs = eltype(A) === T ? A : Matrix{T}(A)
            rhs = eltype(b) === T ? b : Vector{T}(b)
            x = lhs \ rhs
            all(isfinite, x) && return x
        catch err
            err isa InterruptException && rethrow()
        end
    end
    return _pseudoinverse_solve(A, b; rank_tol = rank_tol)
end

function _nullspace_basis(A; rank_tol = nothing)
    T = eltype(A)
    _, n = size(A)
    n == 0 && return zeros(T, 0, 0)
    size(A, 1) == 0 && return Matrix{T}(I, n, n)
    F = _svd_factor(A; full = true)
    r = _rank_from_singular_values(F.S, size(A), T, rank_tol)
    V = F.V
    r == size(V, 2) && return zeros(T, n, 0)
    return Matrix{T}(V[:, (r + 1):end])
end
