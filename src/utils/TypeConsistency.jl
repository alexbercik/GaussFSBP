"""
    TypeConsistency.jl

Strict element-type checks: no silent promotion between `Float64`, `BigFloat`, etc.
"""

function _require_uniform_type(context::AbstractString, types)
    isempty(types) && throw(ArgumentError("$context: no types to check."))
    T = types[1]
    for Ti in types[2:end]
        Ti == T || throw(ArgumentError(
            "$context: all inputs must use the same element type (found $T and $Ti)."))
    end
    T <: Number || throw(ArgumentError("$context: element type $T is not numeric."))
    return T
end

function _array_element_type(vec, name::AbstractString)
    isempty(vec) && throw(ArgumentError("$name must be nonempty."))
    T = typeof(vec[1])
    for i in eachindex(vec)
        typeof(vec[i]) == T || throw(ArgumentError(
            "$name: all entries must have the same element type (found $T and $(typeof(vec[i])))."))
    end
    return T
end

function _interval_endpoint_type(interval, name::AbstractString)
    a, b = interval[1], interval[2]
    Ta, Tb = typeof(a), typeof(b)
    Ta == Tb || throw(ArgumentError(
        "$name: endpoints must have the same element type (found $Ta and $Tb)."))
    Ta <: Number || throw(ArgumentError("$name: element type $Ta is not numeric."))
    return a, b, Ta
end

function _require_eltype(val, ::Type{T}, context::AbstractString) where T
    typeof(val) == T || throw(ArgumentError(
        "$context: expected element type $T, got $(typeof(val))."))
    return val
end

function _require_nodes_match_eltype(xnodes, ::Type{T}, name::AbstractString) where T
    for x in xnodes
        typeof(x) == T || throw(ArgumentError(
            "$name: all nodes must have element type $T (found $(typeof(x)))."))
    end
    return nothing
end
