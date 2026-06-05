"""
    FSBPOperatorPythonExport.jl

Print copy-pasteable Python/NumPy snippets for an `FSBPOperator`'s numeric data.
"""

"""
    _format_py_float(x, num_digits) -> String

Format a real number for Python float literals: round to `num_digits` decimal
places, strip trailing zeros, and ensure a `.0` suffix for integer values.
"""
function _format_py_float(x::Real, num_digits::Int)
    num_digits >= 0 || throw(ArgumentError("num_digits must be >= 0, got $(num_digits)."))
    r = round(x; digits=num_digits)
    v = Float64(r)
    v = v == 0.0 ? 0.0 : v  # avoid signed-zero literals like -0.0
    fmt = Printf.Format("%." * string(num_digits) * "f")
    s = Printf.format(fmt, v)
    if occursin('.', s)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
    end
    if !occursin('.', s)
        s = string(s, ".0")
    end
    return s
end

function _format_py_vector(v::AbstractVector, num_digits::Int)
    parts = [_format_py_float(x, num_digits) for x in v]
    return "np.array([" * join(parts, ", ") * "])"
end

function _format_py_matrix(A::AbstractMatrix, num_digits::Int)
    row_indent = "                "
    rows = String[]
    for i in axes(A, 1)
        row_parts = [_format_py_float(A[i, j], num_digits) for j in axes(A, 2)]
        push!(rows, row_indent * "[" * join(row_parts, ", ") * "],")
    end
    body = join(rows, "\n")
    return "np.array([\n$body\n            ])"
end

"""
    print_fsbp_operator_python(io::IO, op::FSBPOperator; num_digits::Int=5)

Print Python/NumPy assignments for `nodes`, `D`, `H`, `tL`, and `tR` suitable
for copy-paste into a Python script.

# Arguments
- `num_digits` — number of decimal places to round to; trailing zeros are stripped
  (e.g. `num_digits=5` gives `0.33333` and `2.0`).
"""
function print_fsbp_operator_python(io::IO, op::FSBPOperator; num_digits::Int=5)
    println(io, "nodes=", _format_py_vector(op.x, num_digits), ",")
    println(io, "D=", _format_py_matrix(op.D, num_digits), ",")
    println(io, "H=", _format_py_vector(op.w, num_digits), ",")
    println(io, "tL=", _format_py_vector(op.tL, num_digits), ",")
    println(io, "tR=", _format_py_vector(op.tR, num_digits), ",")
    return nothing
end

print_fsbp_operator_python(op::FSBPOperator; kwargs...) =
    print_fsbp_operator_python(stdout, op; kwargs...)
