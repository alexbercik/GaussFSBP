# GaussFSBP

GaussFSBP builds one-dimensional diagonal-norm function-space
summation-by-parts (FSBP) operators from some callable basis functions.

At a high level, the workflow is:

1. Define an approximation basis `F` with functions and derivatives.
2. Define a quadrature exactness basis `G=(FF)'`, the functions that
   the quadrature rule must integrate exactly for SBP compatibility.
3. Let `GeneralizedGauss.jl` compute quadrature nodes and weights for `G`.
4. Build the FSBP matrices `H`, `D`, `Q`, `S`, `E` and boundary extrapolation
   vectors `tL`, `tR`.
5. Verify the result with `check_fsbp_operator`.

The main entry point is:

```julia
build_fsbp_operator(op_basis, quad_basis; kwargs...)
```

The returned `FSBPOperator` object stores:

```julia
fsbp.D          # differentiation matrix
fsbp.H          # diagonal norm / mass matrix
fsbp.Q          # weak derivative matrix, Q = H * D
fsbp.S          # skew-symmetric part, S = Q - E / 2
fsbp.E          # boundary matrix, E = tR * tR' - tL * tL'
fsbp.tL         # left boundary extrapolation vector
fsbp.tR         # right boundary extrapolation vector
fsbp.x          # quadrature nodes
fsbp.w          # quadrature weights
fsbp.op_basis   # approximation basis used to build D
fsbp.quad_basis # quadrature exactness basis used to compute x and w
fsbp.interval   # reference interval, stored as (a, b)
fsbp.nn         # number of quadrature nodes
fsbp.nb         # number of approximation basis functions
```

## Setup

GaussFSBP depends on the local unregistered package `GeneralizedGauss.jl`,
stored in `lib/GeneralizedGauss.jl/`.

For Julia 1.11 and newer, the `[sources]` section in `Project.toml` resolves
the local package automatically:

```julia
julia --project=.
] instantiate
```

For Julia 1.9 and 1.10, develop the local dependency manually:

```julia
julia --project=.
] dev lib/GeneralizedGauss.jl
```

If `julia` is not on your shell path but Julia is installed at
`/opt/local/bin/julia`, use that full path in the commands below.

## Quick Start

This example builds a classical polynomial FSBP operator on `[-1, 1]`.

```julia
using GaussFSBP

p = 3

# Approximation basis F = span(1, x, x^2, x^3).
funcs = [let k = k
    x -> x^k
end for k in 0:p]

derivs = [let k = k
    k == 0 ? (x -> zero(x)) : (x -> k * x^(k - 1))
end for k in 0:p]

op_basis = FunctionBasis(funcs; derivs=derivs, interval=(-1.0, 1.0))

# Quadrature basis G.  For polynomial F of degree p, derivatives of products
# have degree up to 2p - 1.
qfuncs = [let k = k
    x -> x^k
end for k in 0:(2p - 1)]

qderivs = [let k = k
    k == 0 ? (x -> zero(x)) : (x -> k * x^(k - 1))
end for k in 0:(2p - 1)]

quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=(-1.0, 1.0))

# principal=:upper gives the Gauss-Legendre-Lobatto-type rule for an even
# length quadrature basis.
fsbp = build_fsbp_operator(op_basis, quad_basis;
                           orthogonalize=true,
                           principal=:upper)

println(fsbp)
println(fsbp.x)
println(fsbp.w)

report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
println(report)
```

The callable functions are plain Julia functions.  Use `let k = k` in loops so
each closure captures its own exponent.

## How The Code Works

GaussFSBP separates three related spaces:

- `op_basis`: the approximation basis `F`.  The final derivative matrix should
  differentiate these functions exactly at the computed nodes.
- `quad_basis`: the quadrature exactness basis `G`.  GeneralizedGauss computes
  nodes and weights that integrate these functions exactly.
- optional optimization test functions: extra functions used only by the
  optimization path to choose among non-unique FSBP operators.

`build_fsbp_operator` first converts `quad_basis` to the GeneralizedGauss basis
type.  If `orthogonalize=true`, it orthogonalizes that quadrature basis to make
the nonlinear quadrature solve better conditioned.  It then calls
`compute_gauss_rule` to get weights `w` and nodes `x`.

Using the quadrature, the generalized Vandermonde matrix is built:

```julia
V  = eval_basis_matrix(op_basis, x)
Vx = eval_basis_derivative_matrix(op_basis, x)
```

It then builds boundary extrapolation vectors `tL` and `tR`, forms
`E = tR*tR' - tL*tL'`, and constructs `Q`, `D`, and `S` so that:

```julia
D * V == Vx
Q + Q' == E
D == inv(H) * Q
```

When the number of nodes equals the number of basis functions, the derivative
operator is unique.  When there are more nodes than basis functions, the direct
builder chooses a minimum-norm skew-symmetric part.  The optimization path can
instead choose among the free parameters by minimizing accuracy and norm
objectives.

## Passing Callable Functions

Use `FunctionBasis` for both the approximation basis and the quadrature basis:

```julia
basis = FunctionBasis(funcs; derivs=derivs, interval=(a, b))
```

Rules that matter in practice:

- `funcs` and `derivs` must have the same length when derivatives are supplied.
- The approximation basis must have derivatives, because the operator builder
  needs `Vx`.
- The quadrature basis may omit derivatives if GeneralizedGauss can use finite
  differences or the derivative-free path.  Pass
  `quad_kwargs=(differentiable=false,)` to force the derivative-free MADS path.
- The interval endpoint types determine the working type.  If `a` and `b` are
  `Float64`, the operator will be `Float64`.  If they are `BigFloat`, the
  operator will be `BigFloat`.
- `op_basis` and `quad_basis` must use the same interval type.

Prefer functions that preserve the type of their input:

```julia
x -> one(x)
x -> zero(x)
x -> x^2
x -> exp(x)
```

Avoid hard-coded `1.0` or `0.0` in BigFloat calculations unless you explicitly
want Float64 values to enter the computation.

## Choosing The Quadrature Basis

The quadrature basis tells GeneralizedGauss what the rule must integrate
exactly.  For SBP operators, a common choice is the derivative of products of
approximation functions:

```julia
G = (F * F)'
```

For polynomial `F = P_p`, this means `G` contains polynomials through degree
`2p - 1`.

For an even-length quadrature basis with `2n` functions:

- `principal=:lower` gives a Gauss-Legendre-type rule with `n` interior nodes.
- `principal=:upper` gives a Gauss-Legendre-Lobatto-type rule with `n + 1`
  nodes including both endpoints.

For an odd-length quadrature basis with `2n + 1` functions:

- `principal=:lower` gives a left-Radau-type rule.
- `principal=:upper` gives a right-Radau-type rule.

Use `quad_kwargs=(add_endpoint=:left,)` or
`quad_kwargs=(add_endpoint=:right,)` to control the continuation path used by
GeneralizedGauss.  This is useful when one endpoint is singular: anchor the
continuation at the endpoint where the basis is well-defined.

## Weighted Measures

For weighted quadrature, pass the measure as a callable in `quad_kwargs`:

```julia
mu = x -> exp(x)

fsbp = build_fsbp_operator(op_basis, quad_basis;
                           principal=:lower,
                           quad_kwargs=(measure=mu,))
```

When `orthogonalize=true`, GaussFSBP also uses the same `measure` during
GeneralizedGauss basis orthogonalization.

The quadrature weights returned in `fsbp.w` already include the measure.  In
other words, the rule approximates:

```julia
integral(mu(x) * f(x), x in a..b) ~= sum(fsbp.w[i] * f(fsbp.x[i]) for i in eachindex(fsbp.x))
```

## Working Precision

Precision is controlled by the interval endpoints and by the functions you
write.  For BigFloat work, build the whole problem inside a `setprecision`
block and use BigFloat endpoints:

```julia
using GaussFSBP

setprecision(BigFloat, 80; base=10) do
    a = BigFloat(-1)
    b = BigFloat(1)
    interval = (a, b)

    p = 3
    funcs = [let k = k
        x -> x^k
    end for k in 0:p]

    derivs = [let k = k
        k == 0 ? (x -> zero(x)) : (x -> k * x^(k - 1))
    end for k in 0:p]

    op_basis = FunctionBasis(funcs; derivs=derivs, interval=interval)

    qfuncs = [let k = k
        x -> x^k
    end for k in 0:(2p - 1)]

    qderivs = [let k = k
        k == 0 ? (x -> zero(x)) : (x -> k * x^(k - 1))
    end for k in 0:(2p - 1)]

    quad_basis = FunctionBasis(qfuncs; derivs=qderivs, interval=interval)

    fsbp = build_fsbp_operator(op_basis, quad_basis;
                               principal=:upper,
                               quad_kwargs=(
                                   solver_tolerance=BigFloat(10)^(-50),
                                   intermediate_tolerance=BigFloat(10)^(-20),
                                   lost_digits=5,
                               ))

    println(eltype(fsbp.x))
end
```

Practical BigFloat precision notes:

- Use `BigFloat(-1)`, not `-1.0`, for endpoints.
- Use `one(x)` and `zero(x)` in callables.
- Pass BigFloat tolerances when using BigFloat.
- `solver_tolerance` controls the final nonlinear residual target in
  GeneralizedGauss.
- `intermediate_tolerance` can be looser than `solver_tolerance` to speed up
  continuation solves.  The final rule is still polished to
  `solver_tolerance`.
- `lost_digits` allows GeneralizedGauss to accept a nonlinear solve whose
  residual missed the requested tolerance by a small number of decimal digits.
  This is useful for ill-conditioned bases.

## `build_fsbp_operator` Keyword Reference

```julia
build_fsbp_operator(op_basis, quad_basis;
                    orthogonalize=true,
                    use_optimization=false,
                    principal=:lower,
                    extrapolation_norm=:Hinv,
                    rank_tol=nothing,
                    quad_kwargs=NamedTuple(),
                    verbose=false,
                    opt_kwargs...)
```

### Core Keywords

- `orthogonalize=true`: Orthogonalize the quadrature basis before computing the
  rule.  This often improves nonlinear solve conditioning.  Turn it off when
  you want to debug the raw basis or when orthogonalization is itself
  ill-conditioned.
- `principal=:lower`: Choose the GeneralizedGauss principal representation.
  Use `:lower` for GL-type rules and `:upper` for GLL-type rules when
  `quad_basis` has even length.
- `extrapolation_norm=:Hinv`: Norm used to choose boundary extrapolation
  vectors when they are not unique.  Supported values are `:Hinv`, `:H`, and
  `:Euclidean`.
- `rank_tol=nothing`: Tolerance for rank checks and pseudoinverses.  Leave as
  `nothing` unless a basis is nearly rank deficient.
- `verbose=false`: Print quadrature and construction diagnostics.

### Quadrature Keywords

Pass GeneralizedGauss options through `quad_kwargs`:

```julia
fsbp = build_fsbp_operator(op_basis, quad_basis;
                           principal=:upper,
                           quad_kwargs=(
                               add_endpoint=:right,
                               lost_digits=5,
                               solver_tolerance=1e-12,
                               intermediate_tolerance=1e-8,
                               differentiable=true,
                           ))
```

Useful `quad_kwargs` entries:

- `add_endpoint`: endpoint anchor for the continuation path.  Use `:left` or
  `:right`.
- `measure`: callable weight function for weighted moments.
- `lost_digits`: per-call lost-digit allowance for nonlinear solves.
- `solver_tolerance`: final nonlinear solve tolerance.
- `intermediate_tolerance`: looser tolerance for continuation checkpoints.
- `differentiable`: use analytic or finite-difference Newton-style solves when
  `true`; use derivative-free MADS solves when `false`.
- `max_adaptive_steps`: maximum number of continuation retries.

Do not put `principal` or `verbose` inside `quad_kwargs`; those are top-level
`build_fsbp_operator` keywords.

### Optimization Keywords

Set `use_optimization=true` to delegate the non-unique parts of the FSBP
operator to `optimize_fsbp_operator`:

```julia
extra_tests = [x -> exp(x)]
extra_derivs = [x -> exp(x)]

fsbp = build_fsbp_operator(op_basis, quad_basis;
                           principal=:lower,
                           use_optimization=true,
                           test_functions=extra_tests,
                           test_derivatives=extra_derivs,
                           test_weights=[1.0],
                           opt_method=:simultaneous,
                           simultaneous_num_starts=10)
```

Common optimization keywords:

- `test_functions`: extra callables used to measure extrapolation and
  derivative accuracy.
- `test_derivatives`: derivatives of `test_functions`.
- `test_weights`: relative weights for the test functions.
- `extrapolation_objective_weights`: named tuple with `accuracy` and `norm`
  weights for `tL` and `tR`.
- `S_objective_weights`: named tuple with `accuracy` and `norm` weights for
  the derivative/skew part.
- `derivative_error_norm`: norm for derivative test errors.  Commonly `:H`,
  `:Hinv`, or `:Euclidean`.
- `zero_boundary_scaling`: `:fallback` or `:omit` for tests whose boundary
  scale is zero.
- `extrapolation_symmetry`: `:none` or `:flip`.
- `compatibility_action`: `:warn`, `:error`, or `:ignore` if quadrature/SBP
  compatibility is not within tolerance.
- `opt_method`: `:simultaneous` or `:sequential`.
- `simultaneous_num_starts`: number of local optimization starts for the
  simultaneous method.
- `simultaneous_max_iter`, `simultaneous_step_tol`,
  `simultaneous_grad_tol`, `simultaneous_obj_tol`: local solver controls.

Optimization keywords are accepted only when `use_optimization=true`.

## Direct Quadrature Use

GaussFSBP re-exports the GeneralizedGauss quadrature API.  You can compute a
rule directly from a `FunctionBasis`:

```julia
using GaussFSBP

funcs = [x -> one(x), x -> x, x -> x^2, x -> x^3]
derivs = [x -> zero(x), x -> one(x), x -> 2x, x -> 3x^2]
basis = FunctionBasis(funcs; derivs=derivs)

w, x = compute_gauss_rule(basis; principal=:lower)
```

The return order is `(w, x)`, matching GeneralizedGauss.  The `FSBPOperator`
stores them as fields named `fsbp.w` and `fsbp.x`.

For lower-level GeneralizedGauss use, build a `quadbasis` explicitly:

```julia
gg_basis = quadbasis(funcs, derivs, -1.0, 1.0)
w, x = compute_gauss_rule(gg_basis; principal=:upper)
```

## Verification

Use `check_fsbp_operator` after construction:

```julia
report = check_fsbp_operator(fsbp; atol=1e-10, rtol=1e-10)
println(report)
```

The report checks:

- derivative exactness on `op_basis`
- quadrature exactness on `quad_basis`
- SBP property `Q + Q' == E`
- boundary decomposition `E == tR*tR' - tL*tL'`
- extrapolation exactness at interval endpoints
- positive weights
- weight sum
- nullspace consistency
- skew-symmetry of `S`
- quadrature/SBP compatibility

For quadrature only, use `check_quadrature_exactness`:

```julia
report = check_quadrature_exactness(quad_basis, fsbp.x, fsbp.w;
                                    interval=(-1.0, 1.0),
                                    atol=1e-12,
                                    rtol=1e-12)
println(report)
```

## Exporting Operators

Use `print_fsbp_operator_python` to print arrays in a Python-friendly format:

```julia
print_fsbp_operator_python(fsbp; num_digits=16)
```

This is useful when copying nodes, weights, and matrices into a Python solver
or plotting script.

## Running Tests

Run the full package test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run a focused test file during development:

```bash
julia --project=. -e 'using Test; include("test/test_fsbp_operator.jl")'
julia --project=. -e 'using Test; include("test/test_bigfloat.jl")'
```

## Running Drivers

The `drivers/` directory contains executable examples:

```bash
julia --project=. drivers/quadrature_verification.jl
julia --project=. drivers/build_operator.jl
```

`drivers/build_operator.jl` shows end-to-end FSBP construction for polynomial
and non-polynomial bases.  `drivers/quadrature_verification.jl` shows how to
check quadrature rules independently of the FSBP builder.

## Package Layout

```text
.
|-- Project.toml
|-- README.md
|-- src/
|   |-- GaussFSBP.jl
|   |-- basis/
|   |   |-- Basis.jl
|   |   |-- FunctionBasis.jl
|   |   |-- BasisEvaluation.jl
|   |   `-- GeneralizedGaussInterop.jl
|   |-- builders/
|   |   |-- OperatorBuilders.jl
|   |   `-- OptimizedOperatorBuilders.jl
|   |-- verification/
|   |   |-- QuadratureVerification.jl
|   |   `-- OperatorVerification.jl
|   |-- io/
|   |   `-- FSBPOperatorPythonExport.jl
|   `-- utils/
|       |-- LinearAlgebraHelpers.jl
|       |-- ReferenceIntegrals.jl
|       `-- TypeConsistency.jl
|-- test/
|-- drivers/
`-- lib/
    `-- GeneralizedGauss.jl/
```

## Common Pitfalls / General Tips

- Mismatched interval types cause an error.  Use either all `Float64` endpoints
  or all `BigFloat` endpoints.
- The approximation basis needs derivatives.  The quadrature basis can use
  finite-difference or derivative-free quadrature solves, but `D * V == Vx`
  cannot be checked or constructed without derivatives of `op_basis`.
- If the number of quadrature nodes is less than the number of approximation
  basis functions, try `principal=:upper` or enlarge the quadrature basis.
- Ill-conditioned bases may need `orthogonalize=true`, higher BigFloat
  precision, or larger `lost_digits`.
- A looser `intermediate_tolerance` can significantly speed up the quadrature
  computation time.
