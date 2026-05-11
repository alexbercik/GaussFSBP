# GaussFSBP

A Julia package for building generalized SBP/FEM/DG/SEM-style element operators
from arbitrary approximation bases, backed by
[GeneralizedGauss.jl](lib/GeneralizedGauss.jl/) for quadrature rule
construction on arbitrary function spaces.

## Status

This repository is under active development. The basis interface,
quadrature verification, and GeneralizedGauss.jl integration are in
place. Full operator construction (mass matrix, differentiation matrix,
SBP matrices) is not yet available.

## Setup

GaussFSBP depends on the local unregistered package `GeneralizedGauss.jl`,
which lives in `lib/GeneralizedGauss.jl/` (a separate git repository,
excluded from this repo's tracking via `.gitignore`).

### Julia 1.11+ (recommended)

The `[sources]` section in `Project.toml` resolves the local path
automatically. Just instantiate:

```julia
julia --project=.
] instantiate
```

### Julia 1.9–1.10

Register the local package as a dev dependency manually:

```julia
julia --project=.
] dev lib/GeneralizedGauss.jl
```

## Running tests

```julia
julia --project=. -e "import Pkg; Pkg.test()"
```

## Running drivers

```julia
julia --project=. drivers/quadrature_verification_driver.jl
julia --project=. drivers/build_operator_driver.jl
```

## Package structure

```
.
├── Project.toml
├── README.md
├── src/
│   ├── GaussFSBP.jl              # Main module (imports & re-exports GeneralizedGauss)
│   ├── basis/
│   │   ├── Basis.jl              # Abstract basis interface
│   │   ├── FunctionBasis.jl      # Concrete callable-function basis
│   │   └── BasisEvaluation.jl    # Evaluation utilities
│   ├── builders/
│   │   └── OperatorBuilders.jl   # Placeholder for operator construction
│   ├── verification/
│   │   ├── Verification.jl       # Verification exports
│   │   └── QuadratureVerification.jl  # Quadrature exactness checker
│   └── utils/
│       └── ReferenceIntegrals.jl # Adaptive Gauss-Legendre reference integrator
├── test/
│   ├── runtests.jl
│   ├── test_basis.jl
│   ├── test_quadrature_verification.jl
│   └── test_polynomial_sbp_placeholders.jl
├── drivers/
│   ├── build_operator_driver.jl          # End-to-end operator construction demo
│   └── quadrature_verification_driver.jl # Quadrature exactness verification demo
└── lib/
    └── GeneralizedGauss.jl/      # Local dependency (separate git repo)
```


# TODO:
- add something about the util calc_basis.py