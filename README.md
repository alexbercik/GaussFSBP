# GaussFSBP

A Julia package for building generalized SBP/FEM/DG/SEM-style element operators
from arbitrary approximation bases.

## Status

This repository is under active development. Currently, only the repository
structure and the basis/verification scaffolding are implemented. Full operator
construction is not yet available.

## Local dependency: GeneralizedGauss.jl

This package will eventually depend on the local unregistered package
`GeneralizedGauss.jl`. Once you have obtained the `GeneralizedGauss.jl` package,
place it in the `lib/GeneralizedGauss.jl/` directory and then register it as a
development dependency by running:

```julia
julia --project=.
```

and then in the Julia REPL:

```julia
] dev lib/GeneralizedGauss.jl
```

Until `GeneralizedGauss.jl` is available, a self-contained placeholder
Gauss-Legendre integrator is used internally for testing and verification.

## Running tests

```julia
julia --project=. -e "import Pkg; Pkg.test()"
```

## Package structure

```
.
├── Project.toml
├── README.md
├── src/
│   ├── GaussFSBP.jl              # Main module
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
│       └── ReferenceIntegrals.jl # Gauss-Legendre reference integrator
├── test/
│   ├── runtests.jl
│   ├── test_basis.jl
│   ├── test_quadrature_verification.jl
│   └── test_polynomial_sbp_placeholders.jl
├── drivers/
│   ├── build_operator_driver.jl
│   └── quadrature_verification_driver.jl
└── lib/
    └── GeneralizedGauss.jl/      # Placeholder — place local package here
        └── .gitkeep
```
