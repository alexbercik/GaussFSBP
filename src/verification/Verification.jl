"""
    Verification.jl

Collects and re-exports all verification-related functionality.

Planned future checks (not yet implemented):
- Quadrature exactness checks (currently implemented).
- SBP accuracy checks: `D * V ≈ Vx`.
- SBP property checks: `Q + Q' ≈ B`.
- Interpolation / extrapolation-to-boundary checks.
"""

include("QuadratureVerification.jl")
