"""
    Verification.jl

NOTE: This file is no longer included directly by GaussFSBP.jl.
The individual verification files are included separately to control
dependency ordering (OperatorVerification.jl depends on FSBPOperator
from OperatorBuilders.jl).

See GaussFSBP.jl for the actual include order:
1. QuadratureVerification.jl  (no dependency on builders)
2. OperatorBuilders.jl        (defines FSBPOperator)
3. OperatorVerification.jl    (depends on FSBPOperator)
"""
