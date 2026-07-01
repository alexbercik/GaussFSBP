"""
    SBPConstructionChecks.jl

Shared construction-time diagnostic checks for exact and optimization-based FSBP
builders.  Use `sbp_check_action` to control whether a failed check errors (default),
warns, or is ignored.
"""

const _SBP_CHECK_ACTION_HINT =
    " Set sbp_check_action=:warn or :ignore to proceed without stopping."

"""Format a scalar for construction-check messages (3 significant figures, scientific)."""
function _format_check_value(x::Real)
    xf = float(x)
    isfinite(xf) || return string(x)
    xf == 0 && return "0.00e+00"
    return Printf.@sprintf("%.2e", xf)
end

function _validate_sbp_check_action(action::Symbol,
                                    context::AbstractString = "sbp_check_action")
    action in (:warn, :error, :ignore) ||
        throw(ArgumentError("$context must be :warn, :error, or :ignore."))
    return action
end

function _apply_sbp_check_action(action::Symbol, msg::AbstractString)
    if action === :error
        error(msg * _SBP_CHECK_ACTION_HINT)
    elseif action === :warn
        println(" ")
        println(stderr, "WARNING [GaussFSBP]: ", msg)
        println(" ")
    end
    return nothing
end

function _construction_check_tolerance(scale, ::Type{T}) where T
    return T(100) * sqrt(eps(T)) * max(one(T), scale)
end

function _sbp_compatibility_residual(V, Vx, w, vL, vR)
    Vxw = Vx .* reshape(w, :, 1)
    Vw = V .* reshape(w, :, 1)
    return V' * Vxw + Vx' * Vw - (vR * vR' - vL * vL')
end

"""
    _check_sbp_compatibility(V, Vx, w, vL, vR, T; compatibility_tol, action)

Check the quadrature/SBP compatibility condition before constructing `S`.
"""
function _check_sbp_compatibility(V, Vx, w, vL, vR, ::Type{T};
                                    compatibility_tol = nothing,
                                    action::Symbol = :error) where T
    _validate_sbp_check_action(action)
    residual = norm(_sbp_compatibility_residual(V, Vx, w, vL, vR))
    scale = max(one(T), norm(V' * (Vx .* reshape(w, :, 1))),
                  norm(vR * vR' - vL * vL'))
    tol_eff = compatibility_tol === nothing ?
        _construction_check_tolerance(scale, T) : T(compatibility_tol)
    if residual > tol_eff
        msg = "Quadrature/SBP compatibility residual $(_format_check_value(residual)) " *
              "exceeds tolerance $(_format_check_value(tol_eff)); exact construction of S may be impossible."
        _apply_sbp_check_action(action, msg)
    end
    return residual
end

"""
    _check_boundary_matrix_match(E_from_Q, E; action)

Check that `Q + Qᵀ` agrees with the extrapolation boundary matrix `E`.
"""
function _check_boundary_matrix_match(E_from_Q, E; action::Symbol = :error)
    _validate_sbp_check_action(action)
    eltype(E_from_Q) == eltype(E) || throw(ArgumentError(
        "Boundary matrix type mismatch: $(eltype(E_from_Q)) vs $(eltype(E))."))
    T = eltype(E)
    residual = maximum(abs.(E_from_Q - E))
    scale = max(one(T), maximum(abs.(E_from_Q)), maximum(abs.(E)))
    tol = _construction_check_tolerance(scale, T)
    if residual > tol
        msg = "Boundary matrix from the unique square operator does not match " *
              "the extrapolation boundary matrix: residual $(_format_check_value(residual)) exceeds " *
              "tolerance $(_format_check_value(tol))."
        _apply_sbp_check_action(action, msg)
    end
    return residual
end

function _symmetry_tolerance(scale, ::Type{T}) where T
    return T(100) * sqrt(eps(T)) * max(one(T), scale)
end

function _check_flip_symmetric_grid(x, w, xL, xR; action::Symbol = :error)
    _validate_sbp_check_action(action)
    T = eltype(x)
    node_scale = max(one(T), maximum(abs.(x)), abs(xL), abs(xR))
    node_tol = _symmetry_tolerance(node_scale, T)
    center = xL + xR
    node_err = maximum(abs.(x .+ reverse(x) .- center))
    if node_err > node_tol
        msg = "extrapolation_symmetry=:flip requires reflection-paired nodes; " *
              "max |x[i] + x[N+1-i] - (xL+xR)| = $(_format_check_value(node_err)) exceeds $(_format_check_value(node_tol))."
        _apply_sbp_check_action(action, msg)
    end

    weight_scale = max(one(T), maximum(abs.(w)))
    weight_tol = _symmetry_tolerance(weight_scale, T)
    weight_err = maximum(abs.(w .- reverse(w)))
    if weight_err > weight_tol
        msg = "extrapolation_symmetry=:flip requires reflection-paired weights; " *
              "max |w[i] - w[N+1-i]| = $(_format_check_value(weight_err)) exceeds $(_format_check_value(weight_tol))."
        _apply_sbp_check_action(action, msg)
    end
    return nothing
end

function _check_constraint_residual(C, t, d, context::AbstractString;
                                    action::Symbol = :error)
    _validate_sbp_check_action(action)
    T = eltype(t)
    residual = norm(C * t - d)
    scale = max(one(T), norm(C) * norm(t), norm(d))
    tol = T(1000) * sqrt(eps(T)) * scale
    if residual > tol
        msg = "$context: exact flip-symmetric extrapolation constraints are inconsistent; " *
              "residual $(_format_check_value(residual)) exceeds $(_format_check_value(tol))."
        _apply_sbp_check_action(action, msg)
    end
    return nothing
end
