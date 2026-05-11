#!/usr/bin/env python3
"""
calc_basis.py

Symbolic helper for function-space SBP / FSBP basis construction.

Given a list of input basis functions F = {f_i}, this script computes

    (a) the derivatives f_i',
    (b) the quadrature basis G = (F F)' = span{ d/dx (f_i f_j) },
    (c) the derivatives g_k' of the quadrature-basis functions.

For the second-derivative FSBP construction of Glaubitz et al., the first-
derivative operator should be exact on F + F'. In that case, use

    --mode second

which first builds H = span(F union F') and then computes

    G = (H H)'.

Examples
--------
    python calc_basis.py "1, x, x^2, exp(x), sqrt(x)"
    python calc_basis.py "[1, x, x^2, exp(x)]" --mode first
    python calc_basis.py "1, x, exp(x^2)" --mode second
    python calc_basis.py "1, sin(pi*x), cos(pi*x)" --mode both
    python calc_basis.py "1, x, x^2, exp(x)" --format latex

Notes
-----
- Use ^ or ** for powers. Both are accepted.
- The parser accepts common SymPy functions: exp, sqrt, log, sin, cos, tan,
  sinh, cosh, asin, acos, atan, erf, etc.
- The script tries to reduce spanning lists to bases by removing expressions
  it recognizes as linearly dependent. This is useful, but symbolic linear
  independence for arbitrary functions is a hard problem. Use --no-reduce to
  see the unreduced generators.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


# -----------------------------------------------------------------------------
# Basic symbolic utilities
# -----------------------------------------------------------------------------


def clean_expr(expr: sp.Expr) -> sp.Expr:
    """Return a reasonably simplified exact symbolic expression."""
    expr = sp.sympify(expr)
    expr = sp.simplify(expr)
    expr = sp.together(expr)
    expr = sp.factor(expr)
    return expr


def is_zero_expr(expr: sp.Expr) -> bool:
    """Best-effort symbolic zero test."""
    expr = clean_expr(expr)
    if expr == 0:
        return True
    return bool(expr.equals(0))


def depends_on(expr: sp.Expr, x: sp.Symbol) -> bool:
    return bool(sp.sympify(expr).has(x))


def top_level_comma_split(text: str) -> List[str]:
    """Split a string on top-level commas, respecting (), [], and {}."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    pieces: List[str] = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())

    for k, ch in enumerate(text):
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced closing bracket in basis string.")
        elif ch == "," and depth == 0:
            pieces.append(text[start:k].strip())
            start = k + 1

    if depth != 0:
        raise ValueError("Unbalanced brackets in basis string.")

    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def make_local_dict(x: sp.Symbol) -> dict:
    """Safe-ish namespace of names allowed in basis expressions."""
    # This is still SymPy's parser, so treat command-line input as trusted.
    return {
        "x": x,
        "X": x,
        "pi": sp.pi,
        "Pi": sp.pi,
        "PI": sp.pi,
        "E": sp.E,
        "e": sp.E,
        "I": sp.I,
        "oo": sp.oo,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "ln": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "erf": sp.erf,
        "erfc": sp.erfc,
        "Abs": sp.Abs,
        "Rational": sp.Rational,
    }


def parse_basis(text: str, x: sp.Symbol) -> List[sp.Expr]:
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    local_dict = make_local_dict(x)
    pieces = top_level_comma_split(text)
    if not pieces:
        raise ValueError("No basis functions were provided.")

    basis = []
    for piece in pieces:
        try:
            expr = parse_expr(
                piece,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            )
        except Exception as exc:
            raise ValueError(f"Could not parse basis entry {piece!r}: {exc}") from exc
        basis.append(clean_expr(expr))
    return basis


# -----------------------------------------------------------------------------
# Span reduction
# -----------------------------------------------------------------------------


def candidate_sample_points() -> List[sp.Rational]:
    """Rational sample points used in best-effort span-dependence checks."""
    # Mostly positive points make examples with sqrt(x) and log(x) easier.
    # We still include 0 and a few negatives for polynomial/trig cases.
    return [
        sp.Rational(1, 7),
        sp.Rational(1, 5),
        sp.Rational(1, 3),
        sp.Rational(1, 2),
        sp.Rational(2, 3),
        sp.Rational(3, 4),
        sp.Rational(4, 5),
        sp.Rational(1, 1),
        sp.Rational(3, 2),
        sp.Rational(2, 1),
        sp.Rational(0, 1),
        sp.Rational(-1, 2),
        sp.Rational(-1, 1),
    ]


def finite_after_substitution(expr: sp.Expr, x: sp.Symbol, pt: sp.Expr) -> bool:
    try:
        val = sp.simplify(expr.subs(x, pt))
    except Exception:
        return False
    bad_atoms = {sp.zoo, sp.oo, -sp.oo, sp.nan}
    if val in bad_atoms:
        return False
    if any(atom in bad_atoms for atom in val.atoms()):
        return False
    return True


def is_constant_multiple(candidate: sp.Expr, basis_expr: sp.Expr, x: sp.Symbol) -> bool:
    if is_zero_expr(basis_expr):
        return False
    try:
        ratio = clean_expr(candidate / basis_expr)
    except Exception:
        return False
    return not depends_on(ratio, x)


def is_in_span(
    candidate: sp.Expr,
    basis: Sequence[sp.Expr],
    x: sp.Symbol,
    *,
    aggressive: bool = False,
) -> bool:
    """
    Best-effort check whether candidate is in span(basis).

    The default check is deliberately fast: it removes exact duplicates and
    constant multiples. With aggressive=True, the routine also tries to solve
    for constant coefficients using sample-point equations and then verifies
    the resulting symbolic identity. Aggressive reduction can be slow for mixed
    algebraic/transcendental bases.
    """
    candidate = clean_expr(candidate)
    if is_zero_expr(candidate):
        return True
    if not basis:
        return False

    for b in basis:
        if clean_expr(candidate - b) == 0 or bool(clean_expr(candidate - b).equals(0)):
            return True
        if is_constant_multiple(candidate, b, x):
            return True

    if not aggressive:
        return False

    # Try to detect a general constant-coefficient linear combination.
    n = len(basis)
    good_points = []
    all_exprs = list(basis) + [candidate]
    for pt in candidate_sample_points():
        if all(finite_after_substitution(expr, x, pt) for expr in all_exprs):
            good_points.append(pt)
        if len(good_points) >= max(n + 3, 2 * n):
            break

    if len(good_points) < n:
        return False

    coeffs = sp.symbols(f"c0:{n}")
    equations = []
    for pt in good_points[: max(n + 2, n)]:
        lhs = candidate.subs(x, pt)
        rhs = sum(coeffs[j] * basis[j].subs(x, pt) for j in range(n))
        equations.append(clean_expr(rhs - lhs))

    try:
        sol_set = sp.linsolve(equations, coeffs)
    except Exception:
        return False

    if sol_set == sp.EmptySet:
        return False

    try:
        sol = next(iter(sol_set))
    except StopIteration:
        return False

    free_symbols = set().union(*(s.free_symbols for s in sol)) - set(coeffs)
    if free_symbols:
        sol = tuple(s.subs({p: 0 for p in free_symbols}) for s in sol)

    residual = clean_expr(candidate - sum(sol[j] * basis[j] for j in range(n)))
    return is_zero_expr(residual)


def reduce_span(
    exprs: Iterable[sp.Expr],
    x: sp.Symbol,
    *,
    do_reduce: bool = True,
    aggressive: bool = False,
) -> List[sp.Expr]:
    """Remove zero expressions and recognized linear dependencies."""
    out: List[sp.Expr] = []
    for expr in exprs:
        expr = clean_expr(expr)
        if is_zero_expr(expr):
            continue
        if do_reduce and is_in_span(expr, out, x, aggressive=aggressive):
            continue
        # Even with --no-reduce, remove literal duplicates.
        if not do_reduce and any(clean_expr(expr - b) == 0 for b in out):
            continue
        out.append(expr)
    return out


# -----------------------------------------------------------------------------
# Basis construction
# -----------------------------------------------------------------------------


@dataclass
class BasisData:
    input_basis: List[sp.Expr]
    input_derivatives: List[sp.Expr]
    operator_basis: List[sp.Expr]
    raw_quadrature_generators: List[Tuple[Tuple[int, int], sp.Expr]]
    quadrature_basis: List[sp.Expr]
    quadrature_derivatives: List[sp.Expr]


def derivatives(basis: Sequence[sp.Expr], x: sp.Symbol) -> List[sp.Expr]:
    return [clean_expr(sp.diff(f, x)) for f in basis]


def product_derivative_generators(
    basis: Sequence[sp.Expr], x: sp.Symbol
) -> List[Tuple[Tuple[int, int], sp.Expr]]:
    """Return raw generators d/dx(f_i f_j), using i <= j."""
    out: List[Tuple[Tuple[int, int], sp.Expr]] = []
    for i, fi in enumerate(basis):
        for j in range(i, len(basis)):
            fj = basis[j]
            gij = clean_expr(sp.diff(fi * fj, x))
            if not is_zero_expr(gij):
                out.append(((i, j), gij))
    return out


def build_basis_data(
    input_basis: Sequence[sp.Expr],
    x: sp.Symbol,
    *,
    mode: str,
    do_reduce: bool,
    aggressive_reduce: bool = False,
) -> BasisData:
    input_basis = reduce_span(input_basis, x, do_reduce=do_reduce, aggressive=aggressive_reduce)
    input_derivatives = derivatives(input_basis, x)

    if mode == "first":
        operator_basis = input_basis
    elif mode == "second":
        operator_basis = reduce_span(
            list(input_basis) + list(input_derivatives),
            x,
            do_reduce=do_reduce,
            aggressive=aggressive_reduce,
        )
    else:
        raise ValueError(f"Unexpected mode {mode!r}.")

    raw_generators = product_derivative_generators(operator_basis, x)
    quadrature_basis = reduce_span(
        [expr for _, expr in raw_generators],
        x,
        do_reduce=do_reduce,
        aggressive=aggressive_reduce,
    )
    quadrature_derivatives = derivatives(quadrature_basis, x)

    return BasisData(
        input_basis=list(input_basis),
        input_derivatives=input_derivatives,
        operator_basis=operator_basis,
        raw_quadrature_generators=raw_generators,
        quadrature_basis=quadrature_basis,
        quadrature_derivatives=quadrature_derivatives,
    )


# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------


def format_expr(expr: sp.Expr, style: str) -> str:
    if style == "str":
        return str(expr)
    if style == "sstr":
        return sp.sstr(expr)
    if style == "pretty":
        return sp.pretty(expr, use_unicode=True)
    if style == "latex":
        return sp.latex(expr)
    raise ValueError(f"Unknown format style {style!r}.")


def print_rule(char: str = "=", width: int = 88) -> None:
    print(char * width)


def print_expr_list(title: str, exprs: Sequence[sp.Expr], *, style: str) -> None:
    print()
    print(title)
    print_rule("-")
    if not exprs:
        print("  <empty>")
        return
    for k, expr in enumerate(exprs, start=1):
        text = format_expr(expr, style)
        lines = text.splitlines()
        if len(lines) == 1:
            print(f"  [{k:02d}] {lines[0]}")
        else:
            print(f"  [{k:02d}]")
            for line in lines:
                print(f"       {line}")


def print_labeled_generators(
    title: str,
    generators: Sequence[Tuple[Tuple[int, int], sp.Expr]],
    *,
    style: str,
) -> None:
    print()
    print(title)
    print_rule("-")
    if not generators:
        print("  <empty>")
        return
    for k, ((i, j), expr) in enumerate(generators, start=1):
        text = format_expr(expr, style)
        prefix = f"  [{k:02d}] d/dx(F[{i+1}]*F[{j+1}]) = "
        lines = text.splitlines()
        if len(lines) == 1:
            print(prefix + lines[0])
        else:
            print(prefix)
            for line in lines:
                print(" " * len(prefix) + line)


def print_basis_report(
    data: BasisData,
    *,
    mode: str,
    style: str,
    show_generators: bool,
    do_reduce: bool,
) -> None:
    print_rule("=")
    print(f"Symbolic basis report  |  mode = {mode}")
    print_rule("=")

    if mode == "first":
        print("Quadrature space used here: G = (F F)' = span{ d/dx(f_i f_j) }.")
    elif mode == "second":
        print("Second-derivative FSBP mode.")
        print("First form H = span(F union F'), then use G = (H H)'.")

    print(f"Span reduction: {'on' if do_reduce else 'off, except literal duplicates'}")

    print_expr_list("Input basis F", data.input_basis, style=style)
    print_expr_list("Derivatives F'", data.input_derivatives, style=style)

    if mode == "second":
        print_expr_list("Augmented operator basis H = span(F union F')", data.operator_basis, style=style)

    if show_generators:
        label = "Raw quadrature generators d/dx(F[i]*F[j])"
        if mode == "second":
            label = "Raw quadrature generators d/dx(H[i]*H[j])"
        print_labeled_generators(label, data.raw_quadrature_generators, style=style)

    print()
    print(
        f"Raw nonzero product-derivative generators: {len(data.raw_quadrature_generators)}"
    )
    print(f"Reduced quadrature-basis size: {len(data.quadrature_basis)}")

    print_expr_list("Quadrature basis G", data.quadrature_basis, style=style)
    print_expr_list("Derivatives G'", data.quadrature_derivatives, style=style)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute symbolic derivatives, FSBP quadrature basis functions, "
            "and their derivatives from a list of basis functions."
        )
    )
    parser.add_argument(
        "basis",
        help=(
            "Comma-separated basis functions, e.g. "
            "'1, x, x^2, exp(x), sqrt(x)' or '[1, x, x^2]'."
        ),
    )
    parser.add_argument(
        "--var",
        default="x",
        help="Symbolic variable name. Default: x.",
    )
    parser.add_argument(
        "--positive-var",
        action="store_true",
        help="Assume the variable is positive. Useful for sqrt/log simplification.",
    )
    parser.add_argument(
        "--mode",
        choices=("first", "second", "both"),
        default="first",
        help=(
            "first: use G=(F F)'. "
            "second: use H=span(F union F') and G=(H H)'. "
            "both: print both reports. Default: first."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("str", "sstr", "pretty", "latex"),
        default="str",
        help="Expression output format. Default: str.",
    )
    parser.add_argument(
        "--show-generators",
        action="store_true",
        help="Also print all raw product-derivative generators before reduction.",
    )
    parser.add_argument(
        "--no-reduce",
        action="store_true",
        help=(
            "Do not try to reduce spanning lists to linearly independent bases. "
            "Literal duplicate expressions are still removed."
        ),
    )
    parser.add_argument(
        "--aggressive-reduce",
        action="store_true",
        help=(
            "Try a slower sample-point linear-dependence test, in addition to "
            "the default duplicate/constant-multiple checks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.positive_var:
        x = sp.Symbol(args.var, positive=True, real=True)
    else:
        x = sp.Symbol(args.var, real=True)

    try:
        input_basis = parse_basis(args.basis, x)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    do_reduce = not args.no_reduce

    modes = ["first", "second"] if args.mode == "both" else [args.mode]
    for k, mode in enumerate(modes):
        if k > 0:
            print("\n")
        data = build_basis_data(
            input_basis,
            x,
            mode=mode,
            do_reduce=do_reduce,
            aggressive_reduce=args.aggressive_reduce,
        )
        print_basis_report(
            data,
            mode=mode,
            style=args.format,
            show_generators=args.show_generators,
            do_reduce=do_reduce,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
