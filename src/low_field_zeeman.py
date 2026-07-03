from __future__ import annotations

import math
import re
from dataclasses import dataclass

from scipy.constants import physical_constants
from sympy import Rational


MU_B_OVER_H_HZ_PER_T = physical_constants["Bohr magneton in Hz/T"][0]
GAUSS_TO_TESLA = 1.0e-4
I87 = Rational(9, 2)


@dataclass(frozen=True)
class ZeemanResult:
    g_j: float
    g_f: float
    shift_hz: float
    model: str


def parse_half_integer(value: str | int | float | Rational) -> Rational:
    return value if isinstance(value, Rational) else Rational(str(value))


def lande_g_from_term(term: str, J: int | float) -> float:
    """LS-coupling Landé g_J. This is a first-pass engineering estimate."""
    match = re.fullmatch(r"(\d+)([SPDFGHIK])(?:\d+)?", term.strip())
    if not match:
        raise ValueError(f"Cannot infer L and S from term {term!r}.")
    multiplicity = int(match.group(1))
    l_symbol = match.group(2)
    l_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6, "K": 7}
    L = float(l_map[l_symbol])
    S = multiplicity_to_spin(multiplicity)
    Jf = float(J)
    if Jf == 0.0:
        return 0.0
    return 1.0 + (Jf * (Jf + 1.0) + S * (S + 1.0) - L * (L + 1.0)) / (2.0 * Jf * (Jf + 1.0))


def multiplicity_to_spin(multiplicity: int) -> float:
    return (multiplicity - 1) / 2.0


def hyperfine_g_f(g_j: float, J: int | float, F: Rational, g_i_effective: float = 0.0) -> float:
    """Low-field hyperfine g_F.

    `g_i_effective` is in Bohr-magneton units. It defaults to zero because the
    nuclear term is orders of magnitude smaller than the electronic term and
    should be supplied from a precision Sr-87 clock reference when needed.
    """
    Jf = float(J)
    Ff = float(F)
    If = float(I87)
    if Ff == 0.0:
        return 0.0
    denom = 2.0 * Ff * (Ff + 1.0)
    electronic = g_j * (Ff * (Ff + 1.0) + Jf * (Jf + 1.0) - If * (If + 1.0)) / denom
    nuclear = g_i_effective * (Ff * (Ff + 1.0) + If * (If + 1.0) - Jf * (Jf + 1.0)) / denom
    return electronic + nuclear


def low_field_zeeman_shift_hz(
    term: str,
    J: int | float,
    F: str | Rational,
    mF: str | Rational,
    magnetic_field_g: float,
    g_i_effective: float = 0.0,
) -> ZeemanResult:
    F_value = parse_half_integer(F)
    mF_value = parse_half_integer(mF)
    g_j = lande_g_from_term(term, J)
    g_f = hyperfine_g_f(g_j, J, F_value, g_i_effective=g_i_effective)
    shift_hz = g_f * MU_B_OVER_H_HZ_PER_T * (magnetic_field_g * GAUSS_TO_TESLA) * float(mF_value)
    return ZeemanResult(g_j=g_j, g_f=g_f, shift_hz=shift_hz, model="low_field_gF_mF_B")


def add_low_field_zeeman_to_result(
    result: dict,
    magnetic_field_g: float,
    term: str | None = None,
    g_i_effective: float = 0.0,
) -> dict:
    """Annotate a `compute_sr87_polarizability()` result with low-field Zeeman shifts."""
    annotated = dict(result)
    electronic = dict(result["electronic_state"])
    term_value = term if term is not None else _term_from_state_label(electronic["state_label"])
    rows = []
    for row in result["hyperfine_results"]:
        z = low_field_zeeman_shift_hz(
            term=term_value,
            J=electronic["J"],
            F=row["F"],
            mF=row["mF"],
            magnetic_field_g=magnetic_field_g,
            g_i_effective=g_i_effective,
        )
        updated = dict(row)
        updated["zeeman_shift_hz"] = z.shift_hz
        updated["zeeman_shift_mhz"] = z.shift_hz / 1e6
        updated["g_J_estimate"] = z.g_j
        updated["g_F_estimate"] = z.g_f
        updated["zeeman_model"] = z.model
        if "light_shift_mhz" in updated:
            updated["light_plus_zeeman_shift_mhz"] = updated["light_shift_mhz"] + updated["zeeman_shift_mhz"]
        rows.append(updated)
    annotated["input"] = dict(result["input"])
    annotated["input"]["magnetic_field_G"] = magnetic_field_g
    annotated["input"]["g_i_effective"] = g_i_effective
    annotated["hyperfine_results"] = rows
    annotated["zeeman_note"] = (
        "First-order low-field Zeeman estimate. J=0 electronic states default to g_J=0, "
        "so nuclear/clock-state Zeeman shifts require an explicit precision g_i_effective or a later clock-specific model."
    )
    return annotated


def _term_from_state_label(state_label: str) -> str:
    match = re.search(r"(\d+[SPDFGHIK])", state_label)
    if not match:
        raise ValueError(f"Cannot infer term from state label {state_label!r}.")
    return match.group(1)
