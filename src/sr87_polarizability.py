from __future__ import annotations

import argparse
import json
from functools import lru_cache

from scipy.constants import c, epsilon_0, h
from sympy import N, Rational, S
from sympy.physics.wigner import clebsch_gordan

import reproduce_sr_polarizability as sr88
import field_models
import low_field_zeeman
import scattering_heating
from quantization_axis_geometry import (
    GEOMETRY_NAMES,
    PolarizabilityComponents as GeometryPolarizabilityComponents,
    build_geometry,
    effective_polarizability,
)


I87 = Rational(9, 2)
ALPHA_AU_TO_SI = 1.64877727436e-41  # C m^2 / V


def parse_half_integer(value: str, label: str) -> Rational:
    try:
        parsed = Rational(value)
    except Exception as exc:  # pragma: no cover - argparse guards this path in practice
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    return parsed


def rational_str(value: Rational) -> str:
    return str(value)


def allowed_F_values(J: int) -> list[Rational]:
    lower = abs(I87 - S(J))
    upper = I87 + S(J)
    values: list[Rational] = []
    current = lower
    while current <= upper:
        values.append(current)
        current += 1
    return values


def allowed_mF_values(F: Rational) -> list[Rational]:
    values: list[Rational] = []
    current = -F
    while current <= F:
        values.append(current)
        current += 1
    return values


def light_shift_mhz(alpha_au: float, intensity_w_m2: float) -> float:
    alpha_si = alpha_au * ALPHA_AU_TO_SI
    nu_shift_hz = -alpha_si * intensity_w_m2 / (4.0 * epsilon_0 * c * h)
    return nu_shift_hz / 1e6


@lru_cache(maxsize=8)
def load_model_data(max_energy_cm: float, include_optional_residuals: bool):
    config = sr88.ModelConfig(
        max_energy_cm=max_energy_cm,
        include_optional_residuals=include_optional_residuals,
    )
    levels = sr88.load_levels(sr88.LEVELS_PATH)
    lines = sr88.load_lines(sr88.LINES_PATH)
    literature = sr88.load_literature_matrix_elements(sr88.LITERATURE_MATRIX_PATH)
    residuals = sr88.load_residuals(sr88.RESIDUALS_PATH)
    line_index = sr88.build_line_index(lines)
    return config, levels, line_index, literature, residuals


def build_catalog_by_state(
    state_label: str,
    max_energy_cm: float,
    include_optional_residuals: bool,
):
    config, levels, line_index, literature, residuals = load_model_data(
        max_energy_cm=max_energy_cm,
        include_optional_residuals=include_optional_residuals,
    )
    if state_label not in levels:
        available = ", ".join(sorted(levels))
        raise ValueError(f"Unknown state_label '{state_label}'. Available levels: {available}")
    catalog = sr88.build_channel_catalog(
        state_label,
        levels=levels,
        line_index=line_index,
        literature_index=literature,
        config=config,
    )
    return config, levels, residuals, {state_label: catalog}


def electronic_components_for_state(
    state_label: str,
    wavelength_nm: float,
    max_energy_cm: float,
    include_optional_residuals: bool,
):
    config, levels, residuals, catalog_by_state = build_catalog_by_state(
        state_label,
        max_energy_cm=max_energy_cm,
        include_optional_residuals=include_optional_residuals,
    )
    components = sr88.polarizability_components(
        state_label=state_label,
        wavelength_nm=wavelength_nm,
        catalog_by_state=catalog_by_state,
        residuals=residuals,
        config=config,
    )
    level = levels[state_label]
    return config, level, components


def electronic_model_for_state(
    state_label: str,
    wavelength_nm: float,
    max_energy_cm: float,
    include_optional_residuals: bool,
):
    config, levels, residuals, catalog_by_state = build_catalog_by_state(
        state_label,
        max_energy_cm=max_energy_cm,
        include_optional_residuals=include_optional_residuals,
    )
    components = sr88.polarizability_components(
        state_label=state_label,
        wavelength_nm=wavelength_nm,
        catalog_by_state=catalog_by_state,
        residuals=residuals,
        config=config,
    )
    return config, levels[state_label], components, catalog_by_state


def mJ_resolved_alpha_eff(
    level: sr88.Level,
    components: sr88.PolarizabilityComponents,
    geometry_name: str,
    theta_deg: float,
    gamma_deg: float,
) -> tuple[dict[int, float], object]:
    geometry = build_geometry(geometry_name, theta_deg=theta_deg, gamma_deg=gamma_deg)
    geometry_components = GeometryPolarizabilityComponents(
        scalar=components.scalar,
        vector=components.vector,
        tensor=components.tensor,
    )
    alpha_by_mJ: dict[int, float] = {}
    for mJ in range(-level.J, level.J + 1):
        alpha_by_mJ[mJ] = effective_polarizability(
            J=level.J,
            mJ=mJ,
            components=geometry_components,
            geometry=geometry,
        )
    return alpha_by_mJ, geometry


def hyperfine_alpha_from_mJ(
    F: Rational,
    mF: Rational,
    J: int,
    alpha_by_mJ: dict[int, float],
) -> tuple[float, dict[str, float]]:
    total = 0.0
    weights: dict[str, float] = {}
    for mJ, alpha in alpha_by_mJ.items():
        mI = mF - S(mJ)
        if abs(mI) > I87:
            continue
        cg = clebsch_gordan(S(J), I87, F, S(mJ), mI, mF)
        weight = float(N(cg**2))
        if weight == 0.0:
            continue
        weights[str(mJ)] = weight
        total += weight * alpha
    return total, weights


def selected_hyperfine_targets(
    J: int,
    F_value: Rational | None,
    mF_value: Rational | None,
) -> list[tuple[Rational, Rational]]:
    all_F = allowed_F_values(J)
    if F_value is not None and F_value not in all_F:
        supported = ", ".join(rational_str(value) for value in all_F)
        raise ValueError(f"F={F_value} is not allowed for J={J}. Supported F values: {supported}")

    if mF_value is not None and F_value is None:
        raise ValueError("mF requires an explicit F value.")

    selected: list[tuple[Rational, Rational]] = []
    F_values = [F_value] if F_value is not None else all_F
    for F in F_values:
        allowed_mF = allowed_mF_values(F)
        if mF_value is not None:
            if mF_value not in allowed_mF:
                supported = ", ".join(rational_str(value) for value in allowed_mF)
                raise ValueError(f"mF={mF_value} is not allowed for F={F}. Supported mF values: {supported}")
            selected.append((F, mF_value))
        else:
            selected.extend((F, mF) for mF in allowed_mF)
    return selected


def compute_sr87_polarizability(
    state_label: str,
    wavelength_nm: float,
    geometry_name: str,
    theta_deg: float = 0.0,
    gamma_deg: float = 0.0,
    F_value: Rational | None = None,
    mF_value: Rational | None = None,
    intensity_w_m2: float | None = None,
    max_energy_cm: float = 70000.0,
    include_optional_residuals: bool = False,
    include_scattering: bool = False,
    magnetic_field_g: float | None = None,
) -> dict:
    _, level, components, catalog_by_state = electronic_model_for_state(
        state_label=state_label,
        wavelength_nm=wavelength_nm,
        max_energy_cm=max_energy_cm,
        include_optional_residuals=include_optional_residuals,
    )
    alpha_by_mJ, geometry = mJ_resolved_alpha_eff(
        level=level,
        components=components,
        geometry_name=geometry_name,
        theta_deg=theta_deg,
        gamma_deg=gamma_deg,
    )
    targets = selected_hyperfine_targets(level.J, F_value, mF_value)

    hyperfine_results = []
    for F, mF in targets:
        alpha_total, weights = hyperfine_alpha_from_mJ(F=F, mF=mF, J=level.J, alpha_by_mJ=alpha_by_mJ)
        row = {
            "F": rational_str(F),
            "mF": rational_str(mF),
            "alpha_au": alpha_total,
            "cg_weights": weights,
        }
        if intensity_w_m2 is not None:
            row["light_shift_mhz"] = light_shift_mhz(alpha_total, intensity_w_m2)
        hyperfine_results.append(row)

    geometry_vector = [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in geometry.normalized()
    ]
    result = {
        "input": {
            "state_label": state_label,
            "wavelength_nm": wavelength_nm,
            "geometry_name": geometry_name,
            "theta_deg": theta_deg,
            "gamma_deg": gamma_deg,
            "F": None if F_value is None else rational_str(F_value),
            "mF": None if mF_value is None else rational_str(mF_value),
            "intensity_w_m2": intensity_w_m2,
            "max_energy_cm": max_energy_cm,
            "include_optional_residuals": include_optional_residuals,
        },
        "electronic_state": {
            "state_label": level.state_label,
            "J": level.J,
            "energy_cm": level.energy_cm,
        },
        "geometry": {
            "name": geometry.name,
            "polarization_vector": geometry_vector,
        },
        "electronic_components_au": {
            "scalar": components.scalar,
            "vector": components.vector,
            "tensor": components.tensor,
        },
        "electronic_alpha_eff_by_mJ_au": {
            str(mJ): alpha for mJ, alpha in alpha_by_mJ.items()
        },
        "hyperfine_results": hyperfine_results,
    }
    if include_scattering:
        summary = scattering_heating.scalar_scattering_summary(
            state_label=state_label,
            wavelength_nm=wavelength_nm,
            catalog_by_state=catalog_by_state,
        )
        result["scattering_heating"] = {
            "alpha_imag_scalar_au": summary.alpha_imag_scalar_au,
            "gamma_sc_s_inv_per_W_m2": summary.gamma_sc_s_inv_per_w_m2,
            "recoil_energy_kHz": summary.recoil_energy_hz / 1e3,
            "heating_rate_nK_s_per_W_m2": summary.heating_rate_nk_s_per_w_m2,
            "model": summary.model,
            "note": "First-pass scalar E1 scattering estimate from selected SOS channels; vector/tensor imaginary parts are not included.",
        }
        if intensity_w_m2 is not None:
            result["scattering_heating"]["gamma_sc_s_inv"] = (
                summary.gamma_sc_s_inv_per_w_m2 * intensity_w_m2
            )
            result["scattering_heating"]["heating_rate_nK_s"] = (
                summary.heating_rate_nk_s_per_w_m2 * intensity_w_m2
            )
    if magnetic_field_g is not None:
        result = low_field_zeeman.add_low_field_zeeman_to_result(
            result,
            magnetic_field_g=magnetic_field_g,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Sr-87 hyperfine polarizability from the electronic Sr model and geometry inputs.",
    )
    parser.add_argument("--state-label", required=True, help="Electronic state label, e.g. '5s5p 3P1'.")
    parser.add_argument("--wavelength-nm", required=True, type=float, help="Laser wavelength in nm.")
    parser.add_argument(
        "--geometry",
        default="linear_z",
        choices=GEOMETRY_NAMES,
        help="Geometry name defined in quantization_axis_geometry.py.",
    )
    parser.add_argument(
        "--theta-deg",
        type=float,
        default=0.0,
        help="Polarization angle for geometry='linear_xz'.",
    )
    parser.add_argument(
        "--gamma-deg",
        type=float,
        default=0.0,
        help="Ellipticity angle for geometry='elliptical_xy'.",
    )
    parser.add_argument("--F", help="Target hyperfine F, e.g. 11/2.")
    parser.add_argument("--mF", help="Target hyperfine mF, e.g. 9/2 or -3/2.")
    parser.add_argument(
        "--intensity-w-m2",
        type=float,
        help="Optional tweezer intensity. If provided, also report U/h in MHz.",
    )
    parser.add_argument("--power-W", type=float, help="Optional Gaussian beam power for center intensity.")
    parser.add_argument("--waist-um", type=float, help="Optional Gaussian beam waist for center intensity.")
    parser.add_argument("--transmission", type=float, default=1.0, help="Optional Gaussian beam transmission.")
    parser.add_argument("--include-scattering", action="store_true", help="Also report first-pass scalar scattering/heating estimates.")
    parser.add_argument("--magnetic-field-G", type=float, help="Optional low-field Zeeman B field in gauss.")
    parser.add_argument(
        "--max-energy-cm",
        type=float,
        default=70000.0,
        help="Maximum |Delta E| cutoff used to build the electronic channel catalog.",
    )
    parser.add_argument(
        "--include-optional-residuals",
        action="store_true",
        help="Enable optional residual corrections from residual_contributions_sr1.csv.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the full result JSON.",
    )
    return parser


def print_summary(result: dict) -> None:
    print("=== Sr-87 Polarizability Summary ===")
    print(f"state_label        : {result['input']['state_label']}")
    print(f"wavelength_nm      : {result['input']['wavelength_nm']}")
    print(f"geometry           : {result['geometry']['name']}")
    print(f"electronic J       : {result['electronic_state']['J']}")
    print(
        "components (a.u.)  : "
        f"scalar={result['electronic_components_au']['scalar']:.6f}, "
        f"vector={result['electronic_components_au']['vector']:.6f}, "
        f"tensor={result['electronic_components_au']['tensor']:.6f}"
    )
    print("mJ-resolved alpha_eff (a.u.):")
    for mJ, alpha in result["electronic_alpha_eff_by_mJ_au"].items():
        print(f"  mJ={mJ:>2}  alpha_eff={alpha:.6f}")
    print("hyperfine results:")
    for row in result["hyperfine_results"]:
        line = f"  F={row['F']:>4}  mF={row['mF']:>4}  alpha={row['alpha_au']:.6f} a.u."
        if "light_shift_mhz" in row:
            line += f"  U/h={row['light_shift_mhz']:.6f} MHz"
        if "zeeman_shift_mhz" in row:
            line += f"  Zeeman={row['zeeman_shift_mhz']:.6f} MHz"
        if "light_plus_zeeman_shift_mhz" in row:
            line += f"  total={row['light_plus_zeeman_shift_mhz']:.6f} MHz"
        print(line)
    if "scattering_heating" in result:
        sc = result["scattering_heating"]
        print("scattering/heating:")
        print(f"  gamma_sc/I       : {sc['gamma_sc_s_inv_per_W_m2']:.6e} s^-1 per W/m^2")
        print(f"  heating/I        : {sc['heating_rate_nK_s_per_W_m2']:.6e} nK/s per W/m^2")
        if "gamma_sc_s_inv" in sc:
            print(f"  gamma_sc         : {sc['gamma_sc_s_inv']:.6e} s^-1")
            print(f"  heating          : {sc['heating_rate_nK_s']:.6e} nK/s")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    F_value = parse_half_integer(args.F, "F") if args.F else None
    mF_value = parse_half_integer(args.mF, "mF") if args.mF else None
    intensity_w_m2 = args.intensity_w_m2
    if intensity_w_m2 is None and args.power_W is not None and args.waist_um is not None:
        intensity_w_m2 = field_models.gaussian_center_intensity_w_m2(
            power_w=args.power_W,
            waist_um=args.waist_um,
            transmission=args.transmission,
        )
    result = compute_sr87_polarizability(
        state_label=args.state_label,
        wavelength_nm=args.wavelength_nm,
        geometry_name=args.geometry,
        theta_deg=args.theta_deg,
        gamma_deg=args.gamma_deg,
        F_value=F_value,
        mF_value=mF_value,
        intensity_w_m2=intensity_w_m2,
        max_energy_cm=args.max_energy_cm,
        include_optional_residuals=args.include_optional_residuals,
        include_scattering=args.include_scattering,
        magnetic_field_g=args.magnetic_field_G,
    )
    print_summary(result)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Wrote JSON result to {args.output_json}")


if __name__ == "__main__":
    main()
