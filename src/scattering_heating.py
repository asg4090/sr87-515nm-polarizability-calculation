from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from scipy.constants import c, e, epsilon_0, h, hbar, k as k_B, physical_constants, pi

import reproduce_sr_polarizability as sr


ALPHA_AU_TO_SI = 1.64877727436e-41  # C m^2 / V
AU_TIME_S = physical_constants["atomic unit of time"][0]
EA0_SI = e * physical_constants["Bohr radius"][0]
SR87_MASS_KG = 86.908877497 * physical_constants["atomic mass constant"][0]


@dataclass(frozen=True)
class ScatteringSummary:
    alpha_imag_scalar_au: float
    gamma_sc_s_inv_per_w_m2: float
    recoil_energy_hz: float
    heating_rate_nk_s_per_w_m2: float
    model: str


def aki_s_from_rdme(delta_energy_cm: float, reduced_dipole_au: float, upper_j: int) -> float:
    omega_si = 2.0 * pi * c * (abs(delta_energy_cm) * 100.0)
    d2_si = (reduced_dipole_au * EA0_SI) ** 2
    return omega_si**3 * d2_si / (3.0 * pi * epsilon_0 * hbar * c**3 * (2 * upper_j + 1))


def channel_decay_rate_s(channel: sr.Channel) -> float:
    """Transition partial linewidth inferred from the selected RDME."""
    upper_j = channel.partner_J if channel.delta_energy_cm > 0 else channel.initial_J
    return aki_s_from_rdme(channel.delta_energy_cm, channel.reduced_dipole_au, upper_j)


def scalar_complex_channel_contribution(
    channel: sr.Channel,
    wavelength_nm: float,
    linewidth_s_inv: float | None = None,
) -> complex:
    omega = sr.omega_from_wavelength_au(wavelength_nm)
    delta = sr.omega_from_wavenumber_au(channel.delta_energy_cm)
    gamma = (linewidth_s_inv if linewidth_s_inv is not None else channel_decay_rate_s(channel)) * AU_TIME_S
    denominator = delta**2 - (omega + 0.5j * gamma) ** 2
    return (
        2.0
        / (3.0 * (2 * channel.initial_J + 1))
        * delta
        * channel.reduced_dipole_au**2
        / denominator
    )


def scattering_rate_per_intensity_from_imag_alpha(imag_alpha_au: float) -> float:
    """Rate in s^-1 per W/m^2 using the same 1/4 light-shift convention as the project."""
    alpha_imag_si = abs(imag_alpha_au) * ALPHA_AU_TO_SI
    return alpha_imag_si / (2.0 * epsilon_0 * c * hbar)


def recoil_energy_hz(wavelength_nm: float, mass_kg: float = SR87_MASS_KG) -> float:
    wavelength_m = wavelength_nm * 1e-9
    return h / (2.0 * mass_kg * wavelength_m**2)


def heating_rate_nk_s(scattering_rate_s_inv: float, wavelength_nm: float, mass_kg: float = SR87_MASS_KG) -> float:
    recoil_j = h * recoil_energy_hz(wavelength_nm, mass_kg)
    return 2.0 * recoil_j * scattering_rate_s_inv / k_B * 1e9


def selected_channel_scattering_rows(
    state_label: str,
    wavelength_nm: float,
    catalog_by_state: dict[str, list[sr.ChannelCandidate]],
) -> list[dict]:
    rows = []
    for channel in sr.selected_channels_for_state(state_label, wavelength_nm, catalog_by_state[state_label]):
        linewidth = channel_decay_rate_s(channel)
        alpha_complex = scalar_complex_channel_contribution(channel, wavelength_nm, linewidth)
        gamma_per_i = scattering_rate_per_intensity_from_imag_alpha(alpha_complex.imag)
        rows.append(
            {
                "state_label": state_label,
                "wavelength_nm": wavelength_nm,
                "partner_label": channel.partner_label,
                "delta_energy_cm": channel.delta_energy_cm,
                "source_kind": channel.source_kind,
                "source": channel.source,
                "reduced_dipole_au": channel.reduced_dipole_au,
                "partial_linewidth_s_inv": linewidth,
                "alpha_imag_scalar_au": abs(alpha_complex.imag),
                "gamma_sc_s_inv_per_W_m2": gamma_per_i,
                "recoil_energy_kHz": recoil_energy_hz(wavelength_nm) / 1e3,
                "heating_rate_nK_s_per_W_m2": heating_rate_nk_s(gamma_per_i, wavelength_nm),
            }
        )
    rows.sort(key=lambda item: item["gamma_sc_s_inv_per_W_m2"], reverse=True)
    return rows


def scalar_scattering_summary(
    state_label: str,
    wavelength_nm: float,
    catalog_by_state: dict[str, list[sr.ChannelCandidate]],
) -> ScatteringSummary:
    rows = selected_channel_scattering_rows(state_label, wavelength_nm, catalog_by_state)
    alpha_imag = math.fsum(row["alpha_imag_scalar_au"] for row in rows)
    gamma_per_i = math.fsum(row["gamma_sc_s_inv_per_W_m2"] for row in rows)
    return ScatteringSummary(
        alpha_imag_scalar_au=alpha_imag,
        gamma_sc_s_inv_per_w_m2=gamma_per_i,
        recoil_energy_hz=recoil_energy_hz(wavelength_nm),
        heating_rate_nk_s_per_w_m2=heating_rate_nk_s(gamma_per_i, wavelength_nm),
        model="scalar_partial_linewidth_sum",
    )
