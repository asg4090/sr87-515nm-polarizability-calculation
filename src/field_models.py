from __future__ import annotations

import math

from scipy.constants import c, epsilon_0, h, pi


ALPHA_AU_TO_SI = 1.64877727436e-41  # C m^2 / V


def light_shift_mhz(alpha_au: float, intensity_w_m2: float) -> float:
    """AC Stark shift U/h in MHz, using the convention already used by the Sr-87 code."""
    alpha_si = alpha_au * ALPHA_AU_TO_SI
    shift_hz = -alpha_si * intensity_w_m2 / (4.0 * epsilon_0 * c * h)
    return shift_hz / 1e6


def gaussian_center_intensity_w_m2(power_w: float, waist_um: float, transmission: float = 1.0) -> float:
    if power_w < 0:
        raise ValueError("power_w must be non-negative.")
    if waist_um <= 0:
        raise ValueError("waist_um must be positive.")
    if transmission < 0:
        raise ValueError("transmission must be non-negative.")
    waist_m = waist_um * 1e-6
    return 2.0 * power_w * transmission / (pi * waist_m**2)


def rayleigh_range_um(waist_um: float, wavelength_nm: float) -> float:
    if waist_um <= 0:
        raise ValueError("waist_um must be positive.")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    waist_m = waist_um * 1e-6
    wavelength_m = wavelength_nm * 1e-9
    return pi * waist_m**2 / wavelength_m * 1e6


def gaussian_waist_um(waist_um: float, wavelength_nm: float, z_um: float) -> float:
    z_r_um = rayleigh_range_um(waist_um, wavelength_nm)
    return waist_um * math.sqrt(1.0 + (z_um / z_r_um) ** 2)


def gaussian_intensity_w_m2(
    r_um: float,
    z_um: float,
    power_w: float,
    waist_um: float,
    wavelength_nm: float,
    transmission: float = 1.0,
) -> float:
    """Paraxial Gaussian beam intensity at cylindrical coordinates (r, z)."""
    w_z_um = gaussian_waist_um(waist_um, wavelength_nm, z_um)
    center = gaussian_center_intensity_w_m2(power_w, waist_um, transmission)
    envelope = (waist_um / w_z_um) ** 2 * math.exp(-2.0 * (r_um / w_z_um) ** 2)
    return center * envelope


def lattice_period_um(wavelength_nm: float, crossing_angle_deg: float = 180.0) -> float:
    """Standing-wave period for two equal-frequency beams crossing at crossing_angle_deg."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    angle = math.radians(crossing_angle_deg)
    denom = 2.0 * math.sin(angle / 2.0)
    if denom <= 0:
        raise ValueError("crossing_angle_deg must be positive.")
    return (wavelength_nm * 1e-3) / denom


def standing_wave_1d_intensity_w_m2(
    x_um: float,
    single_beam_power_w: float,
    waist_um: float,
    wavelength_nm: float,
    transmission: float = 1.0,
    contrast: float = 1.0,
    phase_rad: float = 0.0,
    crossing_angle_deg: float = 180.0,
) -> float:
    """Simple equal-power 1D lattice at the beam center.

    `single_beam_power_w` is the power in each counter-propagating beam before
    transmission. The model ignores Gaussian envelope variation along x and is
    intended as a first-pass lattice-depth converter, not a full optical layout.
    """
    if not 0.0 <= contrast <= 1.0:
        raise ValueError("contrast must be between 0 and 1.")
    period = lattice_period_um(wavelength_nm, crossing_angle_deg)
    single_beam_i0 = gaussian_center_intensity_w_m2(single_beam_power_w, waist_um, transmission)
    return 2.0 * single_beam_i0 * (1.0 + contrast * math.cos(2.0 * pi * x_um / period + phase_rad))


def potential_mhz_from_intensity(alpha_au: float, intensity_w_m2: float) -> float:
    return light_shift_mhz(alpha_au, intensity_w_m2)
