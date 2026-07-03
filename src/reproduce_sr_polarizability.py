from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.constants import c, e, epsilon_0, hbar, physical_constants, pi
from scipy.optimize import brentq
from sympy.physics.wigner import wigner_6j


SRC_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SRC_DIR.parent
DATA_DIR = Path(os.environ.get("SR87_POL_DATA_DIR", PACKAGE_ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("SR87_POL_OUTPUT_DIR", PACKAGE_ROOT / "outputs")).resolve()

LEVELS_PATH = DATA_DIR / "levels_sr1.csv"
LINES_PATH = DATA_DIR / "lines_sr1.csv"
LITERATURE_MATRIX_PATH = DATA_DIR / "literature_matrix_elements_sr1.csv"
RESIDUALS_PATH = DATA_DIR / "residual_contributions_sr1.csv"
BENCHMARKS_PATH = DATA_DIR / "benchmarks_sr1.csv"

AU_TIME_S = physical_constants["atomic unit of time"][0]
EA0_SI = e * physical_constants["Bohr radius"][0]
MU_B_OVER_H_HZ_PER_T = physical_constants["Bohr magneton in Hz/T"][0]

ACCURACY_CODE_TO_FRACTION = {
    "AAA": 0.003,
    "AA": 0.01,
    "A+": 0.02,
    "A": 0.03,
    "B+": 0.07,
    "B": 0.10,
    "C+": 0.18,
    "C": 0.25,
    "D+": 0.40,
    "D": 0.50,
    "E": 1.00,
}

STRICT_LITERATURE_WINDOWS = {
    "cooper_2018_515": {"5s2 1S0", "5s5p 3P1"},
    "safronova_2013_813": {"5s5p 3P0"},
}

LITERATURE_DATASET_PRIORITY = (
    "safronova_2013_813",
    "cooper_2018_515",
)

TARGET_STATES = ("5s2 1S0", "5s5p 3P0", "5s5p 3P1")


@dataclass(frozen=True)
class Level:
    state_label: str
    configuration: str
    term: str
    J: int
    parity: str
    energy_cm: float
    uncertainty_cm: float | None
    reference: str


@dataclass(frozen=True)
class SpectralLine:
    lower_label: str
    upper_label: str
    lower_parity: str
    upper_parity: str
    wavelength_nm: float | None
    ritz_wavelength_nm: float | None
    transition_wavenumber_cm: float
    Aki_s: float | None
    accuracy: str
    lower_energy_cm: float
    upper_energy_cm: float
    lower_J: int
    upper_J: int
    transition_type: str
    tp_reference: str
    line_reference: str


@dataclass(frozen=True)
class LiteratureRDME:
    dataset: str
    initial_label: str
    partner_label: str
    reduced_dipole_au: float
    reduced_dipole_unc_au: float | None
    source: str
    note: str


@dataclass(frozen=True)
class ResidualContribution:
    dataset: str
    state_label: str
    component: str
    window_min_nm: float
    window_max_nm: float
    value_au: float
    uncertainty_au: float | None
    enabled_by_default: bool
    source: str
    note: str

    def applies_to(self, wavelength_nm: float, include_optional_residuals: bool) -> bool:
        if not self.enabled_by_default and not include_optional_residuals:
            return False
        return self.window_min_nm <= wavelength_nm <= self.window_max_nm


@dataclass(frozen=True)
class ChannelCandidate:
    initial_level: Level
    partner_level: Level
    delta_energy_cm: float
    line: SpectralLine | None
    literature_entries: tuple[LiteratureRDME, ...]


@dataclass(frozen=True)
class Channel:
    initial_label: str
    partner_label: str
    initial_J: int
    partner_J: int
    delta_energy_cm: float
    reduced_dipole_au: float
    reduced_dipole_unc_au: float | None
    source_kind: str
    source: str
    note: str


@dataclass(frozen=True)
class PolarizabilityComponents:
    scalar: float
    vector: float
    tensor: float


@dataclass(frozen=True)
class LightGeometry:
    name: str
    polarization_vector: np.ndarray

    def normalized(self) -> np.ndarray:
        vec = np.asarray(self.polarization_vector, dtype=np.complex128)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Polarization vector must be nonzero.")
        return vec / norm

    @classmethod
    def linear_along_z(cls) -> "LightGeometry":
        return cls("linear_z", np.array([0.0, 0.0, 1.0], dtype=np.complex128))

    @classmethod
    def linear_along_x(cls) -> "LightGeometry":
        return cls("linear_x", np.array([1.0, 0.0, 0.0], dtype=np.complex128))

    @classmethod
    def linear_at_angle_xz(cls, theta_deg: float) -> "LightGeometry":
        theta = np.deg2rad(theta_deg)
        return cls(
            f"linear_xz_{theta_deg:g}deg",
            np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.complex128),
        )

    @classmethod
    def elliptical_xy(cls, gamma_deg: float) -> "LightGeometry":
        gamma = np.deg2rad(gamma_deg)
        return cls(
            f"elliptical_xy_{gamma_deg:g}deg",
            np.array([math.cos(gamma), 1j * math.sin(gamma), 0.0], dtype=np.complex128),
        )


@dataclass(frozen=True)
class ModelConfig:
    max_energy_cm: float = 70000.0
    include_optional_residuals: bool = False
    monte_carlo_samples: int = 200


def accuracy_code_fraction(code: str) -> float | None:
    normalized = code.replace("'", "").strip().upper()
    if not normalized:
        return None
    return ACCURACY_CODE_TO_FRACTION.get(normalized)


def reduced_dipole_au_from_aki(wavenumber_cm: float, aki_s: float, upper_j: int) -> float:
    omega_si = 2.0 * pi * c * (wavenumber_cm * 100.0)
    d2_si = 3.0 * pi * epsilon_0 * hbar * c**3 * (2 * upper_j + 1) * aki_s / omega_si**3
    return math.sqrt(d2_si / EA0_SI**2)


def omega_from_wavelength_au(wavelength_nm: float | np.ndarray) -> float | np.ndarray:
    return (2.0 * pi * c / (np.asarray(wavelength_nm) * 1e-9)) * AU_TIME_S


def omega_from_wavenumber_au(wavenumber_cm: float) -> float:
    return (2.0 * pi * c * (wavenumber_cm * 100.0)) * AU_TIME_S


def parse_numeric(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_levels(path: Path) -> dict[str, Level]:
    df = pd.read_csv(path)
    levels: dict[str, Level] = {}
    for row in df.to_dict(orient="records"):
        j_value = parse_numeric(row["J"])
        if j_value is None:
            continue
        level = Level(
            state_label=parse_text(row["state_label"]),
            configuration=parse_text(row["configuration"]),
            term=parse_text(row["term"]),
            J=int(j_value),
            parity=parse_text(row["parity"]),
            energy_cm=float(row["energy_cm"]),
            uncertainty_cm=parse_numeric(row["uncertainty_cm"]),
            reference=parse_text(row["reference"]),
        )
        levels[level.state_label] = level
    return levels


def load_lines(path: Path) -> list[SpectralLine]:
    df = pd.read_csv(path)
    numeric = pd.to_numeric(df["transition_wavenumber_cm"], errors="coerce")
    df = df[numeric.notna()].copy()
    df = df[~df["lower_label"].astype(str).str.contains("conf_i", regex=False)]
    lines: list[SpectralLine] = []
    for row in df.to_dict(orient="records"):
        lines.append(
            SpectralLine(
                lower_label=parse_text(row["lower_label"]),
                upper_label=parse_text(row["upper_label"]),
                lower_parity=parse_text(row["lower_parity"]),
                upper_parity=parse_text(row["upper_parity"]),
                wavelength_nm=parse_numeric(row["wavelength_nm"]),
                ritz_wavelength_nm=parse_numeric(row["ritz_wavelength_nm"]),
                transition_wavenumber_cm=float(row["transition_wavenumber_cm"]),
                Aki_s=parse_numeric(row["Aki_s"]),
                accuracy=parse_text(row["accuracy"]),
                lower_energy_cm=float(row["lower_energy_cm"]),
                upper_energy_cm=float(row["upper_energy_cm"]),
                lower_J=int(row["lower_J"]),
                upper_J=int(row["upper_J"]),
                transition_type=parse_text(row["transition_type"]),
                tp_reference=parse_text(row["tp_reference"]),
                line_reference=parse_text(row["line_reference"]),
            )
        )
    return lines


def load_literature_matrix_elements(path: Path) -> dict[frozenset[str], list[LiteratureRDME]]:
    df = pd.read_csv(path)
    mapping: dict[frozenset[str], list[LiteratureRDME]] = {}
    for row in df.to_dict(orient="records"):
        entry = LiteratureRDME(
            dataset=parse_text(row["dataset"]),
            initial_label=parse_text(row["initial_label"]),
            partner_label=parse_text(row["partner_label"]),
            reduced_dipole_au=float(row["reduced_dipole_au"]),
            reduced_dipole_unc_au=parse_numeric(row["reduced_dipole_unc_au"]),
            source=parse_text(row["source"]),
            note=parse_text(row["note"]),
        )
        mapping.setdefault(frozenset({entry.initial_label, entry.partner_label}), []).append(entry)
    return mapping


def load_residuals(path: Path) -> list[ResidualContribution]:
    df = pd.read_csv(path)
    residuals: list[ResidualContribution] = []
    for row in df.to_dict(orient="records"):
        residuals.append(
            ResidualContribution(
                dataset=parse_text(row["dataset"]),
                state_label=parse_text(row["state_label"]),
                component=parse_text(row["component"]),
                window_min_nm=float(row["window_min_nm"]),
                window_max_nm=float(row["window_max_nm"]),
                value_au=float(row["value_au"]),
                uncertainty_au=parse_numeric(row["uncertainty_au"]),
                enabled_by_default=bool(int(row["enabled_by_default"])),
                source=parse_text(row["source"]),
                note=parse_text(row["note"]),
            )
        )
    return residuals


def load_benchmarks(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def line_is_e1_candidate(line: SpectralLine) -> bool:
    if line.lower_parity == line.upper_parity:
        return False
    delta_j = abs(line.upper_J - line.lower_J)
    if delta_j > 1:
        return False
    if line.upper_J == 0 and line.lower_J == 0:
        return False
    return True


def build_line_index(lines: Iterable[SpectralLine]) -> dict[frozenset[str], list[SpectralLine]]:
    index: dict[frozenset[str], list[SpectralLine]] = {}
    for line in lines:
        if not line_is_e1_candidate(line):
            continue
        key = frozenset({line.lower_label, line.upper_label})
        index.setdefault(key, []).append(line)
    return index


def allowed_e1_partners(initial_level: Level, levels: Iterable[Level], max_energy_cm: float) -> list[Level]:
    partners: list[Level] = []
    for partner in levels:
        if partner.state_label == initial_level.state_label:
            continue
        if partner.parity == initial_level.parity:
            continue
        delta_j = abs(partner.J - initial_level.J)
        if delta_j > 1:
            continue
        if partner.J == 0 and initial_level.J == 0:
            continue
        if abs(partner.energy_cm - initial_level.energy_cm) > max_energy_cm:
            continue
        partners.append(partner)
    partners.sort(key=lambda item: item.energy_cm)
    return partners


def choose_best_line(lines: list[SpectralLine] | None) -> SpectralLine | None:
    if not lines:
        return None
    with_aki = [line for line in lines if line.Aki_s is not None]
    if with_aki:
        return sorted(with_aki, key=lambda item: (item.wavelength_nm is None, item.wavelength_nm or 0.0))[0]
    return sorted(lines, key=lambda item: (item.wavelength_nm is None, item.wavelength_nm or 0.0))[0]


def build_channel_catalog(
    initial_label: str,
    levels: dict[str, Level],
    line_index: dict[frozenset[str], list[SpectralLine]],
    literature_index: dict[frozenset[str], list[LiteratureRDME]],
    config: ModelConfig,
) -> list[ChannelCandidate]:
    initial_level = levels[initial_label]
    catalog: list[ChannelCandidate] = []
    for partner in allowed_e1_partners(initial_level, levels.values(), config.max_energy_cm):
        key = frozenset({initial_label, partner.state_label})
        line = choose_best_line(line_index.get(key))
        literature_entries = tuple(literature_index.get(key, []))
        catalog.append(
            ChannelCandidate(
                initial_level=initial_level,
                partner_level=partner,
                delta_energy_cm=partner.energy_cm - initial_level.energy_cm,
                line=line,
                literature_entries=literature_entries,
            )
        )
    return catalog


def preferred_dataset(state_label: str, wavelength_nm: float) -> str | None:
    if 495.0 <= wavelength_nm <= 535.0 and state_label in {"5s2 1S0", "5s5p 3P1"}:
        return "cooper_2018_515"
    if 800.0 <= wavelength_nm <= 1070.0 and state_label == "5s5p 3P0":
        return "safronova_2013_813"
    return None


def choose_literature_entry(
    entries: tuple[LiteratureRDME, ...],
    dataset: str | None = None,
) -> LiteratureRDME | None:
    if dataset:
        for entry in entries:
            if entry.dataset == dataset:
                return entry
    for preferred in LITERATURE_DATASET_PRIORITY:
        for entry in entries:
            if entry.dataset == preferred:
                return entry
    return entries[0] if entries else None


def select_channel(
    candidate: ChannelCandidate,
    dataset: str | None,
    strict_literature: bool,
) -> Channel | None:
    entry = choose_literature_entry(candidate.literature_entries, dataset=dataset)
    if entry is not None and (dataset is None or entry.dataset == dataset or not strict_literature):
        return Channel(
            initial_label=candidate.initial_level.state_label,
            partner_label=candidate.partner_level.state_label,
            initial_J=candidate.initial_level.J,
            partner_J=candidate.partner_level.J,
            delta_energy_cm=candidate.delta_energy_cm,
            reduced_dipole_au=entry.reduced_dipole_au,
            reduced_dipole_unc_au=entry.reduced_dipole_unc_au,
            source_kind="literature_rdme",
            source=entry.source,
            note=entry.note,
        )
    if strict_literature:
        return None
    line = candidate.line
    if line is None or line.Aki_s is None:
        return None
    upper_j = candidate.partner_level.J if candidate.delta_energy_cm > 0 else candidate.initial_level.J
    reduced_dipole_au = reduced_dipole_au_from_aki(abs(candidate.delta_energy_cm), line.Aki_s, upper_j)
    acc_fraction = accuracy_code_fraction(line.accuracy)
    reduced_dipole_unc = None
    if acc_fraction is not None:
        reduced_dipole_unc = 0.5 * acc_fraction * reduced_dipole_au
    note = f"NIST Aki={line.Aki_s:g} s^-1, accuracy={line.accuracy or 'unknown'}"
    return Channel(
        initial_label=candidate.initial_level.state_label,
        partner_label=candidate.partner_level.state_label,
        initial_J=candidate.initial_level.J,
        partner_J=candidate.partner_level.J,
        delta_energy_cm=candidate.delta_energy_cm,
        reduced_dipole_au=reduced_dipole_au,
        reduced_dipole_unc_au=reduced_dipole_unc,
        source_kind="line_aki",
        source="NIST ASD / Sansonetti-Nave 2010",
        note=note,
    )


def selected_channels_for_state(
    state_label: str,
    wavelength_nm: float,
    catalog: list[ChannelCandidate],
) -> list[Channel]:
    dataset = preferred_dataset(state_label, wavelength_nm)
    strict = bool(dataset and state_label in STRICT_LITERATURE_WINDOWS.get(dataset, set()))
    channels: list[Channel] = []
    for candidate in catalog:
        channel = select_channel(candidate, dataset, strict)
        if channel is not None:
            channels.append(channel)
    return channels


def residuals_for_state(
    residuals: Iterable[ResidualContribution],
    state_label: str,
    wavelength_nm: float,
    include_optional_residuals: bool,
) -> list[ResidualContribution]:
    selected = []
    for residual in residuals:
        if residual.state_label != state_label:
            continue
        if residual.applies_to(wavelength_nm, include_optional_residuals):
            selected.append(residual)
    return selected


def scalar_channel_contribution(channel: Channel, wavelength_nm: float) -> float:
    omega = omega_from_wavelength_au(wavelength_nm)
    delta = omega_from_wavenumber_au(channel.delta_energy_cm)
    return (
        2.0
        / (3.0 * (2 * channel.initial_J + 1))
        * delta
        * channel.reduced_dipole_au**2
        / (delta**2 - omega**2)
    )


@lru_cache(maxsize=None)
def vector_angular_factor(initial_j: int, partner_j: int) -> float:
    J = initial_j
    return float(
        (-1) ** (J + partner_j + 1)
        * math.sqrt(6.0 * J / ((J + 1) * (2 * J + 1)))
        * float(wigner_6j(1, 1, 1, J, J, partner_j))
    )


def vector_channel_contribution(channel: Channel, wavelength_nm: float) -> float:
    """Rank-1 contribution in the Madjarov / Cooper convention.

    For the irreducible tensor decomposition used in Madjarov thesis Eq. (2.25),
    the dynamic vector polarizability carries an overall ``2 * omega`` numerator,
    equivalently ``1 / (Delta - omega) - 1 / (Delta + omega)``.
    This keeps our alpha^(1) convention aligned with the Sr1Pol reference tables
    and with the effective magnetic-field form used in Cooper Appendix B.
    """
    J = channel.initial_J
    if J == 0:
        return 0.0
    omega = omega_from_wavelength_au(wavelength_nm)
    delta = omega_from_wavenumber_au(channel.delta_energy_cm)
    angular = vector_angular_factor(J, channel.partner_J)
    return 2.0 * angular * omega * channel.reduced_dipole_au**2 / (delta**2 - omega**2)


@lru_cache(maxsize=None)
def tensor_angular_factor(initial_j: int, partner_j: int) -> float:
    J = initial_j
    prefactor = math.sqrt(5.0 * J * (2 * J - 1) / (6.0 * (J + 1) * (2 * J + 1) * (2 * J + 3)))
    return float(
        4.0
        * prefactor
        * ((-1) ** (J + partner_j))
        * float(wigner_6j(J, 1, partner_j, 1, J, 2))
    )


def tensor_channel_contribution(channel: Channel, wavelength_nm: float) -> float:
    J = channel.initial_J
    if J < 1:
        return 0.0
    omega = omega_from_wavelength_au(wavelength_nm)
    delta = omega_from_wavenumber_au(channel.delta_energy_cm)
    angular = tensor_angular_factor(J, channel.partner_J)
    return angular * delta * channel.reduced_dipole_au**2 / (delta**2 - omega**2)


def polarizability_components(
    state_label: str,
    wavelength_nm: float,
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
) -> PolarizabilityComponents:
    channels = selected_channels_for_state(state_label, wavelength_nm, catalog_by_state[state_label])
    scalar = sum(scalar_channel_contribution(channel, wavelength_nm) for channel in channels)
    vector = sum(vector_channel_contribution(channel, wavelength_nm) for channel in channels)
    tensor = sum(tensor_channel_contribution(channel, wavelength_nm) for channel in channels)
    for residual in residuals_for_state(
        residuals,
        state_label,
        wavelength_nm,
        include_optional_residuals=config.include_optional_residuals,
    ):
        if residual.component == "scalar":
            scalar += residual.value_au
        elif residual.component == "vector":
            vector += residual.value_au
        elif residual.component == "tensor":
            tensor += residual.value_au
    return PolarizabilityComponents(scalar=scalar, vector=vector, tensor=tensor)


def effective_polarizability(
    J: int,
    mJ: int,
    components: PolarizabilityComponents,
    geometry: LightGeometry,
) -> float:
    """Return alpha_eff in the same convention used by Young Eq. (2.24).

    The vector piece multiplies Re[i (epsilon x epsilon*)_z]. This convention is
    kept consistent with stark_hamiltonian_j1().
    """
    evec = geometry.normalized()
    alpha = components.scalar
    if J > 0:
        circ = float(np.real((1j * np.cross(evec, np.conjugate(evec)))[2]))
        alpha += components.vector * circ * (mJ / J)
    if J >= 1:
        tensor_geom = (3.0 * abs(evec[2]) ** 2 - 1.0) / 2.0
        tensor_state = (3.0 * mJ**2 - J * (J + 1)) / (J * (2 * J - 1))
        alpha += components.tensor * tensor_geom * tensor_state
    return float(np.real_if_close(alpha))


def spin_one_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Jx = (1 / math.sqrt(2)) * np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        dtype=np.complex128,
    )
    Jy = (1 / math.sqrt(2)) * np.array(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]],
        dtype=np.complex128,
    )
    Jz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
    return Jx, Jy, Jz


def stark_hamiltonian_j1(
    components: PolarizabilityComponents,
    geometry: LightGeometry,
) -> np.ndarray:
    """AC Stark Hamiltonian for J=1 in the same sign/prefactor convention as alpha_eff.

    For linear polarization along the quantization axis this reduces to eigenvalues
    -(alpha_s + alpha_t), -(alpha_s - 2 alpha_t), -(alpha_s + alpha_t), matching
    Cooper Appendix B and the m_J-resolved effective polarizabilities.
    """
    evec = geometry.normalized()
    Jx, Jy, Jz = spin_one_matrices()
    Jvec = np.array([Jx, Jy, Jz])
    ident = np.eye(3, dtype=np.complex128)
    vector_term = -(components.vector / 1.0) * np.tensordot(
        1j * np.cross(evec, np.conjugate(evec)), Jvec, axes=(0, 0)
    )
    edotJ = evec[0] * Jx + evec[1] * Jy + evec[2] * Jz
    ecdotJ = np.conjugate(evec[0]) * Jx + np.conjugate(evec[1]) * Jy + np.conjugate(evec[2]) * Jz
    tensor_operator = 0.5 * (edotJ @ ecdotJ + ecdotJ @ edotJ) - (2.0 / 3.0) * ident
    tensor_term = -3.0 * components.tensor * tensor_operator
    return -components.scalar * ident + vector_term + tensor_term


def lande_g_from_level(level: Level) -> float | None:
    match = re.fullmatch(r"(\d+)([SPDFGHIK])", level.term)
    if not match:
        return None
    multiplicity = int(match.group(1))
    L_symbol = match.group(2)
    L_map = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6, "K": 7}
    L = L_map[L_symbol]
    S = (multiplicity - 1) / 2.0
    J = float(level.J)
    if J == 0:
        return 0.0
    return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2.0 * J * (J + 1))


def zeeman_hamiltonian_hz(level: Level, magnetic_field_t: np.ndarray) -> np.ndarray:
    if level.J != 1:
        raise ValueError("This helper currently supports only J=1 manifolds.")
    g_j = lande_g_from_level(level)
    if g_j is None:
        raise ValueError(f"Cannot infer Landé g-factor for {level.state_label}.")
    Jx, Jy, Jz = spin_one_matrices()
    Jvec = np.array([Jx, Jy, Jz])
    return MU_B_OVER_H_HZ_PER_T * g_j * np.tensordot(magnetic_field_t, Jvec, axes=(0, 0))


def channel_contribution_breakdown(
    state_label: str,
    wavelength_nm: float,
    catalog_by_state: dict[str, list[ChannelCandidate]],
) -> list[dict]:
    channels = selected_channels_for_state(state_label, wavelength_nm, catalog_by_state[state_label])
    rows = []
    for channel in channels:
        rows.append(
            {
                "partner_label": channel.partner_label,
                "source_kind": channel.source_kind,
                "source": channel.source,
                "delta_energy_cm": channel.delta_energy_cm,
                "reduced_dipole_au": channel.reduced_dipole_au,
                "scalar": scalar_channel_contribution(channel, wavelength_nm),
                "vector": vector_channel_contribution(channel, wavelength_nm),
                "tensor": tensor_channel_contribution(channel, wavelength_nm),
            }
        )
    rows.sort(key=lambda item: abs(item["scalar"]) + abs(item["vector"]) + abs(item["tensor"]), reverse=True)
    return rows


def convergence_series(
    state_label: str,
    wavelength_nm: float,
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
) -> list[dict]:
    thresholds = [15000, 18000, 22000, 26000, 30000, 34000, 38000, 42000, 46000]
    channels = selected_channels_for_state(state_label, wavelength_nm, catalog_by_state[state_label])
    rows = []
    residual_subset = residuals_for_state(
        residuals,
        state_label,
        wavelength_nm,
        include_optional_residuals=config.include_optional_residuals,
    )
    residual_scalar = sum(item.value_au for item in residual_subset if item.component == "scalar")
    residual_vector = sum(item.value_au for item in residual_subset if item.component == "vector")
    residual_tensor = sum(item.value_au for item in residual_subset if item.component == "tensor")
    for threshold in thresholds:
        subset = [channel for channel in channels if abs(channel.delta_energy_cm) <= threshold]
        scalar = sum(scalar_channel_contribution(channel, wavelength_nm) for channel in subset)
        vector = sum(vector_channel_contribution(channel, wavelength_nm) for channel in subset)
        tensor = sum(tensor_channel_contribution(channel, wavelength_nm) for channel in subset)
        rows.append(
            {
                "cutoff_cm": threshold,
                "known_scalar": scalar,
                "known_vector": vector,
                "known_tensor": tensor,
                "residual_scalar": residual_scalar,
                "residual_vector": residual_vector,
                "residual_tensor": residual_tensor,
                "total_scalar": scalar + residual_scalar,
                "total_vector": vector + residual_vector,
                "total_tensor": tensor + residual_tensor,
                "channels_included": len(subset),
            }
        )
    return rows


def find_crossings(scan_x: np.ndarray, scan_y: np.ndarray) -> list[tuple[float, float]]:
    roots: list[tuple[float, float]] = []
    for idx in np.where(np.sign(scan_y[:-1]) != np.sign(scan_y[1:]))[0]:
        x0, x1 = float(scan_x[idx]), float(scan_x[idx + 1])
        y0, y1 = float(scan_y[idx]), float(scan_y[idx + 1])
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        slope = (y1 - y0) / (x1 - x0)
        roots.append((root, slope))
    return roots


def magic_wavelengths(
    left_fn,
    right_fn,
    lower_nm: float,
    upper_nm: float,
    samples: int = 1500,
) -> list[float]:
    grid = np.linspace(lower_nm, upper_nm, samples)
    diff = np.array([left_fn(x) - right_fn(x) for x in grid], dtype=float)
    roots = []
    for idx in np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]:
        a, b = float(grid[idx]), float(grid[idx + 1])
        try:
            roots.append(float(brentq(lambda x: left_fn(x) - right_fn(x), a, b)))
        except ValueError:
            continue
    return roots


def state_components_function(
    state_label: str,
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
):
    return lambda wavelength_nm: polarizability_components(
        state_label=state_label,
        wavelength_nm=wavelength_nm,
        catalog_by_state=catalog_by_state,
        residuals=residuals,
        config=config,
    )


def state_effective_function(
    state_label: str,
    mJ: int,
    geometry: LightGeometry,
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
):
    J = catalog_by_state[state_label][0].initial_level.J if catalog_by_state[state_label] else 0
    return lambda wavelength_nm: effective_polarizability(
        J=J,
        mJ=mJ,
        components=polarizability_components(
            state_label=state_label,
            wavelength_nm=wavelength_nm,
            catalog_by_state=catalog_by_state,
            residuals=residuals,
            config=config,
        ),
        geometry=geometry,
    )


def write_missing_channel_report(
    catalog_by_state: dict[str, list[ChannelCandidate]],
    config: ModelConfig,
) -> pd.DataFrame:
    rows = []
    for state_label, catalog in catalog_by_state.items():
        for candidate in catalog:
            literature_datasets = ",".join(sorted({item.dataset for item in candidate.literature_entries}))
            rows.append(
                {
                    "state_label": state_label,
                    "partner_label": candidate.partner_level.state_label,
                    "delta_energy_cm": candidate.delta_energy_cm,
                    "partner_J": candidate.partner_level.J,
                    "line_exists": candidate.line is not None,
                    "line_has_Aki": candidate.line is not None and candidate.line.Aki_s is not None,
                    "literature_datasets": literature_datasets,
                    "status": (
                        "literature"
                        if literature_datasets
                        else "line_aki"
                        if candidate.line is not None and candidate.line.Aki_s is not None
                        else "missing"
                    ),
                }
            )
    df = pd.DataFrame(rows).sort_values(["state_label", "status", "delta_energy_cm"])
    df.to_csv(OUTPUT_DIR / "missing_channel_report_sr1.csv", index=False)
    return df


def write_plots(
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
) -> None:
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    geometry_z = LightGeometry.linear_along_z()

    alpha_1s0 = state_effective_function("5s2 1S0", 0, geometry_z, catalog_by_state, residuals, config)
    alpha_3p1_m0 = state_effective_function("5s5p 3P1", 0, geometry_z, catalog_by_state, residuals, config)
    alpha_3p1_m1 = state_effective_function("5s5p 3P1", 1, geometry_z, catalog_by_state, residuals, config)
    alpha_3p0 = state_effective_function("5s5p 3P0", 0, geometry_z, catalog_by_state, residuals, config)

    wl_515 = np.linspace(495.0, 535.0, 1201)
    y_1s0 = np.array([alpha_1s0(wl) for wl in wl_515])
    y_3p1_m0 = np.array([alpha_3p1_m0(wl) for wl in wl_515])
    y_3p1_m1 = np.array([alpha_3p1_m1(wl) for wl in wl_515])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_515, y_1s0, label=r"$^1S_0$")
    ax.plot(wl_515, y_3p1_m0, label=r"$^3P_1(m_J=0)$")
    ax.plot(wl_515, y_3p1_m1, label=r"$^3P_1(|m_J|=1)$")
    for root, _ in find_crossings(wl_515, y_1s0 - y_3p1_m0):
        ax.axvline(root, color="0.55", linestyle="--", linewidth=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Effective polarizability ($a_0^3$)")
    ax.set_title("Sr 515 nm effective polarizability (research model)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sr_515_effective_research.png", dpi=220)
    plt.close(fig)

    wl_813 = np.linspace(780.0, 950.0, 1401)
    y_1s0_813 = np.array([alpha_1s0(wl) for wl in wl_813])
    y_3p0_813 = np.array([alpha_3p0(wl) for wl in wl_813])
    y_3p1_813 = np.array([alpha_3p1_m1(wl) for wl in wl_813])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_813, y_1s0_813, label=r"$^1S_0$")
    ax.plot(wl_813, y_3p0_813, label=r"$^3P_0$")
    ax.plot(wl_813, y_3p1_813, label=r"$^3P_1(|m_J|=1)$")
    for root, _ in find_crossings(wl_813, y_1s0_813 - y_3p0_813):
        ax.axvline(root, color="0.55", linestyle="--", linewidth=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Effective polarizability ($a_0^3$)")
    ax.set_title("Sr 813/914 nm effective polarizability (research model)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sr_813_effective_research.png", dpi=220)
    plt.close(fig)

    comp_fn = state_components_function("5s5p 3P1", catalog_by_state, residuals, config)
    alpha0 = np.array([comp_fn(wl).scalar for wl in wl_515])
    alpha1 = np.array([comp_fn(wl).vector for wl in wl_515])
    alpha2 = np.array([comp_fn(wl).tensor for wl in wl_515])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wl_515, alpha0, label=r"$\alpha^{(0)}$")
    ax.plot(wl_515, alpha1, label=r"$\alpha^{(1)}$")
    ax.plot(wl_515, alpha2, label=r"$\alpha^{(2)}$")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Polarizability component ($a_0^3$)")
    ax.set_title(r"$^3P_1$ tensor decomposition near 515 nm")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sr_3p1_components_515.png", dpi=220)
    plt.close(fig)


def monte_carlo_summary(
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    config: ModelConfig,
) -> dict:
    rng = np.random.default_rng(42)
    geometry_z = LightGeometry.linear_along_z()
    targets = [
        ("5s2 1S0", 0, 515.2),
        ("5s5p 3P1", 0, 515.2),
        ("5s5p 3P1", 1, 515.2),
    ]

    sampled_reference_wavelength = {
        "5s2 1S0": 515.2,
        "5s5p 3P1": 515.2,
        "5s5p 3P0": 813.4,
    }
    base_channels = {
        state: selected_channels_for_state(state, sampled_reference_wavelength[state], catalog_by_state[state])
        for state in sampled_reference_wavelength
    }
    base_residuals = {
        state: residuals_for_state(
            residuals,
            state,
            sampled_reference_wavelength[state],
            include_optional_residuals=config.include_optional_residuals,
        )
        for state in sampled_reference_wavelength
    }

    sampled_values: dict[str, list[float]] = {}
    for state, mJ, wl in targets:
        key = f"{state}_m{mJ}_{wl}"
        sampled_values[key] = []

    sampled_magic_1 = []
    sampled_magic_2 = []

    for _ in range(config.monte_carlo_samples):
        sampled_channel_strengths: dict[tuple[str, str, str, str], float] = {}
        sampled_residual_values: dict[tuple[str, str, str], float] = {}

        for state, channels in base_channels.items():
            for channel in channels:
                sigma = channel.reduced_dipole_unc_au or 0.0
                sampled = rng.normal(channel.reduced_dipole_au, sigma) if sigma > 0 else channel.reduced_dipole_au
                sampled_channel_strengths[
                    (state, channel.partner_label, channel.source_kind, channel.source)
                ] = max(sampled, 0.0)
            for residual in base_residuals[state]:
                sigma = residual.uncertainty_au or 0.0
                sampled = rng.normal(residual.value_au, sigma) if sigma > 0 else residual.value_au
                sampled_residual_values[(state, residual.component, residual.source)] = sampled

        def sampled_components(state_label: str, wavelength_nm: float) -> PolarizabilityComponents:
            channels = base_channels[state_label]
            scalar = 0.0
            vector = 0.0
            tensor = 0.0
            for channel in channels:
                sampled_d = sampled_channel_strengths[
                    (state_label, channel.partner_label, channel.source_kind, channel.source)
                ]
                sampled_channel = Channel(
                    initial_label=channel.initial_label,
                    partner_label=channel.partner_label,
                    initial_J=channel.initial_J,
                    partner_J=channel.partner_J,
                    delta_energy_cm=channel.delta_energy_cm,
                    reduced_dipole_au=sampled_d,
                    reduced_dipole_unc_au=channel.reduced_dipole_unc_au,
                    source_kind=channel.source_kind,
                    source=channel.source,
                    note=channel.note,
                )
                scalar += scalar_channel_contribution(sampled_channel, wavelength_nm)
                vector += vector_channel_contribution(sampled_channel, wavelength_nm)
                tensor += tensor_channel_contribution(sampled_channel, wavelength_nm)
            for residual in base_residuals[state_label]:
                sampled = sampled_residual_values[(state_label, residual.component, residual.source)]
                if residual.component == "scalar":
                    scalar += sampled
                elif residual.component == "vector":
                    vector += sampled
                elif residual.component == "tensor":
                    tensor += sampled
            return PolarizabilityComponents(scalar=scalar, vector=vector, tensor=tensor)

        def eff(state_label: str, mJ: int, wavelength_nm: float) -> float:
            J = catalog_by_state[state_label][0].initial_level.J if catalog_by_state[state_label] else 0
            return effective_polarizability(J, mJ, sampled_components(state_label, wavelength_nm), geometry_z)

        for state, mJ, wl in targets:
            sampled_values[f"{state}_m{mJ}_{wl}"].append(eff(state, mJ, wl))

        try:
            sampled_magic_1.append(
                brentq(
                    lambda wl: eff("5s2 1S0", 0, wl) - eff("5s5p 3P1", 0, wl),
                    495.0,
                    505.0,
                )
            )
        except ValueError:
            pass
        try:
            sampled_magic_2.append(
                brentq(
                    lambda wl: eff("5s2 1S0", 0, wl) - eff("5s5p 3P1", 0, wl),
                    515.5,
                    525.0,
                )
            )
        except ValueError:
            pass

    summary = {
        "samples": config.monte_carlo_samples,
        "values": {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for key, values in sampled_values.items()
        },
        "magic_wavelength_500nm_branch": {
            "mean": float(np.mean(sampled_magic_1)) if sampled_magic_1 else None,
            "std": float(np.std(sampled_magic_1, ddof=1)) if len(sampled_magic_1) > 1 else None,
        },
        "magic_wavelength_520nm_branch": {
            "mean": float(np.mean(sampled_magic_2)) if sampled_magic_2 else None,
            "std": float(np.std(sampled_magic_2, ddof=1)) if len(sampled_magic_2) > 1 else None,
        },
    }
    return summary


def write_validation_report(
    levels: dict[str, Level],
    catalog_by_state: dict[str, list[ChannelCandidate]],
    residuals: list[ResidualContribution],
    benchmarks: pd.DataFrame,
    config: ModelConfig,
) -> None:
    geometry_z = LightGeometry.linear_along_z()
    alpha_1s0 = state_effective_function("5s2 1S0", 0, geometry_z, catalog_by_state, residuals, config)
    alpha_3p0 = state_effective_function("5s5p 3P0", 0, geometry_z, catalog_by_state, residuals, config)
    alpha_3p1_m0 = state_effective_function("5s5p 3P1", 0, geometry_z, catalog_by_state, residuals, config)
    alpha_3p1_m1 = state_effective_function("5s5p 3P1", 1, geometry_z, catalog_by_state, residuals, config)
    comp_3p1_515 = polarizability_components("5s5p 3P1", 515.2, catalog_by_state, residuals, config)
    comp_1s0_515 = polarizability_components("5s2 1S0", 515.2, catalog_by_state, residuals, config)
    q_value = (comp_1s0_515.scalar - (comp_3p1_515.scalar + comp_3p1_515.tensor)) / (
        comp_1s0_515.scalar - (comp_3p1_515.scalar - 2 * comp_3p1_515.tensor)
    )
    vector_tensor_ratio = abs(comp_3p1_515.vector / comp_3p1_515.tensor) if comp_3p1_515.tensor else None

    benchmark_results = []
    for row in benchmarks.to_dict(orient="records"):
        state_label = row["state_label"]
        wl = float(row["wavelength_nm"])
        component = row["component"]
        mJ = parse_numeric(row["mJ"])
        if component == "scalar":
            value = polarizability_components(state_label, wl, catalog_by_state, residuals, config).scalar
        elif component == "tensor":
            value = polarizability_components(state_label, wl, catalog_by_state, residuals, config).tensor
        elif component == "vector_tensor_ratio":
            value = vector_tensor_ratio
        elif component == "Q":
            value = q_value
        elif component == "effective":
            geometry = geometry_z
            value = state_effective_function(
                state_label,
                int(mJ),
                geometry,
                catalog_by_state,
                residuals,
                config,
            )(wl)
        else:
            continue
        benchmark_results.append(
            {
                "source": row["source"],
                "state_label": state_label,
                "wavelength_nm": wl,
                "component": component,
                "mJ": None if mJ is None else int(mJ),
                "computed": round(float(value), 6),
                "target": round(float(row["target_au"]), 6),
                "abs_error": round(float(abs(value - float(row["target_au"]))), 6),
                "note": row["note"],
            }
        )

    missing_df = write_missing_channel_report(catalog_by_state, config)
    coverage_summary = {}
    for state_label in TARGET_STATES:
        subset = missing_df[missing_df["state_label"] == state_label]
        coverage_summary[state_label] = {
            "candidate_count": int(len(subset)),
            "literature_count": int((subset["status"] == "literature").sum()),
            "line_aki_count": int((subset["status"] == "line_aki").sum()),
            "missing_count": int((subset["status"] == "missing").sum()),
        }

    breakdown_1s0 = channel_contribution_breakdown("5s2 1S0", 515.2, catalog_by_state)[:10]
    breakdown_3p1 = channel_contribution_breakdown("5s5p 3P1", 515.2, catalog_by_state)[:10]
    convergence = {
        "1S0_515.2": convergence_series("5s2 1S0", 515.2, catalog_by_state, residuals, config),
        "3P1_515.2": convergence_series("5s5p 3P1", 515.2, catalog_by_state, residuals, config),
        "3P0_813.4": convergence_series("5s5p 3P0", 813.4, catalog_by_state, residuals, config),
    }

    magic_results = {
        "1S0_vs_3P1_m0_495_535": magic_wavelengths(alpha_1s0, alpha_3p1_m0, 495.0, 535.0),
        "1S0_vs_3P0_780_850": magic_wavelengths(alpha_1s0, alpha_3p0, 780.0, 850.0),
        "1S0_vs_3P1_m1_860_940": magic_wavelengths(alpha_1s0, alpha_3p1_m1, 860.0, 940.0),
    }

    geometry_metrics = {
        "3P1_515.2_components": {
            "scalar": comp_3p1_515.scalar,
            "vector": comp_3p1_515.vector,
            "tensor": comp_3p1_515.tensor,
        },
        "1S0_515.2_scalar": comp_1s0_515.scalar,
        "Q_value_515.2": q_value,
        "vector_tensor_ratio_515.2": vector_tensor_ratio,
        "stark_matrix_j1_linear_z_515.2": np.real_if_close(stark_hamiltonian_j1(comp_3p1_515, geometry_z)).tolist(),
        "lande_g_3P1": lande_g_from_level(levels["5s5p 3P1"]),
    }

    monte_carlo = monte_carlo_summary(catalog_by_state, residuals, config)

    report = {
        "config": {
            "max_energy_cm": config.max_energy_cm,
            "include_optional_residuals": config.include_optional_residuals,
            "monte_carlo_samples": config.monte_carlo_samples,
        },
        "coverage_summary": coverage_summary,
        "benchmark_results": benchmark_results,
        "magic_wavelengths_nm": magic_results,
        "geometry_metrics": geometry_metrics,
        "top_channel_breakdown_1S0_515.2": breakdown_1s0,
        "top_channel_breakdown_3P1_515.2": breakdown_3p1,
        "convergence": convergence,
        "monte_carlo": monte_carlo,
        "notes": [
            "The 495-535 nm 1S0/3P1 model uses Cooper et al. PRX 2018 recommended RDMEs plus literature residual contributions.",
            "The 800-1070 nm 3P0 model uses Safronova et al. PRA 87 012509 (2013) recommended clock-magic RDMEs plus Madjarov-documented residual scalar terms.",
            "Literature RDMEs are preferred over NIST-derived Aki matrix elements whenever such literature values are available.",
            "Optional residuals are disabled by default to keep the default model free of experiment-matched fitting layers.",
            "Missing-channel coverage is reported explicitly in outputs/missing_channel_report_sr1.csv.",
        ],
    }

    with open(OUTPUT_DIR / "research_validation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    config = ModelConfig()
    levels = load_levels(LEVELS_PATH)
    lines = load_lines(LINES_PATH)
    literature = load_literature_matrix_elements(LITERATURE_MATRIX_PATH)
    residuals = load_residuals(RESIDUALS_PATH)
    benchmarks = load_benchmarks(BENCHMARKS_PATH)

    line_index = build_line_index(lines)
    catalog_by_state = {
        state_label: build_channel_catalog(
            state_label,
            levels=levels,
            line_index=line_index,
            literature_index=literature,
            config=config,
        )
        for state_label in TARGET_STATES
    }

    write_plots(catalog_by_state, residuals, config)
    write_validation_report(levels, catalog_by_state, residuals, benchmarks, config)
    print(f"Wrote research outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
