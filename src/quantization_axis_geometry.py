from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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


GEOMETRY_NAMES = (
    "linear_z",
    "linear_x",
    "linear_xz",
    "elliptical_xy",
)


@dataclass(frozen=True)
class ModelConfig:
    quantization_axis_angle_deg: float = 0.0

    def light_geometry(self) -> LightGeometry:
        return LightGeometry.linear_at_angle_xz(self.quantization_axis_angle_deg)


def build_geometry(
    geometry_name: str,
    theta_deg: float = 0.0,
    gamma_deg: float = 0.0,
) -> LightGeometry:
    """Construct a light geometry from named parameters.

    Supported geometries:
    - linear_z
    - linear_x
    - linear_xz      uses theta_deg
    - elliptical_xy  uses gamma_deg
    """
    normalized = geometry_name.strip().lower()
    if normalized == "linear_z":
        return LightGeometry.linear_along_z()
    if normalized == "linear_x":
        return LightGeometry.linear_along_x()
    if normalized == "linear_xz":
        return LightGeometry.linear_at_angle_xz(theta_deg)
    if normalized == "elliptical_xy":
        return LightGeometry.elliptical_xy(gamma_deg)
    supported = ", ".join(GEOMETRY_NAMES)
    raise ValueError(f"Unsupported geometry '{geometry_name}'. Supported values: {supported}.")


def effective_polarizability(
    J: int,
    mJ: int,
    components: PolarizabilityComponents,
    geometry: LightGeometry,
) -> float:
    """Young Eq. (2.24) convention with quantization axis fixed along z.

    For linear polarization at angle theta to the quantization axis:
    tensor_geom = (3 cos^2(theta) - 1) / 2.
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


def alpha_eff_for_theta(
    J: int,
    mJ: int,
    scalar: float,
    tensor: float,
    theta_deg: float,
) -> float:
    """Convenience helper for linearly polarized light in the x-z plane."""
    if J == 0:
        return scalar
    theta = math.radians(theta_deg)
    tensor_geom = (3.0 * math.cos(theta) ** 2 - 1.0) / 2.0
    tensor_state = (3.0 * mJ**2 - J * (J + 1)) / (J * (2 * J - 1))
    return scalar + tensor * tensor_geom * tensor_state


def magic_quantization_angle_deg(
    ground_alpha_au: float,
    excited_components: PolarizabilityComponents,
) -> float | None:
    """Magic angle for 1S0 vs 3P1(mJ=0) under linear polarization."""
    if excited_components.tensor == 0.0:
        return None
    cos2 = (excited_components.scalar + excited_components.tensor - ground_alpha_au) / (
        3.0 * excited_components.tensor
    )
    if not 0.0 <= cos2 <= 1.0:
        return None
    return math.degrees(math.acos(math.sqrt(cos2)))


if __name__ == "__main__":
    config = ModelConfig(quantization_axis_angle_deg=22.5)
    geometry = config.light_geometry()

    components_3p1 = PolarizabilityComponents(
        scalar=769.15134,
        vector=0.0,
        tensor=-110.561488,
    )
    alpha_3p1_m0 = effective_polarizability(
        J=1,
        mJ=0,
        components=components_3p1,
        geometry=geometry,
    )
    print("geometry:", geometry.name)
    print("3P1 mJ=0 alpha_eff:", alpha_3p1_m0)
