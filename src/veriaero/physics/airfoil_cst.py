"""4-D Frozen UIUC PCA -> 18-Parameter Kulfan CST Airfoil Parametrization.

Implements Section 6.1 of the VeriAero-CEGIS v3.2 Architecture Reference:
- 4-D frozen PCA modes u_CST fit deterministically on a curated UIUC low-Re dev split
- Reconstructed to 18-parameter Kulfan CST (8 upper + 8 lower + 1 leading-edge weight +
  1 blunt trailing edge t_TE/c = 0.002), natively accepted by NeuralFoil 0.3.3.
- Enforces hard leading-edge radius constraint r_LE/c >= 0.008 (|A_0| >= sqrt(0.016))
  and blunt trailing edge t_TE/c = 0.002.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import aerosandbox as asb
import numpy as np

MIN_LE_RADIUS_OVER_C = 0.008
MIN_CST_A0 = math.sqrt(2.0 * MIN_LE_RADIUS_OVER_C)  # 0.126491...
BLUNT_TE_THICKNESS_OVER_C = 0.002

UIUC_DEV_AIRFOILS = (
    "sd7062",
    "e387",
    "sd7037",
    "clarky",
    "mh32",
    "naca2412",
    "naca4412",
    "naca4415",
    "naca0012",
    "s3021",
    "ag24",
    "e214",
)


@functools.lru_cache(maxsize=1)
def _fit_frozen_uiuc_pca() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic 4-D PCA basis over 17 active Kulfan parameters (8+8+LE) of UIUC split."""
    rows: list[np.ndarray] = []
    for name in UIUC_DEV_AIRFOILS:
        af = asb.Airfoil(name)
        k = af.to_kulfan_airfoil(n_weights_per_side=8)
        vec17 = np.concatenate(
            [
                np.asarray(k.upper_weights, dtype=float),
                np.asarray(k.lower_weights, dtype=float),
                np.array([float(k.leading_edge_weight)], dtype=float),
            ]
        )
        rows.append(vec17)

    mat = np.vstack(rows)
    mean17 = np.mean(mat, axis=0)
    centered = mat - mean17
    _u, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
    basis4 = vt[:4, :]
    std4 = s_vals[:4] / math.sqrt(max(len(rows) - 1, 1))
    return mean17, basis4, std4


def reconstruct_kulfan_from_pca(u_pca: np.ndarray) -> dict[str, Any]:
    """Reconstruct 18-param Kulfan CST dict from 4-D PCA coordinates with hard LE/TE guards."""
    mean17, basis4, std4 = _fit_frozen_uiuc_pca()
    u_vec = np.asarray(u_pca, dtype=float).reshape(4)
    w17 = mean17 + (u_vec * std4) @ basis4

    upper_w = w17[:8].copy()
    lower_w = w17[8:16].copy()
    le_weight = float(np.clip(w17[16], -0.45, 0.45))

    # Hard leading-edge radius r_LE/c >= 0.008 -> |A_0| >= sqrt(2 * 0.008) = 0.12649
    upper_w[0] = max(float(upper_w[0]), MIN_CST_A0)
    lower_w[0] = min(float(lower_w[0]), -MIN_CST_A0)

    k_af = asb.KulfanAirfoil(
        name="veriaero_cst",
        lower_weights=lower_w,
        upper_weights=upper_w,
        leading_edge_weight=le_weight,
        TE_thickness=BLUNT_TE_THICKNESS_OVER_C,
    )
    max_tc = float(k_af.max_thickness())
    if max_tc < 0.08:
        scale = 0.085 / max(max_tc, 1e-4)
        camber = 0.5 * (upper_w + lower_w)
        half_t = 0.5 * (upper_w - lower_w) * scale
        upper_w = camber + half_t
        lower_w = camber - half_t
        upper_w[0] = max(float(upper_w[0]), MIN_CST_A0)
        lower_w[0] = min(float(lower_w[0]), -MIN_CST_A0)
        k_af = asb.KulfanAirfoil(
            name="veriaero_cst",
            lower_weights=lower_w,
            upper_weights=upper_w,
            leading_edge_weight=le_weight,
            TE_thickness=BLUNT_TE_THICKNESS_OVER_C,
        )
        max_tc = float(k_af.max_thickness())

    r_le_over_c = 0.5 * (min(abs(float(upper_w[0])), abs(float(lower_w[0]))) ** 2)
    return {
        "upper_weights": [float(v) for v in upper_w],
        "lower_weights": [float(v) for v in lower_w],
        "leading_edge_weight": le_weight,
        "TE_thickness": BLUNT_TE_THICKNESS_OVER_C,
        "r_le_over_c": r_le_over_c,
        "max_thickness_over_c": max_tc,
        "max_camber_over_c": float(k_af.max_camber()),
    }


def kulfan_dict_to_airfoil(
    kulfan_dict: dict[str, Any],
    name: str = "veriaero_cst",
) -> asb.KulfanAirfoil:
    """Instantiate an AeroSandbox KulfanAirfoil from a DesignIR airfoil_cst dict."""
    return asb.KulfanAirfoil(
        name=name,
        lower_weights=np.asarray(kulfan_dict["lower_weights"], dtype=float),
        upper_weights=np.asarray(kulfan_dict["upper_weights"], dtype=float),
        leading_edge_weight=float(kulfan_dict.get("leading_edge_weight", 0.0)),
        TE_thickness=float(kulfan_dict.get("TE_thickness", BLUNT_TE_THICKNESS_OVER_C)),
    )


def kulfan_dict_to_neuralfoil_params(kulfan_dict: dict[str, Any]) -> dict[str, Any]:
    """Return exact kulfan_parameters dict accepted by neuralfoil.get_aero_from_kulfan_parameters."""
    return {
        "upper_weights": np.asarray(kulfan_dict["upper_weights"], dtype=float),
        "lower_weights": np.asarray(kulfan_dict["lower_weights"], dtype=float),
        "leading_edge_weight": float(kulfan_dict.get("leading_edge_weight", 0.0)),
        "TE_thickness": float(kulfan_dict.get("TE_thickness", BLUNT_TE_THICKNESS_OVER_C)),
    }


def kulfan_dict_to_coordinates(
    kulfan_dict: dict[str, Any],
    n_points_per_side: int = 100,
) -> np.ndarray:
    """Convert a DesignIR airfoil_cst dict into Selig-ordered (x, y) coordinates."""
    k_af = kulfan_dict_to_airfoil(kulfan_dict)
    return np.asarray(
        k_af.to_airfoil(n_coordinates_per_side=n_points_per_side).coordinates,
        dtype=float,
    )
