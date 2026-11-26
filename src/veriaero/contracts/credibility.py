"""NASA-STD-7009B / NASA-HDBK-7009B aligned credibility placarding & Verification Ledger.

Provides:
- `boltzmann_softmax_proximity`: bounded C_infinity archive-invariant extrapolation metric
  K_max_tilde(z) in [0, 1] with beta = 10.
- `VerificationCoverageLedger`: progressive tracker of digitally verified tiers vs.
  open virtual-test gaps and the 5 mandatory physical-test gates.
- `CredibilityEvidenceCard`: machine-readable self-assessment placard (explicitly
  disclaiming NASA certification and any "one-shot build" guarantee).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from veriaero.contracts.types import SolverStatus

DEFAULT_OPEN_VIRTUAL_GAPS = [
    "Tier H3a: 3D unsteady rotor-on-wing wake impingement & aeroacoustics (OpenVSP/VSPAERO + PSU-WOPWOP)",
    "Tier H3b: 3D composite shell, joint/pin bearing, thermo-elastic & modal FEA (CalculiX 2.24)",
    "Tier H3c: Closed-loop 6-DoF SITL virtual flight test with sensor/wind/fault injection (JSBSim + PX4/ArduPilot)",
]

MANDATORY_PHYSICAL_TEST_GATES = [
    "Composite manufacturing defects (resin voids, bond-line strength, layup mass variance coupons)",
    "Mechanical joint wear/backlash and cyclic structural fatigue rigs",
    "High-current EMI/RF avionics interference and battery pack thermal-runaway propagation testing",
    "Ground vibration testing (GVT) and propeller/motor harmonic balancing",
    "Full-scale acoustic ground reflections and unconventional rotor bench validation",
]

MANDATORY_SCOPE_LIMIT_TEXT = (
    "Unmodeled 3D unsteady rotor-wake impingement, transition flight dynamics, "
    "composite ply delamination, EMI/wiring interference, and physical flight-test "
    "airworthiness certification."
)

DISCLAIMER_TEXT = (
    "NASA-STD-7009B (March 5, 2024) & NASA-HDBK-7009B (Feb 3, 2026) aligned project "
    "self-assessment ONLY. Does NOT constitute NASA certification, FAA airworthiness "
    "approval, or a one-shot physical build guarantee."
)


def boltzmann_softmax_proximity(
    z: np.ndarray,
    archive_z: np.ndarray,
    lengthscale: float | np.ndarray = 1.0,
    beta: float = 10.0,
) -> float:
    """Compute archive-size-invariant Boltzmann softmax kernel proximity K_max_tilde(z) in [0, 1]."""
    z_vec = np.asarray(z, dtype=float).reshape(1, -1)
    arch = np.asarray(archive_z, dtype=float)
    if arch.size == 0:
        return 0.0
    if arch.ndim == 1:
        arch = arch.reshape(1, -1)

    l_scale = np.asarray(lengthscale, dtype=float)
    diff = (arch - z_vec) / np.maximum(l_scale, 1e-9)
    sq_dist = np.sum(diff * diff, axis=1)
    k_vals = np.exp(-0.5 * sq_dist)

    scaled = beta * k_vals
    m_shift = float(np.max(scaled))
    weights = np.exp(scaled - m_shift)
    k_tilde = float(np.sum(k_vals * weights) / np.sum(weights))
    return float(np.clip(k_tilde, 0.0, 1.0))


class VerificationCoverageLedger(BaseModel):
    """Progressive verification coverage tracker across digital tiers & physical gates."""

    verified_tiers_executed: list[str] = Field(default_factory=lambda: ["L"])
    open_virtual_test_gaps: list[str] = Field(
        default_factory=lambda: list(DEFAULT_OPEN_VIRTUAL_GAPS)
    )
    mandatory_physical_gates_remaining: list[str] = Field(
        default_factory=lambda: list(MANDATORY_PHYSICAL_TEST_GATES)
    )


class CredibilityEvidenceCard(BaseModel):
    """Machine-readable credibility & extrapolation placard for a candidate finalist."""

    design_id: str
    disclaimer: str = DISCLAIMER_TEXT
    placard_level: Literal["GREEN", "AMBER", "RED"]
    flags: list[str] = Field(default_factory=list)
    k_max_tilde: float
    local_mach: float
    min_reynolds: float
    mandatory_scope_limit: str = MANDATORY_SCOPE_LIMIT_TEXT
    coverage_ledger: VerificationCoverageLedger = Field(
        default_factory=VerificationCoverageLedger
    )

    @classmethod
    def evaluate_placard(
        cls,
        design_id: str,
        oracle_statuses: list[SolverStatus],
        local_mach: float,
        min_reynolds: float,
        k_max_tilde: float,
        verified_tiers: list[str] | None = None,
    ) -> CredibilityEvidenceCard:
        """Apply v3.2 Section 6.5 RED/AMBER/GREEN placard logic."""
        flags: list[str] = []
        level: Literal["GREEN", "AMBER", "RED"] = "GREEN"

        if any(st == SolverStatus.UNRESOLVED_NUMERICAL for st in oracle_statuses):
            flags.append("RED (INCONCLUSIVE): >= 1 oracle returned UNRESOLVED_NUMERICAL")
            level = "RED"

        if not (math.isfinite(local_mach) and math.isfinite(min_reynolds)) or (
            local_mach > 0.30 or min_reynolds < 150_000.0
        ):
            flags.append(
                f"RED (OUT_OF_ENVELOPE): Mach={local_mach:.3f} (>0.30) or Re={min_reynolds:.0f} (<150,000)"
            )
            level = "RED"

        if k_max_tilde < 0.25:
            flags.append(
                f"AMBER (EXTRAPOLATION): K_max_tilde={k_max_tilde:.3f} < 0.25 archive support threshold"
            )
            if level != "RED":
                level = "AMBER"

        flags.append(f"MANDATORY_SCOPE_LIMIT: {MANDATORY_SCOPE_LIMIT_TEXT}")
        ledger = VerificationCoverageLedger(
            verified_tiers_executed=verified_tiers or ["L"],
        )
        return cls(
            design_id=design_id,
            placard_level=level,
            flags=flags,
            k_max_tilde=float(k_max_tilde),
            local_mach=float(local_mach),
            min_reynolds=float(min_reynolds),
            coverage_ledger=ledger,
        )
