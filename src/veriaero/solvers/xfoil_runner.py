"""Subprocess wrapper for patched XFoil 6.99 (`bin/xfoil_patched`).

Aligned with VeriAero-CEGIS v3.2 Section 6.5:
- Panel count N = 200, N_crit = 9.0
- Supports both free transition and forced trip (x_tr = 0.05) roughness sweeps
- Tri-state status reporting (`UNRESOLVED_NUMERICAL` whenever missing, timed out,
  crashed via SIGFPE, or failed to converge).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veriaero.contracts.types import SolverStatus


@dataclass(frozen=True)
class XFoilPolarPoint:
    """Single converged viscous XFoil operating point."""

    alpha_deg: float
    cl: float
    cd: float
    cdp: float
    cm: float
    top_xtr: float
    bot_xtr: float


@dataclass(frozen=True)
class XFoilRunResult:
    """Tri-state output of an XFoil 6.99 polar run."""

    status: SolverStatus
    points: tuple[XFoilPolarPoint, ...] = ()
    failure_reason: str | None = None


def default_xfoil_binary() -> Path:
    """Return canonical path to the patched XFoil 6.99 executable."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "bin" / "xfoil_patched"


def parse_xfoil_polar(polar_text: str) -> list[XFoilPolarPoint]:
    """Parse XFoil PACC ASCII polar output lines into structured points."""
    points: list[XFoilPolarPoint] = []
    float_row_re = re.compile(
        r"^\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+"
        r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)"
    )
    for line in polar_text.splitlines():
        match = float_row_re.match(line)
        if not match:
            continue
        vals = [float(g) for g in match.groups()]
        if not all(math.isfinite(v) for v in vals):
            continue
        points.append(
            XFoilPolarPoint(
                alpha_deg=vals[0],
                cl=vals[1],
                cd=vals[2],
                cdp=vals[3],
                cm=vals[4],
                top_xtr=vals[5],
                bot_xtr=vals[6],
            )
        )
    return points


class XFoilRunner:
    """Deterministic subprocess driver for patched XFoil 6.99."""

    def __init__(
        self,
        binary_path: str | Path | None = None,
        timeout_s: float = 12.0,
        max_iter: int = 160,
        n_panels: int = 200,
        n_crit: float = 9.0,
    ) -> None:
        self.binary_path = Path(binary_path) if binary_path else default_xfoil_binary()
        self.timeout_s = float(timeout_s)
        self.max_iter = int(max_iter)
        self.n_panels = int(n_panels)
        self.n_crit = float(n_crit)

    def is_available(self) -> bool:
        return self.binary_path.is_file() and os.access(self.binary_path, os.X_OK)

    def build_command_deck(
        self,
        airfoil_file: str,
        polar_file: str,
        reynolds: float,
        mach: float,
        alphas_deg: list[float] | None = None,
        x_tr_trip: float | None = None,
        cl_targets: list[float] | None = None,
        aseq: tuple[float, float, float] | None = None,
    ) -> str:
        """Build an XFoil batch stdin script (N=200, N_crit=9.0, optional x_tr trip, CL & ASEQ)."""
        lines = [
            "PLOP",
            "G F",
            "",
            f"LOAD {airfoil_file}",
            "PPAR",
            f"N {self.n_panels}",
            "",
            "",
            "PANE",
            "OPER",
            f"VISC {reynolds:.1f}",
            f"MACH {mach:.4f}",
            "VPAR",
            f"N {self.n_crit:.2f}",
        ]
        if x_tr_trip is not None:
            lines.append(f"XTR {x_tr_trip:.3f} {x_tr_trip:.3f}")
        lines.extend(
            [
                "",
                f"ITER {self.max_iter}",
                "PACC",
                polar_file,
                "",
            ]
        )
        for cl_val in cl_targets or ():
            lines.append(f"CL {cl_val:.4f}")
        for alpha in alphas_deg or ():
            lines.append(f"ALFA {alpha:.4f}")
        if aseq is not None:
            a_start, a_end, a_step = aseq
            lines.append(f"ASEQ {a_start:.3f} {a_end:.3f} {a_step:.3f}")
        lines.extend(["PACC", "", "QUIT", ""])
        return "\n".join(lines)

    def run_polar(
        self,
        airfoil_coords: np.ndarray,
        reynolds: float,
        mach: float,
        alphas_deg: list[float] | None = None,
        x_tr_trip: float | None = None,
        cl_targets: list[float] | None = None,
        aseq: tuple[float, float, float] | None = None,
    ) -> XFoilRunResult:
        """Execute patched XFoil 6.99 on airfoil coordinates for target CL(s), alpha(s), or ASEQ."""
        if not self.is_available():
            return XFoilRunResult(
                status=SolverStatus.UNRESOLVED_NUMERICAL,
                failure_reason=(
                    f"XFoil binary not found/executable at '{self.binary_path}'. "
                    "Run scripts/bootstrap_wsl2.sh on C2-WSL2."
                ),
            )

        with tempfile.TemporaryDirectory(prefix="xfoil_run_") as tmpdir:
            tmp = Path(tmpdir)
            dat_name = "airfoil.dat"
            pol_name = "polar.pol"

            with (tmp / dat_name).open("w", encoding="utf-8") as fh:
                fh.write("VERIAERO_AIRFOIL\n")
                for x_val, y_val in airfoil_coords:
                    fh.write(f" {x_val:10.6f} {y_val:10.6f}\n")

            deck = self.build_command_deck(
                airfoil_file=dat_name,
                polar_file=pol_name,
                reynolds=reynolds,
                mach=mach,
                alphas_deg=alphas_deg,
                x_tr_trip=x_tr_trip,
                cl_targets=cl_targets,
                aseq=aseq,
            )

            try:
                env = os.environ.copy()
                env.pop("DISPLAY", None)
                proc = subprocess.run(
                    [str(self.binary_path)],
                    input=deck,
                    text=True,
                    capture_output=True,
                    cwd=str(tmp),
                    env=env,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return XFoilRunResult(
                    status=SolverStatus.UNRESOLVED_NUMERICAL,
                    failure_reason=f"XFoil timed out after {self.timeout_s:.1f}s",
                )
            except Exception as exc:
                return XFoilRunResult(
                    status=SolverStatus.UNRESOLVED_NUMERICAL,
                    failure_reason=f"XFoil subprocess error: {exc}",
                )

            if proc.returncode != 0:
                return XFoilRunResult(
                    status=SolverStatus.UNRESOLVED_NUMERICAL,
                    failure_reason=f"XFoil exited with non-zero code {proc.returncode}",
                )

            polar_path = tmp / pol_name
            if not polar_path.is_file():
                return XFoilRunResult(
                    status=SolverStatus.UNRESOLVED_NUMERICAL,
                    failure_reason="XFoil produced no polar output file",
                )

            points = parse_xfoil_polar(polar_path.read_text(encoding="utf-8", errors="ignore"))
            if not points:
                return XFoilRunResult(
                    status=SolverStatus.UNRESOLVED_NUMERICAL,
                    failure_reason="XFoil failed to converge on any requested alpha",
                )

            return XFoilRunResult(
                status=SolverStatus.RESOLVED_PASS,
                points=tuple(points),
            )
