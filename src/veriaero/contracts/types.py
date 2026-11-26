"""Core tri-state evaluation, provenance, and budget accounting types.

Aligned with VeriAero-CEGIS v3.2 Architecture Reference (Sections 6.5 & 6.7).
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SolverStatus(str, Enum):
    """Tri-state evaluation outcome.

    - RESOLVED_PASS: Solver converged with finite outputs and all constraints <= tol.
    - RESOLVED_VIOLATION: Solver converged with finite outputs, but >= 1 constraint > tol.
    - UNRESOLVED_NUMERICAL: Solver crashed, timed out, failed to converge, or produced NaN/Inf.
      Never count UNRESOLVED_NUMERICAL as a physical constraint violation.
    """

    RESOLVED_PASS = "RESOLVED_PASS"
    RESOLVED_VIOLATION = "RESOLVED_VIOLATION"
    UNRESOLVED_NUMERICAL = "UNRESOLVED_NUMERICAL"


class OracleTier(str, Enum):
    """Multi-fidelity oracle tiers."""

    L = "L"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


def _detect_ram_gb() -> float:
    """Detect total system RAM in GiB without requiring extra packages."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024**3), 2)
        elif hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 2)
    except Exception:
        pass
    return 0.0


def _detect_git_commit() -> str | None:
    """Return short HEAD SHA if running inside a git repo, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        sha = out.decode("utf-8", errors="ignore").strip()
        return sha or None
    except Exception:
        return None


class HostProvenance(BaseModel):
    """Single-host execution provenance recorded on runs and oracle results."""

    host_id: str
    hostname: str
    os_platform: str
    os_release: str
    is_wsl2: bool
    cpu_model: str
    cpu_logical_cores: int
    ram_total_gb: float
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    python_version: str
    openblas_num_threads: str | None = None
    worker_count: int = 8
    git_commit: str | None = None
    captured_at_utc: str

    @classmethod
    def capture(cls, worker_count: int = 8) -> HostProvenance:
        """Capture current host hardware, OS, CUDA, and git metadata safely."""
        sys_uname = platform.uname()
        is_wsl2 = "microsoft" in sys_uname.release.lower() or "wsl" in sys_uname.release.lower()
        cpu_model = sys_uname.processor or platform.machine() or "unknown"
        hostname = platform.node() or "unknown"

        gpu_name: str | None = None
        gpu_vram_gb: float | None = None
        cuda_available = False
        cuda_version: str | None = None
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_version = getattr(torch.version, "cuda", None)
            if cuda_available:
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                gpu_vram_gb = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass

        host_tag = "C2-WSL2" if is_wsl2 else f"{hostname}-{platform.system()}"
        return cls(
            host_id=host_tag,
            hostname=hostname,
            os_platform=platform.system(),
            os_release=platform.release(),
            is_wsl2=is_wsl2,
            cpu_model=cpu_model,
            cpu_logical_cores=os.cpu_count() or 1,
            ram_total_gb=_detect_ram_gb(),
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            python_version=platform.python_version(),
            openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            worker_count=worker_count,
            git_commit=_detect_git_commit(),
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
        )


class EvalResult(BaseModel):
    """Standardized output returned by every DisciplineOracle.evaluate() call.

    Constraint convention:
        g(x) <= 0.0 -> feasible
        g(x) >  0.0 -> violated
    Signed safety margin:
        m_safe = min_m (-g_m / scale_m)  (positive inside feasible interior, negative on violation)
    """

    design_id: str
    oracle_name: str
    tier: OracleTier
    status: SolverStatus
    objectives: dict[str, float] = Field(default_factory=dict)
    constraints: dict[str, float] = Field(default_factory=dict)
    m_safe: float | None = None
    residuals_log: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    wall_time_s: float = 0.0
    failure_reason: str | None = None
    provenance: HostProvenance | None = None
    sub_results: list[EvalResult] = Field(default_factory=list)

    @property
    def max_constraint_violation(self) -> float:
        """Largest positive constraint violation (0.0 if all constraints <= 0)."""
        if not self.constraints:
            return 0.0
        return float(max([0.0, *self.constraints.values()]))


class BudgetLedger(BaseModel):
    """Tracks oracle evaluations, wall-clock cost, and tri-state outcomes by tier."""

    calls_by_tier: dict[str, int] = Field(default_factory=dict)
    wall_time_by_tier: dict[str, float] = Field(default_factory=dict)
    status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    records: list[dict[str, Any]] = Field(default_factory=list)

    def debit(self, result: EvalResult) -> None:
        """Record an oracle evaluation, always debiting calls & wall time even on failure."""
        tier_key = result.tier.value
        status_key = result.status.value

        self.calls_by_tier[tier_key] = self.calls_by_tier.get(tier_key, 0) + 1
        self.wall_time_by_tier[tier_key] = (
            self.wall_time_by_tier.get(tier_key, 0.0) + float(result.wall_time_s)
        )

        tier_statuses = self.status_counts.setdefault(
            tier_key,
            {
                SolverStatus.RESOLVED_PASS.value: 0,
                SolverStatus.RESOLVED_VIOLATION.value: 0,
                SolverStatus.UNRESOLVED_NUMERICAL.value: 0,
            },
        )
        tier_statuses[status_key] = tier_statuses.get(status_key, 0) + 1

        self.records.append(
            {
                "design_id": result.design_id,
                "oracle_name": result.oracle_name,
                "tier": tier_key,
                "status": status_key,
                "wall_time_s": float(result.wall_time_s),
                "failure_reason": result.failure_reason,
            }
        )

    def total_calls(self, tier: OracleTier | str | None = None) -> int:
        if tier is None:
            return sum(self.calls_by_tier.values())
        key = tier.value if isinstance(tier, OracleTier) else tier
        return self.calls_by_tier.get(key, 0)

    def total_wall_time_s(self, tier: OracleTier | str | None = None) -> float:
        if tier is None:
            return sum(self.wall_time_by_tier.values())
        key = tier.value if isinstance(tier, OracleTier) else tier
        return self.wall_time_by_tier.get(key, 0.0)

    def status_count(self, status: SolverStatus, tier: OracleTier | str | None = None) -> int:
        status_key = status.value
        if tier is None:
            return sum(counts.get(status_key, 0) for counts in self.status_counts.values())
        key = tier.value if isinstance(tier, OracleTier) else tier
        return self.status_counts.get(key, {}).get(status_key, 0)

    def unresolved_count(self, tier: OracleTier | str | None = None) -> int:
        return self.status_count(SolverStatus.UNRESOLVED_NUMERICAL, tier=tier)

    def unresolved_rate(self, tier: OracleTier | str | None = None) -> float:
        total = self.total_calls(tier=tier)
        if total == 0:
            return 0.0
        return self.unresolved_count(tier=tier) / total

    def within_budget(
        self,
        max_calls_by_tier: dict[str, int] | None = None,
        max_wall_s: float | None = None,
    ) -> bool:
        if max_wall_s is not None and self.total_wall_time_s() > max_wall_s:
            return False
        if max_calls_by_tier is not None:
            for tier_key, limit in max_calls_by_tier.items():
                if self.calls_by_tier.get(tier_key, 0) > limit:
                    return False
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls(),
            "total_wall_time_s": round(self.total_wall_time_s(), 4),
            "calls_by_tier": dict(self.calls_by_tier),
            "wall_time_by_tier": {k: round(v, 4) for k, v in self.wall_time_by_tier.items()},
            "status_counts": {k: dict(v) for k, v in self.status_counts.items()},
            "unresolved_rate": round(self.unresolved_rate(), 4),
        }
