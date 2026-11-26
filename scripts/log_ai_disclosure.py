"""Automated AI Disclosure Log CLI for VeriAero-CEGIS v3.2 (`docs/ai_disclosure_log.csv`).

Implements the exact 7-column workflow required by `AGENTS.md`:
  - `append`: Appends one entry with `NO_STUDENT_RESPONSE_YET` sentinels (never fabricated).
  - `verify`: Records authentic student verification / defect observations onto a row
    and optionally backfills placeholder git commit SHAs (`--fill-sha`).
  - `review`: Interactive terminal prompt to complete pending rows.
  - `status`: Audits completion state (`--strict` fails on any pending/placeholder cell).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "date_utc",
    "tool_name_and_version",
    "task_category",
    "exact_scope_of_assistance",
    "student_independent_verification",
    "defect_caught_by_student",
    "git_commit_or_artifact_ref",
]
PENDING_SENTINEL = "NO_STUDENT_RESPONSE_YET"
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "docs" / "ai_disclosure_log.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _current_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or "uncommitted"
    except Exception:
        return "uncommitted"


def _is_placeholder(val: str) -> bool:
    v = (val or "").strip()
    return (
        not v
        or v == PENDING_SENTINEL
        or v.startswith("PENDING_")
        or v.startswith("<STUDENT:")
    )


def _read_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.is_file():
        return []
    with log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"Unexpected CSV header in {log_path}: {reader.fieldnames} != {COLUMNS}"
            )
        return [{col: row.get(col, "") for col in COLUMNS} for row in reader]


def _write_rows(log_path: Path, rows: list[dict[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})


def cmd_append(args: argparse.Namespace) -> int:
    log_path: Path = args.log_file
    rows = _read_rows(log_path)
    ref = args.ref or _current_git_sha(log_path.resolve().parents[1])
    rows.append(
        {
            "date_utc": _utc_now_iso(),
            "tool_name_and_version": args.tool,
            "task_category": args.category,
            "exact_scope_of_assistance": args.scope,
            "student_independent_verification": PENDING_SENTINEL,
            "defect_caught_by_student": PENDING_SENTINEL,
            "git_commit_or_artifact_ref": ref,
        }
    )
    _write_rows(log_path, rows)
    print(f"Appended row {len(rows)} to {log_path} (ref={ref}).")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    log_path: Path = args.log_file
    rows = _read_rows(log_path)
    if not rows:
        print("No data rows in disclosure log.", file=sys.stderr)
        return 1

    if args.row is not None:
        idx = args.row - 1
        if idx < 0 or idx >= len(rows):
            print(f"Row {args.row} out of range (1..{len(rows)}).", file=sys.stderr)
            return 1
    else:
        pending_idxs = [
            i
            for i, r in enumerate(rows)
            if r["student_independent_verification"] == PENDING_SENTINEL
            or r["defect_caught_by_student"] == PENDING_SENTINEL
        ]
        idx = pending_idxs[-1] if pending_idxs else len(rows) - 1

    changed = False
    if args.verification is not None:
        rows[idx]["student_independent_verification"] = args.verification.strip()
        changed = True
    if args.defect is not None:
        rows[idx]["defect_caught_by_student"] = args.defect.strip()
        changed = True
    if args.ref is not None:
        rows[idx]["git_commit_or_artifact_ref"] = args.ref.strip()
        changed = True
    if args.fill_sha:
        sha = args.sha or _current_git_sha(log_path.resolve().parents[1])
        for row in rows:
            if _is_placeholder(row["git_commit_or_artifact_ref"]):
                row["git_commit_or_artifact_ref"] = sha
                changed = True

    if changed:
        _write_rows(log_path, rows)
        print(f"Updated row {idx + 1} in {log_path}.")
    else:
        print("Nothing to update (pass --verification, --defect, --ref, or --fill-sha).")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    log_path: Path = args.log_file
    rows = _read_rows(log_path)
    pending = [
        (i, r)
        for i, r in enumerate(rows)
        if _is_placeholder(r["student_independent_verification"])
        or _is_placeholder(r["defect_caught_by_student"])
    ]
    if not pending:
        print("All rows have student verification and defect entries.")
        return 0

    for i, row in pending:
        print(f"\n--- Row {i + 1}: [{row['task_category']}] ({row['date_utc']}) ---")
        print(f"Scope: {row['exact_scope_of_assistance']}")
        if _is_placeholder(row["student_independent_verification"]):
            ans = input("Student independent verification (Enter to skip): ").strip()
            if ans:
                row["student_independent_verification"] = ans
        if _is_placeholder(row["defect_caught_by_student"]):
            ans = input("Defect caught / assumption challenged ('None reported', Enter to skip): ").strip()
            if ans:
                row["defect_caught_by_student"] = ans
    _write_rows(log_path, rows)
    print(f"Saved {log_path}.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    log_path: Path = args.log_file
    rows = _read_rows(log_path)
    bad_cells: list[tuple[int, str, str]] = []
    for i, row in enumerate(rows, 1):
        for col in COLUMNS:
            if _is_placeholder(row[col]):
                bad_cells.append((i, col, row[col]))

    print(f"AI disclosure log: {log_path} ({len(rows)} row(s), {len(bad_cells)} placeholder cell(s))")
    for row_num, col, val in bad_cells:
        print(f"  Row {row_num} [{col}]: {val}")

    if args.strict and bad_cells:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage VeriAero-CEGIS AI disclosure log")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_app = sub.add_parser("append", help="Append a new AI assistance entry with pending sentinels")
    p_app.add_argument("--category", required=True)
    p_app.add_argument("--scope", required=True)
    p_app.add_argument("--tool", default="Antigravity AI Partner")
    p_app.add_argument("--ref", default=None)
    p_app.set_defaults(func=cmd_append)

    p_ver = sub.add_parser("verify", help="Record student verification/defect or backfill git SHAs")
    p_ver.add_argument("--row", type=int, default=None, help="1-indexed data row number")
    p_ver.add_argument("--verification", default=None)
    p_ver.add_argument("--defect", default=None)
    p_ver.add_argument("--ref", default=None)
    p_ver.add_argument("--fill-sha", action="store_true")
    p_ver.add_argument("--sha", default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_rev = sub.add_parser("review", help="Interactively fill pending student responses")
    p_rev.set_defaults(func=cmd_review)

    p_stat = sub.add_parser("status", help="Inspect log completeness")
    p_stat.add_argument("--strict", action="store_true", help="Exit nonzero if placeholders remain")
    p_stat.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
