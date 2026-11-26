"""Import firewall test: src/ must NEVER import from eval_quarantine/."""

from __future__ import annotations

import ast
from pathlib import Path


def test_src_never_imports_eval_quarantine() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    py_files = list(src_root.rglob("*.py"))
    assert len(py_files) > 0

    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "eval_quarantine" or alias.name.startswith("eval_quarantine."):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "eval_quarantine" or mod.startswith("eval_quarantine."):
                    violations.append(f"{path}:{node.lineno} imports from {mod}")

    assert not violations, "Quarantine firewall breached:\n" + "\n".join(violations)
