# Evaluation Quarantine Firewall

This directory holds held-out evaluation scenarios and confirmatory benchmark harnesses.

**HARD RULE:** Nothing inside `src/` may EVER import from `eval_quarantine/`.
This firewall is enforced automatically by `tests/test_quarantine_firewall.py`.
