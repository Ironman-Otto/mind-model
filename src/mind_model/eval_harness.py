"""
eval_harness.py

Tiny evaluation harness (no I/O side-effects beyond prints) to support
unit-like checks for stimulation, learning, and readout consistency.
(Demo scenarios will be added later; this file only provides utilities.)
"""
from __future__ import annotations
from typing import Callable, List, Tuple


def assert_near(x: float, y: float, tol: float = 1e-6) -> None:
    """Raise AssertionError if |x - y| > tol."""
    if abs(x - y) > tol:
        raise AssertionError(f"Values differ: {x} vs {y} (tol={tol})")


def run_checks(checks: List[Tuple[str, Callable[[], None]]]) -> None:
    """Run a list of (name, fn) checks and print a summary."""
    passed = 0
    for i, (name, fn) in enumerate(checks):
        try:
            fn()
            print(f"[PASS] {i+1:02d} – {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {i+1:02d} – {name}: {e}")
    print(f"\nSummary: {passed}/{len(checks)} checks passed.")
