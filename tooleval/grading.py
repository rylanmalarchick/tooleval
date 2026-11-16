"""Checks shared by patch-task graders.

A grader calls run() with its checks. Each check returns None when it passes,
or a Failure. run() prints one JSON line and exits 0 only if every check passed.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tooleval import qmlir


@dataclass(frozen=True)
class Failure:
    stage: str
    detail: object


def build_and_public_tests(tree: Path) -> Failure | None:
    """check-dialects builds quantum-opt and runs the dialect lit tests (pinned suite + tests.patch)."""
    proc = subprocess.run(
        ["cmake", "--build", str(tree / "mlir/build"), "--target", "check-dialects"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return Failure("build_and_public_tests", proc.stdout[-3000:] + proc.stderr[-1000:])
    return None


def equivalence(tree: Path, pipeline: str, hidden: Path, draws: int, tol: float, seed: int,
                summary: dict) -> Failure | None:
    """For each hidden/*.mlir: the unitary after `pipeline` equals the input's up to one global
    phase (Frobenius distance < tol) on `draws` seeded angle draws, and the gate count is at most
    hidden/baseline.json (the pinned compiler's count)."""
    quantum_opt = tree / "mlir/build/bin/quantum-opt"
    baseline = json.loads((hidden / "baseline.json").read_text())
    rng = np.random.default_rng(seed)
    worst = 0.0
    for path in sorted(hidden.glob("*.mlir")):
        src = path.read_text()
        proc = subprocess.run([str(quantum_opt), f"--pass-pipeline={pipeline}", str(path)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return Failure("quantum_opt", {path.name: proc.stderr[-2000:]})
        try:
            n = qmlir.n_angles(src)
            angle_sets = rng.uniform(-np.pi, np.pi, size=(draws, n)).tolist() if n else [[]]
            dist = max(qmlir.phase_distance(qmlir.unitary(src, a), qmlir.unitary(proc.stdout, a))
                       for a in angle_sets)
            count = qmlir.gate_count(proc.stdout)
        except qmlir.Unsupported as e:
            return Failure("unsupported_output", {path.name: str(e)})
        if not dist < tol:
            return Failure("equivalence", {path.name: dist, "output": proc.stdout})
        if count > baseline[path.stem]:
            return Failure("no_regression", {path.name: {"gates": count, "baseline": baseline[path.stem]}})
        worst = max(worst, dist)
    summary["max_phase_distance"] = worst
    return None


def run(*checks) -> None:
    """Run checks in order, stop at the first failure, print the verdict as JSON, and exit."""
    summary: dict = {}
    for check in checks:
        failure = check(summary)
        if failure is not None:
            print(json.dumps({"passed": False, "stage": failure.stage, "detail": failure.detail}))
            sys.exit(1)
    print(json.dumps({"passed": True, **summary}))
    sys.exit(0)
