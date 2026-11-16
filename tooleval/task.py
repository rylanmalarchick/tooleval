"""Task specs. One directory per task under tasks/.

Every task holds:
    task.toml    summary, timeout_s, optional base, optional grade_timeout_s
    prompt.txt   what the agent is told
    grade.py     python grade.py WORKDIR; exit 0 = pass, anything else = fail

A directory task (no base) also holds:
    workspace/   files the agent starts from
    reference/   files that make the grader pass (overlaid on workspace/)
    broken/      a plausible wrong answer the grader must reject

A patch task (base = "catalyst") runs on a Catalyst slot at the pin and holds:
    setup.patch       the start state (may be empty). A mined task also reverts
                      the fix's test here, so the start state does not show the
                      answer. Grading resets mlir/test to the pin either way.
    reference.patch   applied after setup.patch; the grader must pass it
    broken*.patch     plausible wrong answers the grader must reject
    tests.patch       optional: applied to mlir/test after it is reset for
                      grading, when the pinned tests encode the bug
    hidden/           grader inputs the agent never sees
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

BASES = {"catalyst"}


@dataclass(frozen=True)
class Task:
    name: str
    root: Path
    summary: str
    prompt: str
    timeout_s: int
    grade_timeout_s: int
    base: str | None

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def reference(self) -> Path:
        return self.root / "reference"

    @property
    def broken(self) -> Path:
        return self.root / "broken"

    @property
    def grader(self) -> Path:
        return self.root / "grade.py"

    @property
    def setup_patch(self) -> str:
        return (self.root / "setup.patch").read_text()

    @property
    def reference_patch(self) -> str:
        return (self.root / "reference.patch").read_text()

    @property
    def tests_patch(self) -> str:
        path = self.root / "tests.patch"
        return path.read_text() if path.is_file() else ""

    @property
    def broken_patches(self) -> dict[str, str]:
        return {p.stem: p.read_text() for p in sorted(self.root.glob("broken*.patch"))}


def load(root: Path) -> Task:
    spec = tomllib.loads((root / "task.toml").read_text())
    base = spec.get("base")
    if base is not None and base not in BASES:
        raise ValueError(f"{root.name}: unknown base {base!r}")
    timeout_s = int(spec["timeout_s"])
    return Task(
        name=root.name,
        root=root,
        summary=spec["summary"],
        prompt=(root / "prompt.txt").read_text(),
        timeout_s=timeout_s,
        grade_timeout_s=int(spec.get("grade_timeout_s", timeout_s)),
        base=base,
    )


def discover(tasks_dir: Path) -> list[Task]:
    return [load(p) for p in sorted(tasks_dir.iterdir()) if (p / "task.toml").is_file()]
