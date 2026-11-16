"""Catalyst slots: copies of the Catalyst tree at the pinned commit, each with its own build.

A slot is WORK/slots/<name>/. Its tree is slot/catalyst. The pristine state
lives in a git dir outside the tree (slot/pin.git), so an agent that sees
only the tree cannot read Catalyst history or the reset state. LLVM,
StableHLO, and Enzyme are symlinks into the base tree and are never rebuilt.

Slots run on the desktop, where the base tree is built (see PLAN.txt).
"""

import fcntl
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

WORK = Path(os.environ.get("TOOLEVAL_WORK", Path.home() / "dev/projects/tooleval-work"))
BASE = WORK / "catalyst"
PIN = "f2428c5075c50da0bf823c41b721c2916513732f"
PYTHON = Path.home() / ".venvs/catalyst/bin/python"
SUBMODULES = ("mlir/llvm-project", "mlir/stablehlo", "mlir/Enzyme")
KEEP = ("/mlir/build", *(f"/{s}" for s in SUBMODULES))  # survive a reset
PROTECTED = ("mlir/test",)  # reset to the pin before grading
_IDENTITY = {
    "GIT_AUTHOR_NAME": "tooleval",
    "GIT_AUTHOR_EMAIL": "tooleval@localhost",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "tooleval",
    "GIT_COMMITTER_EMAIL": "tooleval@localhost",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


@dataclass(frozen=True)
class Slot:
    name: str

    @property
    def root(self) -> Path:
        return WORK / "slots" / self.name

    @property
    def tree(self) -> Path:
        return self.root / "catalyst"

    @property
    def git_dir(self) -> Path:
        return self.root / "pin.git"

    @property
    def build_dir(self) -> Path:
        return self.tree / "mlir/build"

    def exists(self) -> bool:
        return (self.git_dir / "HEAD").is_file() and (self.build_dir / "bin/quantum-opt").is_file()


TRIAL = Slot("trial-0")
GRADE = Slot("grade")


def _run(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], check=True, text=True, **kw)


def _pin_git(slot: Slot, *args, **kw) -> subprocess.CompletedProcess:
    return _run(["git", f"--git-dir={slot.git_dir}", f"--work-tree={slot.tree}", *args], **kw)


def check_base() -> None:
    head = _run(["git", "-C", BASE, "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if head != PIN:
        raise RuntimeError(f"base tree {BASE} is at {head}, expected pin {PIN}")


@contextmanager
def locked(slot: Slot):
    """Hold an exclusive lock on the slot so two runs cannot share it."""
    slot.root.mkdir(parents=True, exist_ok=True)
    with open(slot.root / "lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield slot


def create(slot: Slot) -> None:
    """Make the slot tree from the base at the pin, then configure and build it (about 6 min)."""
    check_base()
    slot.tree.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(BASE), "archive", PIN], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(slot.tree)], input=archive, check=True)
    for sub in SUBMODULES:
        link = slot.tree / sub
        if link.is_dir():
            link.rmdir()  # git archive leaves an empty dir for a gitlink
        link.symlink_to(BASE / sub)
    _run(["git", "init", "-q", "--bare", slot.git_dir])
    (slot.git_dir / "info").mkdir(exist_ok=True)
    (slot.git_dir / "info/exclude").write_text("\n".join(KEEP) + "\n")
    _pin_git(slot, "config", "core.bare", "false")
    _pin_git(slot, "add", "-A")
    _pin_git(slot, "commit", "-q", "-m", f"catalyst {PIN}", env={**os.environ, **_IDENTITY})
    configure(slot)
    _run(["cmake", "--build", slot.build_dir, "--target", "check-dialects"])


def configure(slot: Slot) -> None:
    """Configure mlir/build the way Catalyst's `make dialects` does, against the base LLVM build."""
    mlir = slot.tree / "mlir"
    numpy_inc = _run([PYTHON, "-c", "import numpy; print(numpy.get_include())"], capture_output=True).stdout.strip()
    _run([
        "cmake", "-G", "Ninja", "-S", mlir, "-B", slot.build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DQUANTUM_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython_EXECUTABLE={PYTHON}",
        f"-DPython3_EXECUTABLE={PYTHON}",
        f"-DPython3_NumPy_INCLUDE_DIRS={numpy_inc}",
        f"-DEnzyme_DIR={mlir}/Enzyme/build",
        f"-DENZYME_SRC_DIR={mlir}/Enzyme",
        f"-DMLIR_DIR={mlir}/llvm-project/build/lib/cmake/mlir",
        f"-DSTABLEHLO_DIR={mlir}/stablehlo",
        f"-DSTABLEHLO_BUILD_DIR={mlir}/stablehlo/build",
        f"-DRUNTIME_LIB_DIR={slot.tree}/runtime/build/lib",
        f"-DMLIR_LIB_DIR={mlir}/llvm-project/build/lib",
        f"-DCMAKE_C_COMPILER={shutil.which('clang')}",
        f"-DCMAKE_CXX_COMPILER={shutil.which('clang++')}",
        f"-DCMAKE_C_COMPILER_LAUNCHER={shutil.which('ccache')}",
        f"-DCMAKE_CXX_COMPILER_LAUNCHER={shutil.which('ccache')}",
        "-DLLVM_ENABLE_LLD=ON",
        "-DLLVM_ENABLE_ZLIB=ON",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DCATALYST_ENABLE_WARNINGS=ON",
    ], capture_output=True)


def reset(slot: Slot) -> None:
    """Return the tree to the pin. The build dir is kept, so the next build is incremental."""
    visible = slot.tree / ".git"
    if visible.exists():
        shutil.rmtree(visible)
    _pin_git(slot, "reset", "-q", "--hard", "HEAD")
    _pin_git(slot, "clean", "-ffdxq", *(f"-e{k}" for k in KEEP))


def apply(slot: Slot, patch: str) -> subprocess.CompletedProcess:
    """git apply the patch text to the tree. Does not raise; check returncode."""
    return subprocess.run(
        ["git", f"--git-dir={slot.git_dir}", f"--work-tree={slot.tree}", "apply", "--binary", "-"],
        input=patch, capture_output=True, text=True, cwd=slot.tree,
    )


def restore_protected(slot: Slot) -> None:
    _pin_git(slot, "checkout", "HEAD", "--", *PROTECTED)
    _pin_git(slot, "clean", "-ffdxq", "--", *PROTECTED)


def touched(patch: str) -> set[str]:
    """Paths a git patch changes (from its diff --git headers)."""
    paths = set()
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            a, _, b = line[len("diff --git a/"):].partition(" b/")
            paths.update((a, b))
    return paths


def is_protected(path: str) -> bool:
    return any(path == d or path.startswith(d + "/") for d in PROTECTED)


def touches_protected(patch: str) -> bool:
    return any(is_protected(p) for p in touched(patch))


def build(tree: Path, *targets: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """ninja the given targets (default quantum-opt). Does not raise; check returncode."""
    return subprocess.run(
        ["ninja", "-C", str(tree / "mlir/build"), *(targets or ("quantum-opt",))],
        capture_output=True, text=True, timeout=timeout,
    )


def start_visible_repo(slot: Slot) -> str:
    """Give the agent a one-commit git repo of its start state. Return that commit."""
    git = ["git", "-C", slot.tree]
    _run([*git, "init", "-q"])
    (slot.tree / ".git/info/exclude").write_text("\n".join(KEEP) + "\n")
    _run([*git, "add", "-A"])
    env = {**os.environ, **_IDENTITY}
    _run([*git, "commit", "-q", "-m", "start"], env=env)
    return _run([*git, "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def agent_diff(slot: Slot, start: str) -> str:
    """Everything the agent changed in the tree since `start`, untracked files included."""
    git = ["git", "-C", slot.tree]
    _run([*git, "add", "-A"])
    return _run([*git, "diff", "--cached", "--binary", start], capture_output=True).stdout
