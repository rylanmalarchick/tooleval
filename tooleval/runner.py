"""Run an agent on a task in a sandbox, then grade the result outside it."""

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tooleval import catalyst, sandbox
from tooleval.task import Task

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Grade:
    passed: bool
    output: str


@dataclass(frozen=True)
class Agent:
    cmd: str  # shell command; the prompt path is in $TOOLEVAL_PROMPT
    home: Path  # template: each trial gets a fresh copy, mounted into $HOME
    files: tuple[Path, ...] = ()  # mounted read-write at their own paths


@dataclass(frozen=True)
class Result:
    task: str
    agent: str
    passed: bool
    timed_out: bool
    agent_returncode: int | None
    agent_wall_s: float
    touched_protected: bool | None  # patch tasks: the agent diff changed mlir/test
    task_sha256: str  # identity of the task version that was run
    meta: dict  # run-level facts: arm, model, effort, CLI version, tooleval commit, pilot
    grader_output: str


def task_sha256(task: Task) -> str:
    """Hash of every file in the task dir (paths and contents). Any change to the
    prompt, grader, patches, or hidden inputs gives a new hash."""
    h = hashlib.sha256()
    for path in sorted(task.root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(str(path.relative_to(task.root)).encode() + b"\0")
            h.update(path.read_bytes() + b"\0")
    return h.hexdigest()


def fresh_home(template: Path, dest: Path) -> Path:
    """Copy the agent-home template to dest, so no state carries over between trials."""
    shutil.copytree(template, dest, symlinks=True)
    return dest


def prepare(task: Task, dest: Path, overlay: Path | None = None) -> None:
    """Directory task: copy the starting workspace to dest, then lay overlay's files on top."""
    shutil.copytree(task.workspace, dest)
    if overlay is not None:
        shutil.copytree(overlay, dest, dirs_exist_ok=True)


def _run_grader(task: Task, workdir: Path) -> Grade:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    try:
        proc = subprocess.run(
            [sys.executable, str(task.grader), str(workdir)],
            capture_output=True,
            text=True,
            timeout=task.grade_timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return Grade(False, f"grader timed out after {task.grade_timeout_s} s")
    return Grade(proc.returncode == 0, (proc.stdout + proc.stderr).strip())


def grade(task: Task, workdir: Path) -> Grade:
    """Directory task: grade the files in workdir."""
    return _run_grader(task, workdir)


def _apply_setup(task: Task, slot: catalyst.Slot) -> None:
    if task.setup_patch.strip():
        setup = catalyst.apply(slot, task.setup_patch)
        if setup.returncode != 0:
            raise RuntimeError(f"{task.name}: setup.patch does not apply:\n{setup.stderr}")


def grade_patch(task: Task, patch: str | None) -> Grade:
    """Patch task: on the grade slot, apply setup.patch then `patch`, reset mlir/test
    (then apply tests.patch, if any), and grade."""
    with catalyst.locked(catalyst.GRADE) as slot:
        catalyst.reset(slot)
        _apply_setup(task, slot)
        if patch:
            applied = catalyst.apply(slot, patch)
            if applied.returncode != 0:
                return Grade(False, f"candidate patch does not apply:\n{applied.stderr}")
        catalyst.restore_protected(slot)
        if task.tests_patch.strip():
            tests = catalyst.apply(slot, task.tests_patch)
            if tests.returncode != 0:
                raise RuntimeError(f"{task.name}: tests.patch does not apply:\n{tests.stderr}")
        return _run_grader(task, slot.tree)


def _run_agent(task: Task, agent: Agent, task_dir: Path, workdir: Path, ro_paths, path_prefix):
    prompt_file = task_dir / "prompt.txt"
    prompt_file.write_text(task.prompt)
    home = fresh_home(agent.home, task_dir / "agent-home")
    argv = sandbox.wrap(agent.cmd, workdir, prompt_file, home, ro_paths, path_prefix, agent.files)
    start = time.monotonic()
    with open(task_dir / "agent.stdout", "w") as out, open(task_dir / "agent.stderr", "w") as err:
        try:
            proc = subprocess.run(argv, stdout=out, stderr=err, timeout=task.timeout_s)
            returncode, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timed_out = None, True
    return returncode, timed_out, round(time.monotonic() - start, 2)


def run(task: Task, agent: Agent, run_dir: Path, meta: dict) -> Result:
    """Run the agent command in the sandbox with cwd = the task's start state.

    The prompt path is in $TOOLEVAL_PROMPT. Agent stdout and stderr go to
    run_dir/<task>/agent.stdout and agent.stderr. The agent's home for this
    trial is kept in run_dir/<task>/agent-home.
    """
    digest = task_sha256(task)
    task_dir = run_dir / task.name
    task_dir.mkdir(parents=True)
    if task.base is None:
        workdir = task_dir / "workspace"
        prepare(task, workdir)
        rc, timed_out, wall = _run_agent(task, agent, task_dir, workdir, (), ())
        g = grade(task, workdir)
        return Result(task.name, agent.cmd, g.passed, timed_out, rc, wall, None, digest, meta, g.output)

    with catalyst.locked(catalyst.TRIAL) as slot:
        catalyst.reset(slot)
        _apply_setup(task, slot)
        built = catalyst.build(slot.tree)
        if built.returncode != 0:
            raise RuntimeError(f"{task.name}: start state does not build:\n{built.stdout[-4000:]}")
        start = catalyst.start_visible_repo(slot)
        ro_paths = [catalyst.BASE / s for s in catalyst.SUBMODULES] + [catalyst.PYTHON.parent.parent]
        rc, timed_out, wall = _run_agent(task, agent, task_dir, slot.tree, ro_paths, [catalyst.PYTHON.parent])
        diff = catalyst.agent_diff(slot, start)
    (task_dir / "agent.diff").write_text(diff)
    g = grade_patch(task, diff)
    touched = catalyst.touches_protected(diff)
    return Result(task.name, agent.cmd, g.passed, timed_out, rc, wall, touched, digest, meta, g.output)
