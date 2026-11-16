"""Command line: python -m tooleval {list,selftest,run,slots}."""

import argparse
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tooleval import arms, catalyst, runner
from tooleval.task import Task, discover

REPO = runner.REPO
TASKS = REPO / "tasks"
RESULTS = REPO / "results"
ARMS = REPO / "arms.toml"


def _selftest_dir(task: Task) -> list[str]:
    expected = {"workspace": (None, False), "reference": (task.reference, True), "broken": (task.broken, False)}
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for label, (overlay, should_pass) in expected.items():
            workdir = Path(tmp) / label
            runner.prepare(task, workdir, overlay)
            g = runner.grade(task, workdir)
            if g.passed != should_pass:
                want = "pass" if should_pass else "fail"
                failures.append(f"{task.name}/{label}: expected {want}, grader said:\n{g.output}")
    return failures


def _selftest_patch(task: Task) -> list[str]:
    outside = sorted(p for p in catalyst.touched(task.tests_patch) if not catalyst.is_protected(p))
    if outside:
        return [f"{task.name}: tests.patch touches paths outside {catalyst.PROTECTED}: {outside}"]
    if not task.broken_patches:
        return [f"{task.name}: no broken*.patch"]
    expected = {"setup": (None, False), "reference": (task.reference_patch, True)}
    expected |= {name: (patch, False) for name, patch in task.broken_patches.items()}
    failures = []
    for label, (patch, should_pass) in expected.items():
        g = runner.grade_patch(task, patch)
        if g.passed != should_pass:
            want = "pass" if should_pass else "fail"
            failures.append(f"{task.name}/{label}: expected {want}, grader said:\n{g.output}")
    return failures


def selftest_task(task: Task) -> list[str]:
    """Return a list of failures. The grader must reject the start state and every broken
    answer, and accept the reference."""
    return _selftest_dir(task) if task.base is None else _selftest_patch(task)


def cmd_list(tasks: list[Task], _args) -> int:
    for t in tasks:
        print(f"{t.name:32s} {t.base or 'dir':8s} {t.timeout_s:5d} s  {runner.task_sha256(t)[:12]}  {t.summary}")
    return 0


def cmd_selftest(tasks: list[Task], _args) -> int:
    failures = [f for t in tasks for f in selftest_task(t)]
    for f in failures:
        print(f"FAIL {f}")
    print(f"{len(tasks)} tasks, {len(failures)} grader failures")
    return 1 if failures else 0


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def _version(cmd: str) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr).strip() else "unknown"


def _run_setup(args) -> tuple[runner.Agent, dict, str]:
    """Pick the agent (an arm, or an ad hoc command for pilots) and the run metadata."""
    dirty = bool(_git("status", "--porcelain"))
    if args.arm:
        arm = arms.load(ARMS)[args.arm]
        agent = runner.Agent(arm.cmd, arm.home, arm.files)
        meta = {"arm": arm.name, "harness": arm.harness, "model": arm.model, "effort": arm.effort,
                "cli_version": _version(arm.version_cmd)}
        label = args.label or f"{arm.name}_{arm.model}"
    else:
        agent = runner.Agent(args.agent_cmd, args.agent_home.resolve(), tuple(f.resolve() for f in args.agent_file))
        meta = {"arm": None}
        label = args.label
    meta |= {"tooleval_commit": _git("rev-parse", "HEAD"), "tooleval_dirty": dirty, "pilot": args.pilot,
             "host": socket.gethostname()}
    return agent, meta, label


def _not_regular(directory: str, names: list[str]) -> list[str]:
    """copytree ignore hook: skip sockets and FIFOs (Cursor leaves a worker.sock)."""
    def special(name: str) -> bool:
        mode = os.lstat(os.path.join(directory, name)).st_mode
        return stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode)
    return [n for n in names if special(n)]


def _backup_task(run_dir: Path, task: str, dest: Path) -> None:
    """Copy one finished task dir, then the run files. Each task dir is copied once,
    because it holds read-only mount points that bwrap leaves behind (empty files)."""
    shutil.copytree(run_dir / task, dest / task, symlinks=True, ignore=_not_regular)
    for name in ("run.json", "results.jsonl"):
        shutil.copy2(run_dir / name, dest / name)


def cmd_run(tasks: list[Task], args) -> int:
    agent, meta, label = _run_setup(args)
    if not args.pilot and (meta["tooleval_dirty"] or meta["arm"] is None):
        why = "the tooleval tree has uncommitted changes" if meta["tooleval_dirty"] else "no --arm given"
        print(f"refusing a counted run: {why}. Commit first, use --arm, or pass --pilot.")
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS / f"{stamp}_{label}{'_pilot' if args.pilot else ''}"
    run_dir.mkdir(parents=True)
    meta["started_utc"] = stamp
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1) + "\n")
    backup_dir = args.backup or os.environ.get("TOOLEVAL_BACKUP")
    backup = Path(backup_dir) if backup_dir else None
    passed = 0
    with (run_dir / "results.jsonl").open("w") as out:
        for t in tasks:
            r = runner.run(t, agent, run_dir, meta)
            out.write(json.dumps(asdict(r)) + "\n")
            out.flush()
            passed += r.passed
            status = "TIMEOUT" if r.timed_out else ("PASS" if r.passed else "FAIL")
            flag = "  (diff touches mlir/test)" if r.touched_protected else ""
            print(f"{status:7s} {t.name}  ({r.agent_wall_s} s){flag}")
            if backup:
                _backup_task(run_dir, t.name, backup / run_dir.name)
    print(f"{passed}/{len(tasks)} passed; results in {run_dir}" + (f", copied to {backup / run_dir.name}" if backup else ""))
    return 0


def cmd_slots(_tasks, args) -> int:
    for slot in (catalyst.TRIAL, catalyst.GRADE):
        if slot.exists():
            print(f"{slot.name}: ready at {slot.tree}")
        elif args.init:
            print(f"{slot.name}: creating at {slot.tree} (about 6 min)", flush=True)
            catalyst.create(slot)
            print(f"{slot.name}: ready")
        else:
            print(f"{slot.name}: missing (run: python -m tooleval slots --init)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="tooleval")
    p.add_argument("--task", action="append", help="limit to this task (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    who = r.add_mutually_exclusive_group(required=True)
    who.add_argument("--arm", help="an arm from arms.toml (counted runs need one)")
    who.add_argument("--agent-cmd", help="ad hoc shell command (pilot only); prompt path is in $TOOLEVAL_PROMPT")
    r.add_argument("--label", help="name for this run (default: <arm>_<model>)")
    r.add_argument("--pilot", action="store_true", help="mark the run as a pilot; allows a dirty tree or --agent-cmd")
    r.add_argument("--backup", help="copy the run dir here after each task (default: $TOOLEVAL_BACKUP)")
    r.add_argument(
        "--agent-home",
        type=Path,
        help="with --agent-cmd: template dir whose entries are mounted into $HOME (fresh copy per trial)",
    )
    r.add_argument(
        "--agent-file",
        action="append",
        default=[],
        type=Path,
        help="file mounted read-write at its own path in the sandbox, e.g. an auth token (repeatable)",
    )
    s = sub.add_parser("slots", help="show or create the Catalyst slots (desktop)")
    s.add_argument("--init", action="store_true", help="create missing slots")
    args = p.parse_args()
    if args.cmd == "run" and args.agent_cmd and not (args.agent_home and args.label):
        p.error("--agent-cmd needs --agent-home and --label")

    tasks = discover(TASKS)
    if args.task:
        unknown = set(args.task) - {t.name for t in tasks}
        if unknown:
            p.error(f"unknown task(s): {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t.name in args.task]

    handlers = {"list": cmd_list, "selftest": cmd_selftest, "run": cmd_run, "slots": cmd_slots}
    return handlers[args.cmd](tasks, args)


if __name__ == "__main__":
    sys.exit(main())
