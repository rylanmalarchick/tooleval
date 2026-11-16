"""Run metadata: arms pin their model, task versions hash, and each trial gets a fresh agent home."""

import shutil

from tooleval import arms, runner
from tooleval.__main__ import ARMS, TASKS
from tooleval.task import load


def test_every_arm_pins_model_and_reads_the_prompt():
    table = arms.load(ARMS)
    assert table
    for arm in table.values():
        assert arm.model in arm.cmd, arm.name
        assert "$TOOLEVAL_PROMPT" in arm.cmd, arm.name
        assert "{" not in arm.cmd, arm.name  # every placeholder was filled


def test_task_hash_is_stable_and_sees_every_file(tmp_path):
    root = tmp_path / "smoke-rk4"
    shutil.copytree(TASKS / "smoke-rk4", root, ignore=shutil.ignore_patterns("__pycache__"))
    first = runner.task_sha256(load(root))
    assert runner.task_sha256(load(root)) == first
    (root / "reference" / "solver.py").write_text((root / "reference" / "solver.py").read_text() + "\n")
    assert runner.task_sha256(load(root)) != first


def test_fresh_home_leaves_the_template_clean(tmp_path):
    template = tmp_path / "template"
    (template / ".agent").mkdir(parents=True)
    trial = runner.fresh_home(template, tmp_path / "trial-1" / "agent-home")
    (trial / ".agent" / "memory.md").write_text("from trial 1")
    second = runner.fresh_home(template, tmp_path / "trial-2" / "agent-home")
    assert not (template / ".agent" / "memory.md").exists()
    assert not (second / ".agent" / "memory.md").exists()


def test_backup_copies_each_task_once_and_refreshes_run_files(tmp_path):
    from tooleval.__main__ import _backup_task
    run, dest = tmp_path / "run", tmp_path / "backup" / "run"
    for task in ("t1", "t2"):
        (run / task / "agent-home").mkdir(parents=True)
        placeholder = run / task / "agent-home" / "token"
        placeholder.touch()
        placeholder.chmod(0o444)  # what bwrap leaves for a mounted token file
    (run / "run.json").write_text("{}")
    (run / "results.jsonl").write_text("t1\n")
    _backup_task(run, "t1", dest)
    (run / "results.jsonl").write_text("t1\nt2\n")
    _backup_task(run, "t2", dest)
    assert (dest / "results.jsonl").read_text() == "t1\nt2\n"
    assert (dest / "t1" / "agent-home" / "token").exists() and (dest / "t2" / "agent-home" / "token").exists()
