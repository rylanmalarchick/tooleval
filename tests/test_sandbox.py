"""The agent sandbox hides the repo and the host home, and keeps the mounts it promises."""

import shutil
import subprocess

import pytest

from tooleval import sandbox
from tooleval.runner import REPO


def _bwrap_works() -> bool:
    if shutil.which("bwrap") is None:
        return False
    return subprocess.run(["bwrap", "--ro-bind", "/", "/", "true"], capture_output=True).returncode == 0


# Ubuntu's apparmor_restrict_unprivileged_userns blocks bwrap on the laptop. Agent runs use the desktop.
pytestmark = pytest.mark.skipif(not _bwrap_works(), reason="bwrap cannot create a user namespace on this host")


def probe(tmp_path, script, agent_files=()):
    workdir = tmp_path / "work"
    workdir.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the task\n")
    home = tmp_path / "agent-home"
    (home / ".agentrc").mkdir(parents=True)
    argv = sandbox.wrap(script, workdir, prompt, home, agent_files=agent_files)
    return subprocess.run(argv, capture_output=True, text=True, timeout=60), workdir


def test_repo_and_mirrors_are_hidden(tmp_path):
    proc, _ = probe(tmp_path, f'test -e "{REPO}/tasks" && echo REPO; ls -A /mnt; true')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_workdir_prompt_and_agent_home(tmp_path):
    proc, workdir = probe(tmp_path, 'cat "$TOOLEVAL_PROMPT"; touch out.txt; ls -A ~')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == ["do the task", ".agentrc", ".local"]
    assert (workdir / "out.txt").is_file()


def test_agent_file_write_reaches_host(tmp_path):
    token = tmp_path / "token.json"
    token.write_text("old")
    proc, _ = probe(tmp_path, f'echo new > "{token}"', agent_files=[token])
    assert proc.returncode == 0, proc.stderr
    assert token.read_text() == "new\n"


def test_host_root_is_read_only(tmp_path):
    proc, _ = probe(tmp_path, "touch /usr/tooleval_probe")
    assert proc.returncode != 0
    assert "Read-only file system" in proc.stderr
