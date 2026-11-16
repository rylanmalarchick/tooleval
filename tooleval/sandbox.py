"""bwrap sandbox for agent runs.

Inside the sandbox the agent sees the host root read-only, a private /tmp,
an empty /mnt, and an empty $HOME with only these paths mounted back:
  * the workdir (read-write)
  * ~/.local (agent CLIs, read-only)
  * each extra read-only path (Catalyst venv, base submodules)
  * each top-level entry of the agent home (read-write, a clean agent config)
  * each agent file (read-write, at its own path; e.g. an OAuth token file,
    shared with the host so a token refresh is not lost)
So the tooleval repo, graders, reference patches, other results, and the
git mirrors are not visible. Network stays on: the agents need their APIs.
"""

import os
from collections.abc import Sequence
from pathlib import Path

PROMPT_PATH = "/tmp/tooleval_prompt.txt"


def wrap(
    cmd: str,
    workdir: Path,
    prompt: Path,
    agent_home: Path,
    ro_paths: Sequence[Path] = (),
    path_prefix: Sequence[Path] = (),
    agent_files: Sequence[Path] = (),
) -> list[str]:
    """Return argv that runs shell command `cmd` in the sandbox, cwd = workdir."""
    home = Path.home()
    args = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", "/mnt",
        "--tmpfs", str(home),
    ]
    for p in (home / ".local", *ro_paths):
        args += ["--ro-bind", str(p), str(p)]
    for entry in sorted(agent_home.iterdir()):
        args += ["--bind", str(entry), str(home / entry.name)]
    for f in agent_files:
        args += ["--bind", str(f), str(f)]
    path = os.pathsep.join([*(str(p) for p in path_prefix), str(home / ".local/bin"), os.environ.get("PATH", "")])
    args += [
        "--bind", str(workdir), str(workdir),
        "--ro-bind", str(prompt), PROMPT_PATH,
        "--setenv", "TOOLEVAL_PROMPT", PROMPT_PATH,
        "--setenv", "HOME", str(home),
        "--setenv", "PATH", path,
        "--setenv", "CCACHE_DISABLE", "1",
        "--chdir", str(workdir),
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "bash", "-c", cmd,
    ]
    return args
