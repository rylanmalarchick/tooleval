"""Agent arms from arms.toml: the exact command, model, effort, and auth files of each arm."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from tooleval import catalyst

FIELDS = ("harness", "model", "effort", "home", "files", "version_cmd", "cmd")


@dataclass(frozen=True)
class Arm:
    name: str
    harness: str
    model: str
    effort: str
    cmd: str  # with {model} and {effort} filled in
    home: Path  # template agent home
    files: tuple[Path, ...]
    version_cmd: str


def load(path: Path) -> dict[str, Arm]:
    arms = {}
    for name, spec in tomllib.loads(path.read_text()).items():
        missing = [f for f in FIELDS if f not in spec]
        if missing:
            raise ValueError(f"arm {name}: missing {missing}")
        arms[name] = Arm(
            name=name,
            harness=spec["harness"],
            model=spec["model"],
            effort=spec["effort"],
            cmd=spec["cmd"].format(model=spec["model"], effort=spec["effort"]),
            home=catalyst.WORK / spec["home"],
            files=tuple(Path(f).expanduser() for f in spec["files"]),
            version_cmd=spec["version_cmd"],
        )
    return arms
