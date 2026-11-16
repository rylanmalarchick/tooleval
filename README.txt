tooleval
========

A harness for capability evals where an agent works inside a real tool
(first target: the Catalyst quantum compiler) and an automatic grader
checks the result. Each task ships a reference answer and plausible wrong
answers. The grader must accept the first and reject the others before any
agent result counts.


Run
---

Python 3.11 or later. The harness uses the standard library. Graders for
Catalyst tasks use numpy. Tests use pytest.

    python -m tooleval list                  # tasks, base, timeouts
    python -m tooleval selftest              # check every grader
    python -m pytest                         # same check, plus unit tests
    python -m tooleval run --arm A           # counted run of arm A on every task
    python -m tooleval --task NAME run --arm B --pilot

arms.toml holds each arm's exact command, model, effort, agent-home
template, and token files. A counted run needs an arm and a clean tooleval
tree. Anything else must pass --pilot, and its run dir ends in _pilot.

"run" writes results/<time>_<label>/: run.json (arm, model, effort, CLI
version, tooleval commit, pilot flag, host), results.jsonl (one line per
task, with the task's sha256), and per task the prompt, agent.stdout,
agent.stderr, the trial's agent-home, and (patch tasks) agent.diff. With
$TOOLEVAL_BACKUP set, the run dir is copied there after each task.

The agent runs in a bwrap sandbox (tooleval/sandbox.py). It sees the host
root read-only, its workdir read-write, ~/.local, and a fresh copy of the
arm's agent-home template mounted into an empty $HOME. The tooleval repo,
graders, and reference answers are not visible. Network stays on. The
template holds a clean agent config (for claude: an empty .claude/). Token
files are mounted at their own paths, so a token refresh reaches the host.


Catalyst tasks (desktop only)
-----------------------------

Catalyst tasks need the base tree built at the pin in tooleval/catalyst.py,
under $TOOLEVAL_WORK (default ~/dev/projects/tooleval-work/catalyst), and
two slots made from it:

    python -m tooleval slots --init          # about 6 min per slot

A slot is a copy of the tree at the pin with its own build. Its git history
lives outside the tree, so the agent sees a one-commit repo. "trial-0" runs
the agent. "grade" applies the agent's diff to a clean tree, resets
mlir/test, and runs the grader.


Task layout
-----------

    tasks/<name>/
        task.toml       summary, timeout_s, optional base, grade_timeout_s
        prompt.txt      what the agent is told
        grade.py        python grade.py WORKDIR; exit 0 = pass

    directory task (no base):
        workspace/      starting files
        reference/      files that make the grader pass
        broken/         a plausible wrong answer

    patch task (base = "catalyst"):
        setup.patch     start state on the pin (may be empty; a mined task
                        reverts the fix and its test)
        reference.patch the grader must pass it
        broken*.patch   plausible wrong answers
        tests.patch     optional: grading copy of mlir/test changes
        hidden/         grader inputs the agent never sees

selftest grades the start state (must fail), the reference (must pass),
and each broken answer (must fail).

tooleval/qmlir.py evaluates the unitary of a small quantum-dialect function
for equivalence graders. It raises on any op it does not know.
tooleval/qasm.py writes the same function as OpenQASM 3 with its angles
evaluated. tests/test_qasm_differential.py checks qmlir against qiskit
(exact) and pyzx (up to global phase) on 150 seeded random circuits. It
needs qiskit, qiskit_qasm3_import, and pyzx, and skips where they are
missing (the desktop's Catalyst venv).

proofs/PhaseDistance.lean proves that the global phase phase_distance
removes is optimal (exact arithmetic, any complex inner product space). It
was checked with Lean v4.28.0 and Mathlib 8f9d9cff. To check it, run from
any Lake project with that Mathlib:

    lake env lean /path/to/tooleval/proofs/PhaseDistance.lean


Tasks
-----

smoke-rk4                     tests the harness itself: one RK4 step on a
                              damped oscillator.

The Catalyst tasks (patch tasks on the pinned compiler) are kept in a
private repo, so that their references, wrong answers, and hidden grader
inputs stay out of public training data. This repo ships smoke-rk4 only.
