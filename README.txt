tooleval
========

Capability evals for coding agents that work inside a real tool. The first
target is the Catalyst quantum compiler. An agent gets a task, works in a
sandbox, and a grader checks the result.

A grader counts only after it passes the task's reference answer and fails
each of its plausible wrong answers.


Run
---

Python 3.11+. Tests need pytest. Catalyst graders need numpy.

    python -m tooleval list             # list tasks
    python -m tooleval selftest         # check each grader
    python -m pytest                    # selftest plus unit tests
    python -m tooleval run --arm A      # counted run of arm A
    python -m tooleval --task NAME run --arm A --pilot

arms.toml sets each arm's command, model, and effort. A counted run needs
a clean tree. Other runs need --pilot.

Each run writes results/<time>_<label>/ with run.json (arm, model, commit,
host), results.jsonl (one line per task), and per task the prompt, agent
output, and diff.


Sandbox
-------

The agent runs under bwrap with a read-only host, a writable workdir, and
an empty $HOME holding a fresh agent config. It cannot see the graders or
the reference answers.


Tasks
-----

    tasks/<name>/
        task.toml          timeout and base
        prompt.txt         what the agent is told
        grade.py           exit 0 = pass
        reference/         passing answer (patch tasks: reference.patch)
        broken/            wrong answers (patch tasks: broken*.patch)
        hidden/            grader inputs the agent never sees

This repo ships one task, smoke-rk4 (one RK4 step on a damped oscillator).
The Catalyst tasks are private, so their answers stay out of training data.
Catalyst tasks need a built Catalyst tree. Run `python -m tooleval slots
--init` first (about 6 min per slot).


Graders
-------

tooleval/qmlir.py computes the unitary of a small quantum-dialect function.
tests/test_qasm_differential.py checks it against Qiskit and PyZX on 150
random circuits.

proofs/PhaseDistance.lean proves that the global phase the grader removes
is optimal. Checked with Lean v4.28.0 and Mathlib 8f9d9cff. From a Lake
project with that Mathlib:

    lake env lean /path/to/tooleval/proofs/PhaseDistance.lean
