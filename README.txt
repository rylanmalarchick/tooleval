tooleval
========

tooleval tests how well coding agents (Claude Code, Cursor, and others) do
real work inside a real compiler. The first target is Catalyst, the quantum
compiler behind PennyLane.

Each task gives the agent a bug to fix or a change to make in the
compiler. When the agent finishes, a grader script checks its change.
Before any grader is trusted, it is run against the known correct answer,
which it must pass, and against a few wrong answers that look right, which
it must fail. So when an agent passes, its change is correct.


Running it
----------

You need Python 3.11 or newer, plus pytest for the tests.

    python -m tooleval list             # show the tasks
    python -m tooleval selftest         # check each grader against the answers
    python -m pytest                    # the same check plus unit tests
    python -m tooleval run --arm A      # run agent setup A on every task

An "arm" is one agent setup: which tool, which model, and how it is
launched. The arms are listed in arms.toml. A normal run only works from a
clean git checkout, so the results always match a known version of the
code. For quick tries on a dirty tree, add --pilot, and the results are
marked as a pilot.

Each run saves the prompt, the agent's full output, its code changes, and
the grade for every task under results/.


Sandbox
-------

The agent runs inside a bwrap sandbox. It can read the system and write
only to its own work folder. It gets a fresh, empty home directory, and it
cannot see the graders or the answers.


Tasks
-----

Each task is a folder under tasks/ with the prompt, the grader, the correct
answer, the wrong answers, and any hidden test inputs.

This repo includes one small example task, smoke-rk4. The real Catalyst
tasks are kept private so their answers do not end up in model training
data. Running Catalyst tasks also needs a local Catalyst build, which
`python -m tooleval slots --init` sets up (about 6 minutes per copy).


Checking the graders
--------------------

Quantum graders compare circuits by computing their unitary matrices
(tooleval/qmlir.py). That code is tested against Qiskit and PyZX on 150
random circuits.

Two circuits that differ only by a global phase are the same circuit, so
the grader removes that phase before comparing. proofs/PhaseDistance.lean
is a Lean proof that the way it picks the phase is the best possible
choice. It was checked with Lean v4.28.0 and Mathlib 8f9d9cff.
