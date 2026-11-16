"""OpenQASM 3 export of a quantum-dialect function, with its angles evaluated.

The interchange form for equivalence checks (D3). Qubit q[i] is register index
i. The function's return order is not part of the QASM: qmlir.unitary applies
it to the output rows, so compare against a function that returns its qubits
in register order, or apply the permutation yourself.

Rot and CRot become rz, ry, rz (controlled: crz, cry, crz). MS(t) and
IsingXX(t) = exp(-i t/2 X⊗X) become h, cx, rz, cx, h. Both rewrites are exact,
including global phase. An adjoint gate becomes the reversed sequence with
negated angles.
"""

import math

from tooleval.qmlir import Trace, Unsupported

_SELF_INVERSE = {"Identity": [], "PauliX": ["x"], "PauliY": ["y"], "PauliZ": ["z"], "Hadamard": ["h"],
                 "CNOT": ["cx"], "CY": ["cy"], "CZ": ["cz"]}
_ADJ_NAME = {"S": ("s", "sdg"), "T": ("t", "tdg")}
_ROTATION = {"RX": "rx", "RY": "ry", "RZ": "rz", "PhaseShift": "p",
             "CRX": "crx", "CRY": "cry", "CRZ": "crz", "ControlledPhaseShift": "cp"}


def _angle(x: float) -> str:
    if not math.isfinite(x):
        raise Unsupported(f"angle {x} cannot be written to QASM")
    return repr(float(x))


def _lines(name: str, params: tuple[float, ...], wires: tuple[int, ...], adjoint: bool) -> list[str]:
    qs = ", ".join(f"q[{w}]" for w in wires)
    if name in _SELF_INVERSE:
        return [f"{g} {qs};" for g in _SELF_INVERSE[name]]
    if name in _ADJ_NAME:
        return [f"{_ADJ_NAME[name][adjoint]} {qs};"]
    sign = -1.0 if adjoint else 1.0
    if name in _ROTATION:
        return [f"{_ROTATION[name]}({_angle(sign * params[0])}) {qs};"]
    if name in ("Rot", "CRot"):
        z, y = ("crz", "cry") if name == "CRot" else ("rz", "ry")
        p, t, w = params
        seq = [(z, p), (y, t), (z, w)]
        if adjoint:
            seq = [(g, -a) for g, a in reversed(seq)]
        return [f"{g}({_angle(a)}) {qs};" for g, a in seq]
    if name in ("MS", "IsingXX"):
        a, b = (f"q[{w}]" for w in wires)
        t = _angle(sign * params[0])
        return [f"h {a};", f"h {b};", f"cx {a}, {b};", f"rz({t}) {b};", f"cx {a}, {b};", f"h {a};", f"h {b};"]
    raise Unsupported(f"no QASM form for gate {name!r}")


def to_qasm3(tr: Trace) -> str:
    out = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{tr.n}] q;"]
    for op in tr.ops:
        out += _lines(op.name, op.params, op.wires, op.adjoint)
    return "\n".join(out) + "\n"
