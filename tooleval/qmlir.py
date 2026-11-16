"""Unitary of a small quantum-dialect function, for graders.

Reads the textual MLIR that quantum-opt prints for a single func.func whose
f64 arguments are gate angles. The function allocates one register, extracts
its qubits, applies gates, and returns qubits. Supports arith/math scalar ops
and the gates in GATES (one-qubit), TWO_QUBIT, and CONTROLLED (two-qubit,
control first).
Any other op raises Unsupported, so a grader never silently skips part of a
circuit.

Gate conventions (PennyLane): RX(t) = exp(-i t X/2), RY, RZ likewise,
PhaseShift(p) = diag(1, e^{ip}), Rot(p, t, w) = RZ(w) RY(t) RZ(p).
Checked against pennylane.matrix (max deviation 3e-16, 2026-10-18), and the
Rot closed form against RZ RY RZ in Wolfram.
"""

import math
import re
import struct
from dataclasses import dataclass

import numpy as np

_SSA = r"%[\w.]+(?:#\d+)?"
_FM = r"(?: fastmath<[\w,]+>)?"
_FUNC = re.compile(r"func\.func @(\w+)\((.*?)\)")
_ARG = re.compile(rf"({_SSA}): f64")
_CONST = re.compile(rf"^({_SSA}) = arith\.constant (0x[0-9A-Fa-f]{{16}}|[-+0-9.eE]+|inf|-inf|nan) : f64$")
_BOOL = re.compile(rf"^({_SSA}) = arith\.constant (true|false)$")
_BINARY = re.compile(rf"^({_SSA}) = (arith\.\w+|math\.\w+) ({_SSA}), ({_SSA}){_FM} : f64$")
_UNARY = re.compile(rf"^({_SSA}) = (arith\.negf|math\.\w+) ({_SSA}){_FM} : f64$")
_CMPF = re.compile(rf"^({_SSA}) = arith\.cmpf (\w+), ({_SSA}), ({_SSA}){_FM} : f64$")
_LOGIC = re.compile(rf"^({_SSA}) = arith\.(andi|ori|xori) ({_SSA}), ({_SSA}) : i1$")
_SELECT = re.compile(rf"^({_SSA}) = arith\.select ({_SSA}), ({_SSA}), ({_SSA}) : f64$")
_ALLOC = re.compile(rf"^({_SSA}) = quantum\.alloc\( ?(\d+)\) : !quantum\.reg$")
_EXTRACT = re.compile(rf"^({_SSA}) = quantum\.extract ({_SSA})\[ ?(\d+)\] : !quantum\.reg -> !quantum\.bit$")
_CUSTOM = re.compile(
    rf'^(%[\w.]+)(?::(\d+))? = quantum\.custom "(\w+)"\(((?:{_SSA}(?:, )?)*)\) '
    rf"((?:{_SSA}(?:, )?)+?)( adj)?(?: \{{adjoint\}})?\s*: [!\w., ]+$"
)
_RETURN = re.compile(rf"^return ((?:{_SSA}(?:, )?)+) : [!\w., ]+$")


def _ieee_min(a, b):  # arith.minimumf: NaN if either is NaN
    return math.nan if math.isnan(a) or math.isnan(b) else min(a, b)


def _ieee_max(a, b):
    return math.nan if math.isnan(a) or math.isnan(b) else max(a, b)


def _num_min(a, b):  # arith.minnumf: the other operand if one is NaN
    return b if math.isnan(a) else a if math.isnan(b) else min(a, b)


def _num_max(a, b):
    return b if math.isnan(a) else a if math.isnan(b) else max(a, b)


def _safe(f):
    """libm-style: a domain error gives NaN, as the compiled code would."""
    def g(*xs):
        try:
            return f(*xs)
        except (ValueError, ZeroDivisionError):
            return math.nan
    return g


_BIN_OPS = {
    "arith.addf": float.__add__,
    "arith.subf": float.__sub__,
    "arith.mulf": float.__mul__,
    "arith.divf": lambda a, b: a / b if b != 0 else (math.copysign(math.inf, a) * math.copysign(1, b) if a != 0 and not math.isnan(a) else math.nan),
    "arith.minimumf": _ieee_min,
    "arith.maximumf": _ieee_max,
    "arith.minnumf": _num_min,
    "arith.maxnumf": _num_max,
    "math.atan2": math.atan2,
    "math.copysign": math.copysign,
    "math.powf": _safe(math.pow),
}
_UN_OPS = {
    "arith.negf": float.__neg__,
    "math.absf": abs,
    "math.cos": _safe(math.cos),
    "math.sin": _safe(math.sin),
    "math.tan": _safe(math.tan),
    "math.sqrt": _safe(math.sqrt),
    "math.acos": _safe(math.acos),
    "math.asin": _safe(math.asin),
    "math.atan": math.atan,
    "math.exp": _safe(math.exp),
    "math.log": _safe(math.log),
}
# arith.cmpf predicates: ordered ones are false on NaN, unordered ones true.
_CMP_OPS = {
    "oeq": lambda a, b: a == b, "one": lambda a, b: a < b or a > b,
    "olt": lambda a, b: a < b, "ole": lambda a, b: a <= b,
    "ogt": lambda a, b: a > b, "oge": lambda a, b: a >= b,
    "ord": lambda a, b: not (math.isnan(a) or math.isnan(b)),
    "ueq": lambda a, b: math.isnan(a) or math.isnan(b) or a == b,
    "une": lambda a, b: not a == b,
    "ult": lambda a, b: not a >= b, "ule": lambda a, b: not a > b,
    "ugt": lambda a, b: not a <= b, "uge": lambda a, b: not a < b,
    "uno": lambda a, b: math.isnan(a) or math.isnan(b),
    "true": lambda a, b: True, "false": lambda a, b: False,
}
_LOGIC_OPS = {"andi": bool.__and__, "ori": bool.__or__, "xori": bool.__xor__}


def _float(literal: str) -> float:
    if literal.startswith("0x"):
        return struct.unpack(">d", bytes.fromhex(literal[2:]))[0]
    return float(literal)


class Unsupported(ValueError):
    pass


def _rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _rot(p, t, w):
    return _rz(w) @ _ry(t) @ _rz(p)


GATES = {
    "Identity": lambda: np.eye(2, dtype=complex),
    "PauliX": lambda: np.array([[0, 1], [1, 0]], dtype=complex),
    "PauliY": lambda: np.array([[0, -1j], [1j, 0]]),
    "PauliZ": lambda: np.diag([1, -1]).astype(complex),
    "Hadamard": lambda: np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "S": lambda: np.diag([1, 1j]),
    "T": lambda: np.diag([1, np.exp(0.25j * math.pi)]),
    "RX": _rx,
    "RY": _ry,
    "RZ": _rz,
    "PhaseShift": lambda p: np.diag([1, np.exp(1j * p)]),
    "Rot": _rot,
}
def _ising_xx(t):
    xx = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
    return math.cos(t / 2) * np.eye(4) - 1j * math.sin(t / 2) * xx


# Two-qubit gates that are not controlled gates. MS is the trapped-ion
# Molmer-Sorensen gate as Catalyst's ions-decomposition uses it:
# MS(t) = exp(-i t/2 X(x)X) = IsingXX(t).
TWO_QUBIT = {"MS": _ising_xx, "IsingXX": _ising_xx}

CONTROLLED = {
    "CNOT": "PauliX",
    "CY": "PauliY",
    "CZ": "PauliZ",
    "CRX": "RX",
    "CRY": "RY",
    "CRZ": "RZ",
    "ControlledPhaseShift": "PhaseShift",
    "CRot": "Rot",
}


def _func_body(text: str) -> tuple[list[str], list[str]]:
    """Return (argument names, body lines) of the only func.func in text."""
    funcs = list(_FUNC.finditer(text))
    if len(funcs) != 1:
        raise Unsupported(f"expected one func.func, found {len(funcs)}")
    args = _ARG.findall(funcs[0].group(2))
    body = text[funcs[0].end():]
    lines = [ln.split("//")[0].strip() for ln in body.splitlines()]
    return args, [ln for ln in lines if ln and ln not in ("{", "}") and not ln.endswith("{")]


def _embed(gate: np.ndarray, wires: list[int], n: int) -> np.ndarray:
    """Full 2^n matrix of `gate` acting on `wires` (wire 0 is the most significant bit)."""
    k = len(wires)
    others = [w for w in range(n) if w not in wires]
    full = np.kron(gate, np.eye(2 ** (n - k)))
    # full acts on order wires + others; permute axes back to 0..n-1
    order = wires + others
    t = full.reshape([2] * (2 * n))
    perm = [order.index(w) for w in range(n)]
    t = t.transpose(perm + [n + p for p in perm])
    return t.reshape(2**n, 2**n)


@dataclass(frozen=True)
class Op:
    name: str  # Catalyst gate name, e.g. "Rot", "CRot"
    params: tuple[float, ...]
    wires: tuple[int, ...]  # register indices; for controlled gates the control is first
    adjoint: bool


@dataclass(frozen=True)
class Trace:
    n: int  # register size
    ops: tuple[Op, ...]  # in application order
    order: tuple[int, ...]  # order[i] = register index of the i-th returned qubit


class _Eval:
    """Interpreter state for one function: scalar values, i1 values, gates, and the register unitary."""

    def __init__(self, args: list[str], angles: list[float]):
        self.scalars: dict[str, float] = dict(zip(args, map(float, angles)))
        self.flags: dict[str, bool] = {}
        self.wire: dict[str, int] = {}  # qubit SSA value -> register index
        self.ops: list[Op] = []
        self.n, self.reg, self.u = 0, None, None

    def step(self, ln: str) -> tuple[int, ...] | None:
        """Interpret one line. At `return`, give the return order, else None."""
        sc, fl = self.scalars, self.flags
        if m := _CONST.match(ln):
            sc[m[1]] = _float(m[2])
        elif m := _BOOL.match(ln):
            fl[m[1]] = m[2] == "true"
        elif (m := _CMPF.match(ln)) and m[2] in _CMP_OPS:
            fl[m[1]] = _CMP_OPS[m[2]](sc[m[3]], sc[m[4]])
        elif m := _LOGIC.match(ln):
            fl[m[1]] = _LOGIC_OPS[m[2]](fl[m[3]], fl[m[4]])
        elif m := _SELECT.match(ln):
            sc[m[1]] = sc[m[3]] if fl[m[2]] else sc[m[4]]
        elif (m := _BINARY.match(ln)) and m[2] in _BIN_OPS:
            sc[m[1]] = _BIN_OPS[m[2]](sc[m[3]], sc[m[4]])
        elif (m := _UNARY.match(ln)) and m[2] in _UN_OPS:
            sc[m[1]] = _UN_OPS[m[2]](sc[m[3]])
        elif m := _ALLOC.match(ln):
            if self.reg is not None:
                raise Unsupported("more than one register")
            self.reg, self.n = m[1], int(m[2])
            self.u = np.eye(2**self.n, dtype=complex)
        elif (m := _EXTRACT.match(ln)) and m[2] == self.reg:
            self.wire[m[1]] = int(m[3])
        elif m := _CUSTOM.match(ln):
            self._gate(ln, m)
        elif m := _RETURN.match(ln):
            order = [self.wire[q] for q in m[1].split(", ")]
            if sorted(order) != list(range(self.n)):
                raise Unsupported(f"return does not list each register qubit once: {ln}")
            return tuple(order)
        else:
            raise Unsupported(ln)
        return None

    def _gate(self, ln: str, m: re.Match) -> None:
        result, n_results, name, params, operands, adj = m[1], m[2], m[3], m[4], m[5], m[6]
        ins = operands.split(", ")
        vals = [self.scalars[p] for p in params.split(", ") if p]
        if name in GATES and len(ins) == 1:
            gate = GATES[name](*vals)
        elif name in TWO_QUBIT and len(ins) == 2:
            gate = TWO_QUBIT[name](*vals)
        elif name in CONTROLLED and len(ins) == 2:
            gate = np.eye(4, dtype=complex)
            gate[2:, 2:] = GATES[CONTROLLED[name]](*vals)
        else:
            raise Unsupported(f"gate {name!r} on {len(ins)} qubits: {ln}")
        adjoint = bool(adj) or "{adjoint}" in ln
        if adjoint:
            gate = gate.conj().T
        wires = [self.wire[q] for q in ins]
        self.ops.append(Op(name, tuple(vals), tuple(wires), adjoint))
        self.u = _embed(gate, wires, self.n) @ self.u
        outs = [result] if n_results is None else [f"{result}#{i}" for i in range(int(n_results))]
        if len(outs) != len(ins):
            raise Unsupported(f"result count does not match operands: {ln}")
        self.wire.update(zip(outs, wires))


def _run(text: str, angles: list[float]) -> tuple[_Eval, tuple[int, ...]]:
    args, lines = _func_body(text)
    if len(args) != len(angles):
        raise Unsupported(f"function takes {len(args)} angles, got {len(angles)}")
    ev = _Eval(args, angles)
    for ln in lines:
        try:
            order = ev.step(ln)
        except KeyError as e:
            raise Unsupported(f"undefined value {e} in: {ln}") from None
        if order is not None:
            return ev, order
    raise Unsupported("no return")


def trace(text: str, angles: list[float]) -> Trace:
    """The gates the function applies, with angles evaluated (for QASM export)."""
    ev, order = _run(text, angles)
    return Trace(ev.n, tuple(ev.ops), order)


def unitary(text: str, angles: list[float]) -> np.ndarray:
    """Evaluate the function with its f64 arguments bound to angles.

    Return the 2^n matrix from the register (columns: extract index order) to
    the returned qubits (rows: return order). Wire 0 is the most significant bit.
    """
    ev, order = _run(text, angles)
    return _reorder(ev.u, list(order), ev.n)


def _reorder(u: np.ndarray, order: list[int], n: int) -> np.ndarray:
    """Permute the output wires so that output i is the wire returned at position i."""
    t = u.reshape([2] * (2 * n)).transpose(order + list(range(n, 2 * n)))
    return t.reshape(2**n, 2**n)


def n_angles(text: str) -> int:
    return len(_func_body(text)[0])


def gate_count(text: str) -> int:
    return sum(1 for ln in _func_body(text)[1] if _CUSTOM.match(ln))


def phase_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Frobenius distance between v and u after removing the best global phase.

    proofs/PhaseDistance.lean (phase_distance_min) proves, in exact arithmetic,
    that the phase used here minimizes ||v - c u|| over all unit c.
    """
    overlap = np.vdot(u, v)  # tr(u^dag v)
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
    return float(np.linalg.norm(v - phase * u))
