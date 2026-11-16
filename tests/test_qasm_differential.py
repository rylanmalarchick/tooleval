"""Differential test of the grader's evaluator: qmlir.unitary against qiskit and pyzx.

Random quantum-dialect circuits (seeded, so deterministic) are evaluated by
qmlir and exported to OpenQASM 3. qiskit loads the QASM; its matrix must equal
qmlir's exactly, global phase included, since the gate conventions match.
pyzx must agree up to one global phase. Both tolerances are float64 round-off.
"""

import numpy as np
import pytest

from tooleval import qasm, qmlir

qiskit_qasm3 = pytest.importorskip("qiskit.qasm3", reason="qiskit not installed in this env")
pytest.importorskip("qiskit_qasm3_import", reason="qiskit_qasm3_import not installed in this env")
pyzx = pytest.importorskip("pyzx", reason="pyzx not installed in this env")
from qiskit.quantum_info import Operator  # noqa: E402

ONE = {"PauliX": 0, "PauliY": 0, "PauliZ": 0, "Hadamard": 0, "S": 0, "T": 0,
       "RX": 1, "RY": 1, "RZ": 1, "PhaseShift": 1, "Rot": 3}
TWO = {"CNOT": 0, "CY": 0, "CZ": 0, "CRX": 1, "CRY": 1, "CRZ": 1, "ControlledPhaseShift": 1,
       "CRot": 3, "MS": 1, "IsingXX": 1}


def random_circuit(rng, n, depth):
    """A quantum-dialect function on n qubits that returns them in register order, and its angles."""
    lines, cur, angles = [], [f"%w{i}" for i in range(n)], []
    for k in range(depth):
        two = n > 1 and rng.random() < 0.4
        name = rng.choice(sorted(TWO if two else ONE))
        wires = list(rng.choice(n, size=2, replace=False)) if two else [int(rng.integers(n))]
        params = []
        for _ in range((TWO if two else ONE)[name]):
            params.append(f"%arg{len(angles)}")
            angles.append(float(rng.uniform(-np.pi, np.pi)))
        adj = " {adjoint}" if rng.random() < 0.3 else ""
        ins = ", ".join(cur[w] for w in wires)
        types = ", ".join(["!quantum.bit"] * len(wires))
        head = f"%g{k}" if len(wires) == 1 else f"%g{k}:2"
        lines.append(f'  {head} = quantum.custom "{name}"({", ".join(params)}) {ins}{adj} : {types}')
        for j, w in enumerate(wires):
            cur[w] = f"%g{k}" if len(wires) == 1 else f"%g{k}#{j}"
    sig = ", ".join(f"%arg{i}: f64" for i in range(len(angles)))
    rtype = ", ".join(["!quantum.bit"] * n)
    extracts = [f"  %w{i} = quantum.extract %r[ {i}] : !quantum.reg -> !quantum.bit" for i in range(n)]
    text = "\n".join([f"func.func @c({sig}) -> ({rtype}) {{", f"  %r = quantum.alloc( {n}) : !quantum.reg",
                      *extracts, *lines, f"  return {', '.join(cur)} : {rtype}", "}"]) + "\n"
    return text, angles


CASES = [random_circuit(np.random.default_rng(seed), n=1 + seed % 3, depth=12) for seed in range(150)]


def test_cases_cover_every_gate_and_the_adjoint_flag():
    text = "".join(t for t, _ in CASES)
    for name in [*ONE, *TWO]:
        assert f'"{name}"' in text, name
    assert "{adjoint}" in text


@pytest.mark.parametrize("case", range(len(CASES)))
def test_qmlir_matches_qiskit_exactly(case):
    text, angles = CASES[case]
    ours = qmlir.unitary(text, angles)
    circuit = qiskit_qasm3.loads(qasm.to_qasm3(qmlir.trace(text, angles)))
    theirs = Operator(circuit.reverse_bits()).data  # qiskit is little-endian, qmlir big-endian
    assert np.max(np.abs(ours - theirs)) < 1e-12


@pytest.mark.parametrize("case", range(0, len(CASES), 3))
def test_qmlir_matches_pyzx_up_to_phase(case):
    text, angles = CASES[case]
    ours = qmlir.unitary(text, angles)
    theirs = pyzx.Circuit.from_qasm(qasm.to_qasm3(qmlir.trace(text, angles))).to_matrix()
    assert qmlir.phase_distance(ours, theirs) < 1e-9
