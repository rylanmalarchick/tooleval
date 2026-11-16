"""The circuit evaluator that graders use. All checks are exact or at float64 round-off."""

import numpy as np
import pytest

from tooleval import qmlir

CNOT_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT_10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


def circuit(n, body, args="", returns=None):
    returns = returns or ", ".join(f"%w{i}" for i in range(n))
    rtype = ", ".join(["!quantum.bit"] * n)
    extracts = "\n".join(f"  %w{i}_0 = quantum.extract %r[ {i}] : !quantum.reg -> !quantum.bit" for i in range(n))
    return (
        f"func.func @c({args}) -> ({rtype}) {{\n  %r = quantum.alloc( {n}) : !quantum.reg\n{extracts}\n"
        f"{body}\n  return {returns} : {rtype}\n}}\n"
    )


def test_control_is_first_operand_and_wire0_is_msb():
    fwd = circuit(2, '  %g:2 = quantum.custom "CNOT"() %w0_0, %w1_0 : !quantum.bit, !quantum.bit', returns="%g#0, %g#1")
    rev = circuit(2, '  %g:2 = quantum.custom "CNOT"() %w1_0, %w0_0 : !quantum.bit, !quantum.bit', returns="%g#1, %g#0")
    assert np.array_equal(qmlir.unitary(fwd, []), CNOT_01)
    assert np.array_equal(qmlir.unitary(rev, []), CNOT_10)


def test_result_k_continues_operand_k():
    body = (
        '  %g:2 = quantum.custom "CNOT"() %w0_0, %w1_0 : !quantum.bit, !quantum.bit\n'
        '  %h = quantum.custom "Hadamard"() %g#0 : !quantum.bit'
    )
    had = qmlir.GATES["Hadamard"]()
    want = np.kron(had, np.eye(2)) @ CNOT_01
    assert np.allclose(qmlir.unitary(circuit(2, body, returns="%h, %g#1"), []), want, rtol=0, atol=1e-15)


def test_return_order_permutes_outputs_only():
    swapped = circuit(2, '  %g:2 = quantum.custom "CNOT"() %w0_0, %w1_0 : !quantum.bit, !quantum.bit', returns="%g#1, %g#0")
    swap = np.eye(4)[[0, 2, 1, 3]]
    assert np.array_equal(qmlir.unitary(swapped, []), swap @ CNOT_01)


def test_scalar_ops_and_adjoint():
    body = (
        "  %h = arith.constant 5.000000e-01 : f64\n"
        "  %t = arith.mulf %arg0, %h : f64\n"
        "  %a = math.atan2 %t, %h : f64\n"
        '  %g = quantum.custom "RZ"(%a) %w0_0 {adjoint} : !quantum.bit'
    )
    x = 1.3
    a = np.arctan2(x * 0.5, 0.5)
    want = np.diag([np.exp(0.5j * a), np.exp(-0.5j * a)])
    assert np.allclose(qmlir.unitary(circuit(1, body, "%arg0: f64", "%g"), [x]), want, rtol=0, atol=1e-15)


def test_sign_flip_is_global_on_one_qubit_but_not_under_control():
    rng = np.random.default_rng(0)
    u = qmlir.GATES["Rot"](*rng.uniform(-3, 3, 3))
    assert qmlir.phase_distance(u, -u) < 1e-15
    cu, cneg = np.eye(4, dtype=complex), np.eye(4, dtype=complex)
    cu[2:, 2:], cneg[2:, 2:] = u, -u
    assert qmlir.phase_distance(cu, cneg) == pytest.approx(2 * np.sqrt(2), abs=1e-12)


def test_unknown_op_raises():
    body = '  %g = quantum.custom "Toffoli"() %w0_0 : !quantum.bit'
    with pytest.raises(qmlir.Unsupported):
        qmlir.unitary(circuit(1, body, returns="%g"), [])
    with pytest.raises(qmlir.Unsupported):
        qmlir.unitary(circuit(1, "  %x = arith.remf %w0_0, %w0_0 : f64", returns="%w0_0"), [])
    with pytest.raises(qmlir.Unsupported, match="undefined value"):
        qmlir.unitary(circuit(1, "  %x = arith.maximumf %nope, %nope : f64", returns="%w0_0"), [])


def test_folded_nan_constant_and_guard_ops():
    # what a folded atan2(0, 0) prints, and a select guard over it
    body = (
        "  %nan = arith.constant 0xFFF8000000000000 : f64\n"
        "  %z = arith.constant 0.000000e+00 : f64\n"
        "  %c = arith.cmpf uno, %nan, %nan : f64\n"
        "  %t = arith.constant true\n"
        "  %b = arith.andi %c, %t : i1\n"
        "  %m = arith.minimumf %nan, %z : f64\n"
        "  %a = arith.select %b, %z, %m : f64\n"
        '  %g = quantum.custom "RZ"(%a) %w0_0 : !quantum.bit\n'
        '  %h = quantum.custom "RZ"(%m) %g : !quantum.bit'
    )
    assert np.isnan(qmlir.unitary(circuit(1, body, returns="%h"), [])).any()  # unguarded NaN angle
    guarded = body.rsplit("\n", 1)[0]
    assert np.array_equal(qmlir.unitary(circuit(1, guarded, returns="%g"), []), np.eye(2))
