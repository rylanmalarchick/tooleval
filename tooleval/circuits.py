"""Write small quantum-dialect functions for graders' hidden inputs.

A gate is (name, params, wires, adjoint). params are SSA names: "%argN" for an
angle the grader draws, a constant name from `consts`, or "%zero". The function
returns its qubits in register order.
"""

ZERO = "%zero"


def arg(i: int) -> str:
    return f"%arg{i}"


def gate(name, *params, q=(0,), adj=False):
    return (name, list(params), list(q), adj)


def render(n_qubits: int, gates: list, consts: dict[str, float]) -> str:
    used = sorted({int(p[4:]) for _, ps, _, _ in gates for p in ps if p.startswith("%arg")})
    assert used == list(range(len(used))), "angle arguments must be %arg0..%argN-1"
    sig = ", ".join(f"%arg{i}: f64" for i in used)
    rtype = ", ".join(["!quantum.bit"] * n_qubits)
    lines = [f"func.func @circuit({sig}) -> ({rtype}) {{", f"    {ZERO} = arith.constant 0.000000e+00 : f64"]
    lines += [f"    {c} = arith.constant {v!r} : f64" for c, v in consts.items()
              if any(c in ps for _, ps, _, _ in gates)]
    lines.append(f"    %r = quantum.alloc( {n_qubits}) : !quantum.reg")
    cur = []
    for q in range(n_qubits):
        lines.append(f"    %w{q}_0 = quantum.extract %r[ {q}] : !quantum.reg -> !quantum.bit")
        cur.append(f"%w{q}_0")
    for i, (name, ps, qs, adj) in enumerate(gates):
        res = f"%g{i}"
        ins = ", ".join(cur[q] for q in qs)
        types = ", ".join(["!quantum.bit"] * len(qs))
        attr = " {adjoint}" if adj else ""
        head = res if len(qs) == 1 else f"{res}:{len(qs)}"
        lines.append(f'    {head} = quantum.custom "{name}"({", ".join(ps)}) {ins}{attr} : {types}')
        for k, q in enumerate(qs):
            cur[q] = res if len(qs) == 1 else f"{res}#{k}"
    lines += [f"    return {', '.join(cur)} : {rtype}", "}"]
    return "\n".join(lines) + "\n"
