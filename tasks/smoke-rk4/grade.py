"""Grade solver.rk4_step on x'' + 2*zeta*w*x' + w^2*x = 0 against the exact solution.

Checks the final-time error at h = 0.01 and the observed convergence order
from h = 0.02 -> 0.01. Exit 0 on pass.
"""

import json
import math
import sys

sys.path.insert(0, sys.argv[1])
from solver import rk4_step  # noqa: E402

W, ZETA, X0, V0, T_END = 2.0, 0.1, 1.0, 0.0, 10.0
MAX_ERR = 1e-6
ORDER_RANGE = (3.8, 4.2)


def f(_t, y):
    x, v = y
    return [v, -2 * ZETA * W * v - W * W * x]


def exact(t):
    wd = W * math.sqrt(1 - ZETA * ZETA)
    return math.exp(-ZETA * W * t) * (X0 * math.cos(wd * t) + (V0 + ZETA * W * X0) / wd * math.sin(wd * t))


def final_error(h):
    n = round(T_END / h)
    y = [X0, V0]
    for i in range(n):
        y = rk4_step(f, i * h, y, h)
    return abs(y[0] - exact(n * h))


e_coarse, e_fine = final_error(0.02), final_error(0.01)
order = math.log2(e_coarse / e_fine)
passed = e_fine < MAX_ERR and ORDER_RANGE[0] < order < ORDER_RANGE[1]
print(json.dumps({"passed": passed, "err_h0.01": e_fine, "order": round(order, 3)}))
sys.exit(0 if passed else 1)
