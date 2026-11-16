def rk4_step(f, t, y, h):
    """Advance y' = f(t, y) from t to t + h. y is a list of floats."""
    def axpy(a, x, b):
        return [bi + a * xi for xi, bi in zip(x, b)]

    k1 = f(t, y)
    k2 = f(t + h / 2, axpy(h / 2, k1, y))
    k3 = f(t + h / 2, axpy(h / 2, k2, y))
    k4 = f(t + h, axpy(h, k3, y))
    return [yi + h / 6 * (a + 2 * b + 2 * c + d) for yi, a, b, c, d in zip(y, k1, k2, k3, k4)]
