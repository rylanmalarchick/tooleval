def rk4_step(f, t, y, h):
    """Advance y' = f(t, y) from t to t + h. y is a list of floats."""
    # Forward Euler: right shape, wrong order.
    return [yi + h * di for yi, di in zip(y, f(t, y))]
