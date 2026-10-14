"""Fitting a model's free parameter so its output matches observed history.

The only knob either model has is an effective mean service time. It is tuned
until the model reproduces the mean wait that was measured on the real system
at the staffing level actually in use. This is the ordinary way a simulation
model gets signed off, and the point of the project is to ask what that
sign-off is worth.
"""

import numpy as np

from .systems import replicate, mean_of


def model_mean_wait(run_fn, mean_service, n_servers, n_customers, reps, seed0):
    s = replicate(run_fn, reps, seed0,
                  n_servers=n_servers, n_customers=n_customers,
                  mean_service=mean_service)
    return mean_of(s, "mean_wait")


def calibrate(run_fn, target_wait, n_servers, n_customers=4000, reps=10,
              seed0=900, lo=1.0, hi=6.0, tol=1e-3, max_iter=40):
    """Bisect on the effective mean service time to hit the observed wait.

    Mean wait rises with the service time, so the function is monotone and
    bisection is enough. The bracket is widened downward if even the smallest
    value still overshoots, and the search fails loudly rather than returning
    an endpoint that never matched.
    """
    def f(x):
        return model_mean_wait(run_fn, x, n_servers, n_customers, reps, seed0) - target_wait

    f_lo, f_hi = f(lo), f(hi)
    tries = 0
    while f_lo > 0 and tries < 6:
        hi, f_hi = lo, f_lo
        lo = lo / 2.0
        f_lo = f(lo)
        tries += 1
    if f_lo > 0 or f_hi < 0:
        raise RuntimeError(
            "target wait %.3f not reachable in service range [%.3f, %.3f]"
            % (target_wait, lo, hi))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < 1e-4:
            return mid
        if f_mid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
