"""The system being studied, and the simplified model that stands in for it.

The system has service times with a long right tail and customers who give up
waiting, which is what service operations actually look like. The model is the
textbook simplification: exponential service, nobody abandons. The model has
one free parameter, an effective service rate, which is what gets calibrated.
"""

import numpy as np

from .des import simulate, summarise

MEAN_SERVICE = 6.0
SERVICE_CV = 1.8
MEAN_PATIENCE = 12.0
BASE_RATE = 1.0
PEAK_RATE = 1.6
PERIOD = 480.0


def rate_fn(t):
    """Arrival rate that rises and falls across a working day."""
    return BASE_RATE + (PEAK_RATE - BASE_RATE) * 0.5 * (
        1.0 - np.cos(2.0 * np.pi * t / PERIOD))


RATE_MAX = PEAK_RATE


def lognormal_sampler(mean, cv):
    """Lognormal service times with the given mean and coefficient of variation."""
    sigma2 = np.log(1.0 + cv ** 2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return lambda rng: float(rng.lognormal(mu, sigma))


def exponential_sampler(mean):
    return lambda rng: float(rng.exponential(mean))


def run_system(n_servers, n_customers, rng):
    """The real system: heavy-tailed service, impatient customers."""
    return simulate(
        n_customers, n_servers, rate_fn, RATE_MAX,
        lognormal_sampler(MEAN_SERVICE, SERVICE_CV),
        exponential_sampler(MEAN_PATIENCE),
        rng)


def run_model_noaband(n_servers, n_customers, rng, mean_service):
    """Model A: exponential service, and nobody ever gives up waiting."""
    return simulate(
        n_customers, n_servers, rate_fn, RATE_MAX,
        exponential_sampler(mean_service),
        None,
        rng)


def run_model_aband(n_servers, n_customers, rng, mean_service):
    """Model B: exponential service, but customers do abandon.

    Still wrong about the shape of the service distribution, so this is not
    the truth with a different label on it. The one thing it adds over model A
    is the mechanism by which an overloaded queue relieves itself.
    """
    return simulate(
        n_customers, n_servers, rate_fn, RATE_MAX,
        exponential_sampler(mean_service),
        exponential_sampler(MEAN_PATIENCE),
        rng)


def replicate(fn, reps, seed0, **kwargs):
    """Run something several times and return the summaries."""
    out = []
    for k in range(reps):
        rng = np.random.default_rng(seed0 + k)
        out.append(summarise(fn(rng=rng, **kwargs)))
    return out


def mean_of(summaries, key):
    return float(np.mean([s[key] for s in summaries]))


def sd_of(summaries, key):
    return float(np.std([s[key] for s in summaries]))
