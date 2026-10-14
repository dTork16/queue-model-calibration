import numpy as np
import pytest

from src.des import summarise
from src.systems import (lognormal_sampler, exponential_sampler, rate_fn,
                         run_system, run_model_noaband, run_model_aband,
                         replicate, mean_of, MEAN_SERVICE, SERVICE_CV,
                         BASE_RATE, PEAK_RATE)
from src.calibrate import calibrate


def test_lognormal_sampler_has_the_mean_and_spread_it_was_asked_for():
    rng = np.random.default_rng(0)
    s = np.array([lognormal_sampler(6.0, 1.8)(rng) for _ in range(40000)])
    assert s.mean() == pytest.approx(6.0, rel=0.05)
    assert s.std() / s.mean() == pytest.approx(1.8, rel=0.1)


def test_exponential_sampler_has_the_right_mean_and_unit_spread():
    rng = np.random.default_rng(1)
    s = np.array([exponential_sampler(6.0)(rng) for _ in range(40000)])
    assert s.mean() == pytest.approx(6.0, rel=0.05)
    assert s.std() / s.mean() == pytest.approx(1.0, rel=0.05)


def test_the_two_service_distributions_share_a_mean_but_not_a_shape():
    """The models are wrong about the shape, not about the average."""
    rng = np.random.default_rng(2)
    a = np.array([lognormal_sampler(MEAN_SERVICE, SERVICE_CV)(rng) for _ in range(20000)])
    b = np.array([exponential_sampler(MEAN_SERVICE)(rng) for _ in range(20000)])
    assert a.mean() == pytest.approx(b.mean(), rel=0.05)
    assert a.std() > 1.5 * b.std()


def test_arrival_rate_stays_inside_its_stated_bounds():
    t = np.linspace(0, 2000, 5000)
    r = np.array([rate_fn(x) for x in t])
    assert r.min() >= BASE_RATE - 1e-9
    assert r.max() <= PEAK_RATE + 1e-9
    assert r.max() - r.min() > 0.4 * (PEAK_RATE - BASE_RATE)


def test_the_model_without_abandonment_never_loses_anyone():
    s = replicate(run_model_noaband, 2, 0, n_servers=6,
                  n_customers=800, mean_service=4.0)
    assert all(x["abandon_rate"] == 0.0 for x in s)


def test_the_real_system_does_lose_people_when_it_is_busy():
    s = replicate(run_system, 2, 0, n_servers=7, n_customers=1500)
    assert mean_of(s, "abandon_rate") > 0.02


def test_calibration_lands_on_the_wait_it_was_aiming_at():
    target = 1.0
    fitted = calibrate(run_model_aband, target, n_servers=8,
                       n_customers=1500, reps=4, seed0=77)
    s = replicate(run_model_aband, 4, 77, n_servers=8,
                  n_customers=1500, mean_service=fitted)
    assert mean_of(s, "mean_wait") == pytest.approx(target, abs=0.05)


def test_an_unreachable_target_fails_loudly_instead_of_quietly():
    """Abandonment caps how bad the wait can get, so a huge target is impossible."""
    with pytest.raises(RuntimeError):
        calibrate(run_model_aband, 50.0, n_servers=8,
                  n_customers=600, reps=2, seed0=5, lo=1.0, hi=6.0)


def test_waiting_gets_worse_as_servers_are_removed():
    waits = []
    for c in (6, 8, 10):
        s = replicate(run_system, 3, 300, n_servers=c, n_customers=2000)
        waits.append(mean_of(s, "mean_wait"))
    assert waits[0] > waits[1] > waits[2]


def test_the_tail_is_always_worse_than_the_average():
    s = replicate(run_system, 2, 400, n_servers=8, n_customers=2000)
    assert mean_of(s, "p95_wait") > mean_of(s, "mean_wait")


def test_with_the_true_service_time_the_richer_model_is_the_closer_one():
    """Before any calibration, the model that keeps abandonment is nearer."""
    truth = mean_of(replicate(run_system, 3, 600, n_servers=8, n_customers=2000),
                    "mean_wait")
    a = mean_of(replicate(run_model_noaband, 3, 600, n_servers=8,
                          n_customers=2000, mean_service=MEAN_SERVICE), "mean_wait")
    b = mean_of(replicate(run_model_aband, 3, 600, n_servers=8,
                          n_customers=2000, mean_service=MEAN_SERVICE), "mean_wait")
    assert abs(b - truth) < abs(a - truth)
