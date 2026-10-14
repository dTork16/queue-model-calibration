import numpy as np
import pytest

from src.des import (simulate, arrival_times, summarise, max_concurrent,
                     DONE, ABANDONED, WAITING)


def const_service(d):
    return lambda rng: d


def flat_rate(t):
    return 1.0


def test_single_server_queue_matches_a_hand_calculation():
    """Arrivals at 0, 1, 2 with one server and 3-long services.

    The first starts at once. The second waits until 3, the third until 6,
    so the waits are 0, 2 and 4.
    """
    res = simulate(3, 1, flat_rate, 1.0, const_service(3.0), None,
                   np.random.default_rng(0), arrivals=[0.0, 1.0, 2.0])
    assert np.allclose(res["wait"], [0.0, 2.0, 4.0])
    assert (res["status"] == DONE).all()


def test_enough_servers_means_nobody_waits():
    res = simulate(4, 4, flat_rate, 1.0, const_service(5.0), None,
                   np.random.default_rng(0), arrivals=[0.0, 0.5, 1.0, 1.5])
    assert np.allclose(res["wait"], 0.0)


def test_servers_are_never_oversubscribed():
    rng = np.random.default_rng(3)
    for c in (1, 3, 7):
        res = simulate(400, c, flat_rate, 1.0,
                       lambda r: float(r.exponential(4.0)), None, rng)
        assert max_concurrent(res) <= c


def test_everyone_leaves_one_way_or_the_other():
    rng = np.random.default_rng(4)
    res = simulate(500, 2, flat_rate, 1.0,
                   lambda r: float(r.exponential(3.0)),
                   lambda r: float(r.exponential(2.0)), rng)
    assert not (res["status"] == WAITING).any()
    assert (res["served"] | res["abandoned"]).all()


def test_impatience_makes_people_leave_and_its_absence_does_not():
    rng = np.random.default_rng(5)
    busy = dict(n_customers=400, n_servers=1, rate_fn=flat_rate, rate_max=1.0,
                service_sampler=const_service(3.0))
    patient = simulate(patience_sampler=None, rng=np.random.default_rng(5), **busy)
    impatient = simulate(patience_sampler=const_service(2.0),
                         rng=np.random.default_rng(5), **busy)
    assert patient["abandoned"].sum() == 0
    assert impatient["abandoned"].sum() > 0


def test_nobody_waits_a_negative_amount_of_time():
    rng = np.random.default_rng(6)
    res = simulate(300, 3, flat_rate, 1.0,
                   lambda r: float(r.exponential(3.0)),
                   lambda r: float(r.exponential(5.0)), rng)
    w = res["wait"][np.isfinite(res["wait"])]
    assert (w >= -1e-12).all()


def test_abandonment_happens_exactly_at_the_patience_limit():
    """One server busy until 10, so the second customer gives up at 2."""
    res = simulate(2, 1, flat_rate, 1.0, const_service(10.0),
                   const_service(2.0), np.random.default_rng(0),
                   arrivals=[0.0, 0.0])
    assert res["status"][0] == DONE
    assert res["status"][1] == ABANDONED
    assert res["wait"][1] == pytest.approx(2.0)


def test_more_servers_never_means_longer_waits():
    waits = []
    for c in (2, 4, 6, 8):
        res = simulate(1500, c, flat_rate, 1.0,
                       lambda r: float(r.exponential(4.0)), None,
                       np.random.default_rng(7))
        waits.append(summarise(res)["mean_wait"])
    assert all(waits[i] > waits[i + 1] for i in range(len(waits) - 1))


def test_arrival_times_are_increasing_and_about_the_right_rate():
    rng = np.random.default_rng(8)
    t = arrival_times(3000, flat_rate, 1.0, rng)
    assert (np.diff(t) > 0).all()
    assert len(t) / t[-1] == pytest.approx(1.0, rel=0.1)


def test_a_time_varying_rate_actually_varies_the_arrivals():
    def peaky(t):
        return 0.2 if (t % 100) < 50 else 2.0
    rng = np.random.default_rng(9)
    t = arrival_times(2000, peaky, 2.0, rng)
    phase = t % 100
    assert (phase >= 50).sum() > 4 * (phase < 50).sum()


def test_same_seed_repeats_and_a_different_one_does_not():
    kw = dict(n_customers=200, n_servers=2, rate_fn=flat_rate, rate_max=1.0,
              service_sampler=lambda r: float(r.exponential(3.0)),
              patience_sampler=None)
    a = simulate(rng=np.random.default_rng(11), **kw)
    b = simulate(rng=np.random.default_rng(11), **kw)
    c = simulate(rng=np.random.default_rng(12), **kw)
    assert np.allclose(a["wait"], b["wait"])
    assert not np.allclose(a["wait"], c["wait"])


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        simulate(5, 0, flat_rate, 1.0, const_service(1.0), None,
                 np.random.default_rng(0))
    with pytest.raises(ValueError):
        simulate(3, 1, flat_rate, 1.0, const_service(1.0), None,
                 np.random.default_rng(0), arrivals=[0.0, 1.0])


def test_summarise_reports_what_it_claims_to():
    res = {
        "wait": np.array([1.0, 2.0, 3.0, 4.0, 99.0]),
        "status": np.array([DONE, DONE, DONE, DONE, ABANDONED]),
        "served": np.array([True, True, True, True, False]),
        "abandoned": np.array([False, False, False, False, True]),
    }
    s = summarise(res)
    assert s["mean_wait"] == pytest.approx(2.5)
    assert s["abandon_rate"] == pytest.approx(0.2)
    assert s["n"] == 5
    assert s["p95_wait"] == pytest.approx(np.percentile([1.0, 2.0, 3.0, 4.0], 95))
