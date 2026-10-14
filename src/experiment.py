"""Calibrate both models on one staffing level, then ask them about others.

The staffing level in use is the baseline. Both models are tuned until they
reproduce the mean wait measured there. Then staffing is varied and each model
is asked to predict what happens, with the real system run at the same levels
to say who was right.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .systems import (run_system, run_model_noaband, run_model_aband,
                      replicate, mean_of, sd_of, MEAN_SERVICE)
from .calibrate import calibrate

BASELINE = 8
STAFFING = [6, 7, 8, 9, 10]
N_CUSTOMERS = 4000
REPS = 8
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "figures")


def main():
    os.makedirs(FIGDIR, exist_ok=True)

    truth = {}
    for c in STAFFING:
        truth[c] = replicate(run_system, REPS, 1000 + 10 * c,
                             n_servers=c, n_customers=N_CUSTOMERS)
    base_wait = mean_of(truth[BASELINE], "mean_wait")
    print("real system at the baseline of %d servers: mean wait %.3f (sd %.3f)"
          % (BASELINE, base_wait, sd_of(truth[BASELINE], "mean_wait")))
    print("true mean service time: %.2f\n" % MEAN_SERVICE)

    fits = {}
    for name, fn in (("no_abandonment", run_model_noaband),
                     ("with_abandonment", run_model_aband)):
        s = calibrate(fn, base_wait, BASELINE)
        fits[name] = s
        print("model %-16s calibrated effective service time %.3f  (%+.1f%% vs truth)"
              % (name, s, 100.0 * (s - MEAN_SERVICE) / MEAN_SERVICE))

    preds = {name: {} for name in fits}
    for name, fn in (("no_abandonment", run_model_noaband),
                     ("with_abandonment", run_model_aband)):
        for c in STAFFING:
            preds[name][c] = replicate(fn, REPS, 2000 + 10 * c,
                                       n_servers=c, n_customers=N_CUSTOMERS,
                                       mean_service=fits[name])

    print("\nmean wait, real vs predicted")
    print("servers |   real | no abandonment | with abandonment")
    rows = []
    for c in STAFFING:
        t = mean_of(truth[c], "mean_wait")
        a = mean_of(preds["no_abandonment"][c], "mean_wait")
        b = mean_of(preds["with_abandonment"][c], "mean_wait")
        marker = "  <- calibrated here" if c == BASELINE else ""
        print("   %2d   | %6.2f | %14.2f | %16.2f%s" % (c, t, a, b, marker))
        rows.append({
            "servers": c,
            "real_mean_wait": t,
            "real_mean_wait_sd": sd_of(truth[c], "mean_wait"),
            "real_p95_wait": mean_of(truth[c], "p95_wait"),
            "real_abandon_rate": mean_of(truth[c], "abandon_rate"),
            "pred_no_abandonment": a,
            "pred_with_abandonment": b,
            "err_no_abandonment": a - t,
            "err_with_abandonment": b - t,
        })

    off = [r for r in rows if r["servers"] != BASELINE]
    mae_a = float(np.mean([abs(r["err_no_abandonment"]) for r in off]))
    mae_b = float(np.mean([abs(r["err_with_abandonment"]) for r in off]))
    print("\nmean absolute error away from the calibration point")
    print("  no abandonment   %.2f" % mae_a)
    print("  with abandonment %.2f" % mae_b)

    figure(rows, fits)

    out = {
        "settings": {"baseline_servers": BASELINE, "staffing": STAFFING,
                     "customers_per_run": N_CUSTOMERS, "replications": REPS,
                     "true_mean_service": MEAN_SERVICE},
        "calibrated_service_times": fits,
        "rows": rows,
        "mae_off_calibration": {"no_abandonment": mae_a,
                                "with_abandonment": mae_b},
    }
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results.json and figures/predictions.png")


def figure(rows, fits):
    c = [r["servers"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].errorbar(c, [r["real_mean_wait"] for r in rows],
                   yerr=[r["real_mean_wait_sd"] for r in rows],
                   marker="o", lw=2, color="k", label="real system")
    ax[0].plot(c, [r["pred_no_abandonment"] for r in rows],
               marker="s", ls="--", label="model, no abandonment")
    ax[0].plot(c, [r["pred_with_abandonment"] for r in rows],
               marker="^", ls="--", label="model, with abandonment")
    ax[0].axvline(BASELINE, color="grey", lw=1, ls=":")
    ax[0].annotate("calibrated here", (BASELINE, ax[0].get_ylim()[1] * 0.9),
                   ha="center", fontsize=9, color="grey")
    ax[0].set_xlabel("servers")
    ax[0].set_ylabel("mean wait")
    ax[0].set_yscale("log")
    ax[0].set_title("prediction away from the calibration point")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    width = 0.36
    x = np.arange(len(c))
    ax[1].bar(x - width / 2, [r["err_no_abandonment"] for r in rows], width,
              label="no abandonment")
    ax[1].bar(x + width / 2, [r["err_with_abandonment"] for r in rows], width,
              label="with abandonment")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(v) for v in c])
    ax[1].set_xlabel("servers")
    ax[1].set_ylabel("predicted minus real (mean wait)")
    ax[1].set_title("error")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "predictions.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
