"""
Grid test: how many observations do we need to distinguish a CDM shape
distribution from a SIDM one with an offset in c/a?

Two axes:
    N              -- number of observed systems (log-spaced, 10 -> 10000)
    SIDM_modifier  -- offset added to ca for the SIDM model (0.1 -> 0.4)

Two families of output:
    Stage "stats" : KS statistic and Earth Mover's (Wasserstein) distance
                    between the projected q distributions.
    Stage "mcmc"  : run infer_intrinsic_shape on both q samples and compare
                    the inferred ca posteriors (tension in sigma).

Usage
-----
    python shape_grid_test.py --stage stats
    python shape_grid_test.py --stage mcmc      # slow
    python shape_grid_test.py --stage plot
    python shape_grid_test.py --stage all
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp, wasserstein_distance

import shape_inference as si

# --------------------------------------------------------------------------
# Fiducial CDM model
# --------------------------------------------------------------------------
BA = 0.76
CA = 0.40
SIGMA_B = 0.11
SIGMA_C = 0.13

# --------------------------------------------------------------------------
# Grid: 10 x 10 = 100 points
# --------------------------------------------------------------------------
N_GRID = np.unique(np.logspace(np.log10(10), np.log10(5000), 8).astype(int))
MOD_GRID = np.linspace(0.05, 0.40, 10)

# Statistics stage: repeat each grid point to beat down realization noise
N_REALIZATIONS = 50

# MCMC stage
N_STEPS = 1000
CA_INDEX = 1          # index of ca in the parameter vector (ba, ca, sig_b, sig_c)
# ca + modifier must stay below b/a, otherwise c > b and the ellipsoid is not a
# valid oblate/triaxial shape. With ba=0.76, ca=0.40 the largest usable
# modifier is ~0.35, so the top of MOD_GRID gets clipped.
CA_MAX = BA - 0.01

OUT = "grid_test"
STATS_FILE = os.path.join(OUT, "stats_grid.npz")
MCMC_FILE = os.path.join(OUT, "mcmc_grid.npz")
MCMC_DIR = os.path.join(OUT, "mcmc_SIDM_runs")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_q(n, ba, ca, sigma_b, sigma_c):
    """Intrinsic shapes -> one random projection each -> axis ratios q."""
    B, C = si.generate_ellipsoid_distribution(n, ba, ca, sigma_b, sigma_c)
    q = si.generate_projections_from_distribution(B, C, 1)
    return np.asarray(q).ravel()


def load_cached(prefix, out_dir):
    """Return a previous run's results dict, or None if it isn't on disk yet."""
    try:
        res = si.load_results(prefix, output_dir=out_dir)
    except:
        return None
    return res if res else None


def ca_summary(res):
    """Median and half-width of the 16-84 range for ca from a results dict."""
    samples = np.asarray(res["samples"])
    if samples.ndim == 3:                       # (walkers, steps, ndim)
        samples = samples.reshape(-1, samples.shape[-1])
    chain = samples[:, CA_INDEX]
    lo, mid, hi = np.percentile(chain, [16, 50, 84])
    return mid, 0.5 * (hi - lo)


def _blank_grids():
    """NaN marks 'not computed yet'."""
    shape = (len(MOD_GRID), len(N_GRID))
    return {k: np.full(shape, np.nan)
            for k in ("ks_stat", "ks_pval", "emd", "frac_reject")}


def _load_stats_cache():
    """Reuse a previous stats grid, but only if it is on the same axes."""
    if not os.path.exists(STATS_FILE):
        return _blank_grids(), 0

    d = np.load(STATS_FILE)
    same_axes = (d["N"].shape == N_GRID.shape
                 and np.allclose(d["N"], N_GRID)
                 and d["mod"].shape == MOD_GRID.shape
                 and np.allclose(d["mod"], MOD_GRID))
    if not same_axes:
        print("  grid axes changed since last run -- recomputing from scratch")
        return _blank_grids(), 0

    if int(d["n_realizations"]) != N_REALIZATIONS:
        print(f"  N_REALIZATIONS changed "
              f"({int(d['n_realizations'])} -> {N_REALIZATIONS}) -- recomputing")
        return _blank_grids(), 0

    grids = {k: np.array(d[k], dtype=float) for k in
             ("ks_stat", "ks_pval", "emd", "frac_reject")}
    n_done = int(np.sum(~np.isnan(grids["ks_stat"])))
    print(f"  loaded {STATS_FILE}: {n_done}/{grids['ks_stat'].size} nodes cached")
    return grids, n_done


def _save_stats(g):
    np.savez(STATS_FILE, N=N_GRID, mod=MOD_GRID,
             n_realizations=N_REALIZATIONS, **g)


def run_stats(seed=42, force=False):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)

    g, _ = (_blank_grids(), 0) if force else _load_stats_cache()
    ks_stat, ks_pval = g["ks_stat"], g["ks_pval"]
    emd, frac_reject = g["emd"], g["frac_reject"]

    for i, mod in enumerate(MOD_GRID):
        ca_s = min(CA + mod, CA_MAX)
        n_new = 0
        for j, n in enumerate(N_GRID):
            if not np.isnan(ks_stat[i, j]):  # already have this node
                continue

            ks_r, p_r, emd_r = [], [], []
            for _ in range(N_REALIZATIONS):
                q_c = make_q(n, BA, CA, SIGMA_B, SIGMA_C, rng=rng)
                q_s = make_q(n, BA, ca_s, SIGMA_B, SIGMA_C, rng=rng)
                d, p = ks_2samp(q_c, q_s)
                ks_r.append(d)
                p_r.append(p)
                emd_r.append(wasserstein_distance(q_c, q_s))

            ks_stat[i, j] = np.median(ks_r)
            ks_pval[i, j] = np.median(p_r)
            emd[i, j] = np.median(emd_r)
            frac_reject[i, j] = np.mean(np.array(p_r) < 0.05)
            n_new += 1

        _save_stats(g)  # checkpoint each row, so a Ctrl-C loses one row max
        print(f"  stats: modifier={mod:.3f} done ({n_new} new)", flush=True)

    print(f"wrote {STATS_FILE}")


def run_mcmc(seed=7):
    os.makedirs(MCMC_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    # CDM does not depend on the modifier: one run per N, reused down the column.
    cdm_ca = np.zeros(len(N_GRID))
    cdm_err = np.zeros(len(N_GRID))
    for j, n in enumerate(N_GRID):
        prefix = f"CDM_N{n}"
        d = os.path.join(MCMC_DIR, prefix)
        res = load_cached(prefix, d)
        if res is None:
            q = make_q(n, BA, CA, SIGMA_B, SIGMA_C)
            t0 = time.time()
            si.infer_intrinsic_shape(q, n_steps=N_STEPS,
                                     output_prefix=prefix, output_dir=d)
            print(f"  CDM N={n}: {time.time() - t0:.0f}s", flush=True)
            res = load_cached(prefix, d)
        else:
            print(f"  CDM N={n}: cached", flush=True)
        cdm_ca[j], cdm_err[j] = ca_summary(res)

    shape = (len(MOD_GRID), len(N_GRID))
    sidm_ca = np.zeros(shape)
    sidm_err = np.zeros(shape)

    for i, mod in enumerate(MOD_GRID):
        ca_s = min(CA + mod, CA_MAX)
        for j, n in enumerate(N_GRID):
            prefix = f"SIDM_N{n}_mod{mod:.3f}".replace(".", "p")
            d = os.path.join(MCMC_DIR, prefix)
            res = load_cached(prefix, d)
            if res is None:
                q = make_q(n, BA, ca_s, SIGMA_B, SIGMA_C)
                t0 = time.time()
                si.infer_intrinsic_shape(q, n_steps=N_STEPS,
                                         output_prefix=prefix, output_dir=d)
                print(f"  SIDM N={n} mod={mod:.3f}: "
                      f"{time.time() - t0:.0f}s", flush=True)
                res = load_cached(prefix, d)
            else:
                print(f"  SIDM N={n} mod={mod:.3f}: cached", flush=True)
            sidm_ca[i, j], sidm_err[i, j] = ca_summary(res)

    delta = sidm_ca - cdm_ca[None, :]
    tension = np.abs(delta) / np.sqrt(sidm_err ** 2 + cdm_err[None, :] ** 2)

    np.savez(MCMC_FILE, N=N_GRID, mod=MOD_GRID,
             cdm_ca=cdm_ca, cdm_err=cdm_err,
             sidm_ca=sidm_ca, sidm_err=sidm_err,
             delta_ca=delta, tension=tension)
    print(f"wrote {MCMC_FILE}")


# --------------------------------------------------------------------------
# Stage 3 -- plots
# --------------------------------------------------------------------------
def _panel(ax, N, mod, Z, title, cbar_label, log=False, levels=None,
           contour_at=None, cmap="viridis"):
    X, Y = np.meshgrid(N, mod)
    kw = dict(cmap=cmap)
    if log:
        pos = Z[Z > 0]
        kw["norm"] = LogNorm(vmin=pos.min(), vmax=Z.max()) if pos.size else None
    cf = ax.contourf(X, Y, Z, levels=levels if levels is not None else 20, **kw)
    if contour_at:
        cs = ax.contour(X, Y, Z, levels=contour_at, colors="w", linewidths=1.2)
        ax.clabel(cs, fmt="%g", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("N observations")
    ax.set_ylabel(r"SIDM modifier  $\Delta(c/a)$")
    ax.set_title(title)
    plt.colorbar(cf, ax=ax, label=cbar_label)


def plot_stats():
    d = np.load(STATS_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    _panel(axes[0], d["N"], d["mod"], d["ks_stat"],
           "KS statistic (median of realizations)", "D")
    _panel(axes[1], d["N"], d["mod"], d["emd"],
           "Earth Mover's distance", r"$W_1(q_{\rm CDM}, q_{\rm SIDM})$",
           cmap="magma")
    _panel(axes[2], d["N"], d["mod"], d["frac_reject"],
           "Fraction of trials rejected at p < 0.05", "power",
           levels=np.linspace(0, 1, 21), contour_at=[0.5, 0.9], cmap="cividis")
    out = os.path.join(OUT, "q_distribution_significance.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def plot_mcmc():
    d = np.load(MCMC_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    _panel(axes[0], d["N"], d["mod"], d["delta_ca"],
           r"Inferred $c/a$ offset", r"$\hat{c/a}_{\rm SIDM}-\hat{c/a}_{\rm CDM}$",
           cmap="coolwarm")
    _panel(axes[1], d["N"], d["mod"], d["tension"],
           r"Posterior tension in $c/a$", r"$\sigma$",
           contour_at=[1, 2, 3, 5], cmap="inferno")
    out = os.path.join(OUT, "inferred_ca_significance.png")
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stats",
                    choices=["stats", "mcmc", "plot", "all"])
    args = ap.parse_args()

    if args.stage in ("stats", "all"):
        run_stats()
    if args.stage in ("mcmc", "all"):
        run_mcmc()
    if args.stage in ("plot", "all"):
        if os.path.exists(STATS_FILE):
            plot_stats()
        if os.path.exists(MCMC_FILE):
            plot_mcmc()