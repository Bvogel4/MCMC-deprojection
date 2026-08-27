"""
shape_plotting.py - Visualization functions for 3D shape inference.

Supports both the 4-parameter and 7-parameter shape models:

    4 params : [B/A, C/A, sigma_B, sigma_C]        (MCMC ordering, see note below)
    7 params : [mu_a, mu_b, mu_c, sigma_a, sigma_b, sigma_c, sigma_ac]
                 mu_a        ellipsoid axis A        (> 0)
                 mu_b, mu_c  axis ratios B/A, C/A    (0 < mu_c <= mu_b <= 1)
                 sigma_*     standard deviations     (> 0)
                 sigma_ac    covariance of A and C/A (|sigma_ac| < sigma_a*sigma_c)

All model realisations go through `generate_ellipsoid_distribution` and
`generate_model_projections` from shape_inference, so the plots always show the
same model your likelihood is using.

--------------------------------------------------------------------------
Parameter ordering
--------------------------------------------------------------------------
The MCMC vector is always [B/A, C/A, sigma_B, sigma_C] at 4 parameters and
[mu_a, mu_b, mu_c, sigma_a, sigma_b, sigma_c, sigma_ac] at 7. That is the only
ordering this module reads or writes.

The one exception is `to_generator_params`, which reorders the 4-parameter
vector to the (mu_B, sigma_B, mu_C, sigma_C) signature documented by
`generate_ellipsoid_distribution` before sampling. Every sampling call in this
module routes through it, so if your generator in fact takes [B/A, C/A, s_B,
s_C] directly, delete the one reorder line there and nothing else changes.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde, ks_2samp

from shape_inference import (
    generate_ellipsoid_distribution,
    generate_model_projections,
    random_viewing_angles
)
from fast_likelihood import projected_semi_axes
# ---------------------------------------------------------------------------
# KF observational data
# ---------------------------------------------------------------------------
KF_DATA = {
    'KF High Mass': {
        '1R_eff': {'center': (0.863, 0.297), 'std': (0.075, 0.103)},
        '2R_eff': {'center': (0.91, 0.352), 'std': (0.04, 0.155)}
    },
    'KF Medium Mass': {
        '1R_eff': {'center': (0.857, 0.323), 'std': (0.05, 0.118)},
        '2R_eff': {'center': (0.902, 0.371), 'std': (0.05, 0.163)}
    },
    'KF Low Mass': {
        '1R_eff': {'center': (0.753, 0.459), 'std': (0.087, 0.194)},
        '2R_eff': {'center': (0.778, 0.446), 'std': (0.102, 0.213)}
    }
}

KF_ELL_COLORS = {
    'KF Low Mass': 'indigo',
    'KF Medium Mass': 'darkred',
    'KF High Mass': 'limegreen',
}


# ===========================================================================
# Parameter-model bookkeeping  (the core of the 4-vs-7 generalisation)
# ===========================================================================

LABELS_4 = ["B/A", "C/A", r"$\sigma_B$", r"$\sigma_C$"]
LABELS_7 = [r"$\mu_a$", r"$\mu_b$ (B/A)", r"$\mu_c$ (C/A)",
            r"$\sigma_a$", r"$\sigma_b$", r"$\sigma_c$", r"$\sigma_{ac}$"]


def n_params(obj):
    """Number of model parameters implied by a sample array or parameter vector."""
    arr = np.asarray(obj)
    return arr.shape[-1] if arr.ndim > 1 else arr.size


def param_index(ndim, name):
    """
    Index of a named quantity in the parameter vector, or None if the model
    does not contain it.

    Valid names: mu_A, mu_B, mu_C, sigma_A, sigma_B, sigma_C, sigma_ac
    """
    if ndim == 4:
        table = {'mu_B': 0, 'mu_C': 1, 'sigma_B': 2, 'sigma_C': 3}
    elif ndim == 7:
        table = {'mu_A': 0, 'mu_B': 1, 'mu_C': 2,
                 'sigma_A': 3, 'sigma_B': 4, 'sigma_C': 5, 'sigma_ac': 6}
    else:
        raise ValueError(f"Unsupported parameter-vector length: {ndim}")
    return table.get(name)


def get_labels(ndim):
    """Plot labels for a 4- or 7-parameter vector."""
    if ndim == 4:
        return list(LABELS_4)
    if ndim == 7:
        return list(LABELS_7)
    raise ValueError(f"Unsupported parameter-vector length: {ndim}")


def get_corner_ranges(ndim, samples=None):
    """
    Per-parameter ranges for corner.corner. Axis ratios are pinned to (0, 1);
    unbounded parameters (mu_a, sigma_a, sigma_ac) get a 99% quantile range,
    which corner interprets from a bare float.
    """
    if ndim == 4:
        rng = [(0, 1)] * 4
        for name in ('sigma_B', 'sigma_C'):
            rng[param_index(4, name)] = (0, 0.5)
        return rng
    if ndim == 7:
        rng = [0.99] * 7
        rng[1] = (0, 1)      # mu_b
        rng[2] = (0, 1)      # mu_c
        rng[4] = (0, 1)      # sigma_b
        rng[5] = (0, 1)      # sigma_c
        return rng
    raise ValueError(f"Unsupported parameter-vector length: {ndim}")


def unpack_params(params):
    """
    Pull the shape-plane quantities out of a 4- or 7-parameter vector.

    Returns
    -------
    dict with keys mu_B, mu_C, sigma_B, sigma_C, and (7-param only, else None)
    mu_A, sigma_A, sigma_ac.
    """
    p = np.asarray(params, dtype=float).ravel()
    ndim = p.size
    out = {}
    for name in ('mu_A', 'mu_B', 'mu_C', 'sigma_A', 'sigma_B', 'sigma_C', 'sigma_ac'):
        idx = param_index(ndim, name)
        out[name] = None if idx is None else float(p[idx])
    return out


# def to_generator_params(params):
#     """
#     Convert an MCMC parameter vector into the ordering expected by
#     `generate_ellipsoid_distribution`:
#
#         [B/A, C/A, sigma_B, sigma_C]  ->  (mu_B, sigma_B, mu_C, sigma_C)
#         7-parameter vectors           ->  unchanged
#
#     Every sampling call in this module goes through here, so this is the single
#     place the generator's argument order is assumed.
#     """
#     p = np.asarray(params, dtype=float).ravel()
#     if p.size == 7:
#         return p
#     if p.size == 4:
#         return p[[0, 2, 1, 3]]      # <- the only reorder in the module
#     raise ValueError(f"params must have length 4 or 7, got {p.size}")


def sample_model(params, n_samples=20000, rng=None):
    """
    Draw intrinsic (A, B/A, C/A) from the model. Thin wrapper that handles the
    parameter reordering and returns the axes in a consistent (A, B, C) order.
    """
    B, C, A,_ = generate_ellipsoid_distribution(
        params, n_samples, rng=rng
    )
    return np.asarray(A), np.asarray(B), np.asarray(C)


def sample_model_projections(params, n_samples=20000):
    """Projected axis ratios q (and projected a) for a parameter vector."""
    B, C, A,_ = generate_ellipsoid_distribution(params,n_samples)
    phi,theta = random_viewing_angles(n_samples)

    alpha, beta = projected_semi_axes(phi, theta, B, C)
    q_model = beta / alpha
    a_model = A * np.sqrt(alpha * beta)
    return np.asarray(q_model), np.asarray(a_model)


def _sensible_ellipse(sigma_B, sigma_C, floor=0.05, max_ratio=10.0):
    """
    Whether a (sigma_B, sigma_C) pair is worth drawing as an ellipse.

    NB: the original had a precedence bug -- `a > f or b > f and (...)` binds as
    `a > f or (b > f and (...))`, so a large sigma_B skipped the ratio check
    entirely. Parenthesised properly here.
    """
    if sigma_B is None or sigma_C is None:
        return False
    if sigma_B <= 0 or sigma_C <= 0:
        return False
    big_enough = (sigma_B > floor) or (sigma_C > floor)
    balanced = (sigma_B / sigma_C < max_ratio) and (sigma_C / sigma_B < max_ratio)
    return big_enough and balanced


def _density_levels(H, fracs=(0.393,0.393)):
    """Contour levels enclosing the given probability fractions."""
    Hs = np.sort(H.ravel())[::-1]
    csum = np.cumsum(Hs)
    csum /= csum[-1]
    levels = [Hs[min(np.searchsorted(csum, f), Hs.size - 1)] for f in fracs]
    return sorted(set(levels))


# ===========================================================================
# Corner plot
# ===========================================================================

def plot_corner(samples, max_prob_params=None, true_params=None,
                output_file=None, title=None, labels=None, ranges=None):
    """
    Corner plot of the MCMC samples. Works for 4- or 7-parameter chains; labels
    and axis ranges are chosen from the sample dimensionality.

    Parameters
    ----------
    samples : (n_samples, ndim) array
    max_prob_params : (ndim,) array, optional -- overplotted as a square
    true_params : (ndim,) array, optional -- passed to corner as truths
    labels, ranges : override the automatic choices if you want
    """
    samples = np.asarray(samples)
    ndim = samples.shape[1]

    if labels is None:
        labels = get_labels(ndim)
    if ranges is None:
        ranges = get_corner_ranges(ndim, samples)

    fig = corner.corner(
        samples,
        labels=labels,
        truths=None if true_params is None else np.asarray(true_params).ravel(),
        range=ranges,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={"density": True},
        levels=(0.68, 0.95),
        plot_datapoints=True,
        fill_contours=True,
        smooth=1.0,
    )

    if max_prob_params is not None:
        corner.overplot_points(
            fig, np.asarray(max_prob_params).reshape(1, -1),
            marker="s", color="C1", markersize=8,
        )

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig




def _shape_contours(ax, params, color, n_samples=20000, bins=30, seed=None,
                    linewidths=1.4, linestyles='-'):
    """Draw 1/2/3-sigma density contours of the B/A-C/A distribution implied
    by `params` (4- or 7-parameter). Returns nothing."""
    rng = np.random.default_rng(seed)
    _, bm, cm = sample_model(params, n_samples, rng=rng)
    H, xe, ye = np.histogram2d(bm, cm, bins=bins, range=[[0, 1], [0, 1]],
                               density=True)
    levels = _density_levels(H.T)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    ax.contour(xc, yc, H.T, levels=levels, colors=color,
               linewidths=linewidths, linestyles=linestyles)


def _param_text(fit, ndim, prefix=""):
    """One text block per inference: mu +/- sigma for B and C, plus sigma_ac
    when the model carries it."""
    lines = [
        rf"$\mu_B={fit['mu_B']:.3f}\pm{fit['sigma_B']:.3f}$",
        rf"$\mu_C={fit['mu_C']:.3f}\pm{fit['sigma_C']:.3f}$",
    ]
    if ndim == 7 and 'sigma_ac' in fit:
        lines.append(rf"$\sigma_{{ac}}={fit['sigma_ac']:.3f}$")
    if prefix:
        lines.insert(0, prefix)
    return "\n".join(lines)


def plot_ellipsoid_shapes(samples_list, max_prob_list, true_params_list=None, labels=None,
                          color=None, output_file=None, title=None,
                          focus_on_max_prob=True, show_samples=False, show_ellipses=True,
                          intrinsic_data=None,
                          kf_key='2R_eff', show_contours=None, n_model_samples=20000,
                          seed=None, annotate=True):
    """
    Plot B/A vs C/A for one or more inferences. Each entry may be a 4- or
    7-parameter fit; the relevant components are extracted by name, so the two
    can be mixed in a single figure.

    Parameters
    ----------
    samples_list : list of (n_samples, ndim) arrays
    max_prob_list : list of (ndim,) parameter vectors
    true_params_list : list of (ndim,) vectors or None
    labels : list of str -- if a label matches a KF_DATA key, that ellipse is drawn
    kf_key : '1R_eff' or '2R_eff'
    show_contours : True/False to force model density contours instead of the
        simple ellipse; None (default) draws contours for 7-parameter fits and
        ellipses for 4-parameter fits.
    annotate : print mu_B, mu_C, sigma_B, sigma_C (and sigma_ac when present)
        in a box on the axes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    if color is None:
        color = plt.cm.tab10.colors[:len(samples_list)]
    elif isinstance(color, str) or (isinstance(color, tuple)
                                    and all(isinstance(v, (int, float)) for v in color)):
        # A bare colour ("blue", (0.1, 0.2, 0.3)) would otherwise be zipped
        # element-wise; broadcast it over every entry instead.
        color = [color] * len(samples_list)
    if labels is None:
        labels = [f"Shape {i + 1}" for i in range(len(samples_list))]
    if true_params_list is None:
        true_params_list = [None] * len(samples_list)

    text_blocks = []
    sc = None
    for samples, max_prob, true_params, label, col in zip(
            samples_list, max_prob_list, true_params_list, labels, color):

        label = label.split('.')[0]
        samples = np.asarray(samples)
        ndim = samples.shape[1]

        iB = param_index(ndim, 'mu_B')
        iC = param_index(ndim, 'mu_C')
        sB = param_index(ndim, 'sigma_B')
        sC = param_index(ndim, 'sigma_C')

        if show_samples and not focus_on_max_prob:
            colors = np.sqrt(samples[:, sB] ** 2 + samples[:, sC]**2)
            sc = ax.scatter(samples[:, iB], samples[:, iC], c=colors, cmap='viridis',
                            alpha=1, label=None)
            # plt.colorbar(sc, ax=ax, label='col')

        max_prob = np.asarray(max_prob, dtype=float).ravel()
        fit = unpack_params(max_prob)

        # contours for the 7-parameter model, ellipses for the 4-parameter one,
        # unless the caller overrides it
        use_contours = False#(ndim == 7) if show_contours is None else show_contours

        if true_params is not None:
            true_params = np.asarray(true_params, dtype=float).ravel()
            truth = unpack_params(true_params)
            ax.scatter(truth['mu_B'], truth['mu_C'], marker='o', s=75, color='k',
                       edgecolors='white', linewidth=1.5, zorder=5)

        if focus_on_max_prob:
            ax.scatter(fit['mu_B'], fit['mu_C'], marker='o', s=75, color=col,
                       edgecolors='white', linewidth=2, zorder=5)
        # else:
        #     ax.scatter(fit['mu_B'], fit['mu_C'], marker='o', s=75, color=col,
        #                edgecolors='white', linewidth=1, label=f"{label} (Inferred)",
        #                zorder=5)

        if show_ellipses or use_contours:
            # --- truth ---
            if true_params is not None:
                if use_contours:
                    _shape_contours(ax, true_params, 'k', n_samples=n_model_samples,
                                    seed=seed, linestyles=':', linewidths=2.0)
                    ax.plot(-1, -1, c='k', linestyle=':', linewidth=2.5,
                            label=rf"Intrinsic Distribution ({label}, 1$\sigma$)")
                elif _sensible_ellipse(truth['sigma_B'], truth['sigma_C']):
                    ax.add_patch(Ellipse(
                        xy=(truth['mu_B'], truth['mu_C']),
                        width=2 * truth['sigma_B'], height=2 * truth['sigma_C'],
                        edgecolor='k', facecolor='none', linestyle=':',
                        alpha=1, linewidth=2.5))
                    ax.plot(-1, -1, c='k', linestyle=':', linewidth=2.5,
                            label=f"Intrinsic Distribution ({label})")

            # --- inferred ---
            if use_contours:
                _shape_contours(ax, max_prob, col, n_samples=n_model_samples,
                                seed=seed, linestyles='-', linewidths=2.0)
                ax.plot(-1, -1, c=col, linestyle='-', linewidth=2.5,
                        label=rf"Inferred Distribution ({label}, 1/2/3$\sigma$)")
            elif _sensible_ellipse(fit['sigma_B'], fit['sigma_C']):
                ax.add_patch(Ellipse(
                    xy=(fit['mu_B'], fit['mu_C']),
                    width=2 * fit['sigma_B'], height=2 * fit['sigma_C'],
                    edgecolor=col, facecolor='none', linestyle='-',
                    alpha=1, linewidth=2.5))
                ax.plot(-1, -1, c=col, linestyle='-', linewidth=2.5,
                        label=f"Inferred Distribution ({label})")

        if annotate:
            src = truth
            head = f"(Intrinsinc) "
            text_blocks.append((_param_text(src, ndim, prefix=head), 'k'))
            src =  fit
            head = f"(Inferred)"
            text_blocks.append((_param_text(src, ndim, prefix=head), col))

        if label in KF_DATA:
            a, b = KF_DATA[label][kf_key]['center']
            a_std, b_std = KF_DATA[label][kf_key]['std']
            ax.add_patch(Ellipse(xy=(a, b), width=2 * a_std, height=2 * b_std,
                                 facecolor='none', edgecolor=KF_ELL_COLORS[label],
                                 linestyle='--', linewidth=2.5))
            ax.plot(-1, -1, c=KF_ELL_COLORS[label], linestyle='--', linewidth=2.5,
                    label=f'{label}')
        if intrinsic_data is not None:
            a,ba,ca = intrinsic_data
            ax.scatter(ba, ca, marker='*', s=30, color = 'gray')
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, label=r'$\sqrt{\sigma_B^2 + \sigma_C^2}$')
        cb.solids.set_alpha(1)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'$Q = B/A$ $(2R_{eff})$', fontsize=28)
    ax.set_ylabel(r'$S = C/A$ $(2R_{eff})$', fontsize=28)
    ax.tick_params(which='both', labelsize=18)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', ncol=1, fontsize=20)

    # parameter annotations, stacked up from the bottom-right corner
    y = .6
    for text, col in reversed(text_blocks):
        t = ax.text(0.4, y, text, transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=16, color=col,
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor=col, alpha=0.8))
        y += -0.055 * (text.count('\n') + 1) + 0.02

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if output_file:
        if focus_on_max_prob:
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}_max_likelihood{ext}"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig
# ===========================================================================
# Intrinsic shape plane (B/A vs C/A)
# ===========================================================================
#
# def plot_ellipsoid_shapes(samples_list, max_prob_list, true_params_list=None, labels=None,
#                           color=None, output_file=None, title=None,
#                           focus_on_max_prob=True, show_samples=False, show_ellipses=True,
#                           kf_key='2R_eff'):
#     """
#     Plot B/A vs C/A for one or more inferences. Each entry may be a 4- or
#     7-parameter fit; the relevant components are extracted by name, so the two
#     can be mixed in a single figure.
#
#     Parameters
#     ----------
#     samples_list : list of (n_samples, ndim) arrays
#     max_prob_list : list of (ndim,) parameter vectors
#     true_params_list : list of (ndim,) vectors or None
#     labels : list of str -- if a label matches a KF_DATA key, that ellipse is drawn
#     kf_key : '1R_eff' or '2R_eff'
#     """
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#
#     if color is None:
#         color = plt.cm.tab10.colors[:len(samples_list)]
#     elif isinstance(color, str) or (isinstance(color, tuple)
#                                     and all(isinstance(v, (int, float)) for v in color)):
#         # A bare colour ("blue", (0.1, 0.2, 0.3)) would otherwise be zipped
#         # element-wise; broadcast it over every entry instead.
#         color = [color] * len(samples_list)
#     if labels is None:
#         labels = [f"Shape {i + 1}" for i in range(len(samples_list))]
#     if true_params_list is None:
#         true_params_list = [None] * len(samples_list)
#
#     for samples, max_prob, true_params, label, col in zip(
#             samples_list, max_prob_list, true_params_list, labels, color):
#
#         label = label.split('.')[0]
#         samples = np.asarray(samples)
#         ndim = samples.shape[1]
#
#         iB = param_index(ndim, 'mu_B')
#         iC = param_index(ndim, 'mu_C')
#
#         if show_samples and not focus_on_max_prob:
#             ax.scatter(samples[:, iB], samples[:, iC], alpha=0.01, color=col, label=None)
#
#         fit = unpack_params(max_prob)
#
#         if true_params is not None:
#             truth = unpack_params(true_params)
#             ax.scatter(truth['mu_B'], truth['mu_C'], marker='o', s=75, color='k',
#                        edgecolors='white', linewidth=1.5, zorder=5)
#
#         if focus_on_max_prob:
#             ax.scatter(fit['mu_B'], fit['mu_C'], marker='o', s=75, color=col,
#                        edgecolors='white', linewidth=2, zorder=5)
#         else:
#             ax.scatter(fit['mu_B'], fit['mu_C'], marker='o', s=75, color=col,
#                        edgecolors='white', linewidth=1, label=f"{label} (Inferred)",
#                        zorder=5)
#
#         if show_ellipses:
#             if true_params is not None:
#                 truth = unpack_params(true_params)
#                 if _sensible_ellipse(truth['sigma_B'], truth['sigma_C']):
#                     ax.add_patch(Ellipse(
#                         xy=(truth['mu_B'], truth['mu_C']),
#                         width=2 * truth['sigma_B'], height=2 * truth['sigma_C'],
#                         edgecolor='k', facecolor='none', linestyle=':',
#                         alpha=1, linewidth=2.5))
#                     ax.plot(-1, -1, c='k', linestyle=':', linewidth=2.5,
#                             label=f"Intrinsic Distribution ({label})")
#
#             if _sensible_ellipse(fit['sigma_B'], fit['sigma_C']):
#                 ax.add_patch(Ellipse(
#                     xy=(fit['mu_B'], fit['mu_C']),
#                     width=2 * fit['sigma_B'], height=2 * fit['sigma_C'],
#                     edgecolor=col, facecolor='none', linestyle='-',
#                     alpha=1, linewidth=2.5))
#                 ax.plot(-1, -1, c=col, linestyle='-', linewidth=2.5,
#                         label=f"Inferred Distribution ({label})")
#
#         if label in KF_DATA:
#             a, b = KF_DATA[label][kf_key]['center']
#             a_std, b_std = KF_DATA[label][kf_key]['std']
#             ax.add_patch(Ellipse(xy=(a, b), width=2 * a_std, height=2 * b_std,
#                                  facecolor='none', edgecolor=KF_ELL_COLORS[label],
#                                  linestyle='--', linewidth=2.5))
#             ax.plot(-1, -1, c=KF_ELL_COLORS[label], linestyle='--', linewidth=2.5,
#                     label=f'{label}')
#
#     ax.set_xlim(0, 1.0)
#     ax.set_ylim(0, 1.0)
#     ax.set_xlabel(r'$Q = B/A$ $(2R_{eff})$', fontsize=28)
#     ax.set_ylabel(r'$S = C/A$ $(2R_{eff})$', fontsize=28)
#     ax.tick_params(which='both', labelsize=18)
#     ax.set_aspect('equal', adjustable='box')
#     ax.legend(loc='upper left', ncol=1, fontsize=20)
#
#     if title:
#         fig.suptitle(title, fontsize=16)
#
#     plt.tight_layout()
#
#     if output_file:
#         if focus_on_max_prob:
#             base, ext = os.path.splitext(output_file)
#             output_file = f"{base}_max_likelihood{ext}"
#         fig.savefig(output_file, dpi=300, bbox_inches="tight")
#
#     return fig
#
#
# # ===========================================================================
# # Full shape-distribution figure (main use for 7-parameter fits)
# # ===========================================================================

def plot_shape_distribution(params=None, a_data=None, b_data=None, c_data=None,
                            n_model_samples=20000, label="data",
                            color="tab:blue", model_color="tab:red",
                            output_file=None, title=None, seed=None,
                            show_a_panel=None):
    """
    B/A vs C/A plane with data points and fitted-model contours, an A-C/A panel
    showing the sigma_ac covariance, and marginal histograms of A, B/A, C/A.

    Model samples come from `generate_ellipsoid_distribution` via `sample_model`,
    so this works for both 4- and 7-parameter vectors. For a 4-parameter model
    the A-related panels carry no information and are dropped by default.

    Parameters
    ----------
    params : (4,) or (7,) parameter vector, or None to plot data only
    a_data, b_data, c_data : intrinsic A, B/A, C/A arrays, or None for model only
    show_a_panel : force the A panels on/off; None = automatic (7-param only)
    """
    have_model = params is not None
    have_data = b_data is not None and c_data is not None

    if not (have_model or have_data):
        raise ValueError("Need at least one of `params` or (b_data, c_data).")

    if have_model:
        params = np.asarray(params, dtype=float).ravel()
        ndim = params.size
        fit = unpack_params(params)
        rng = np.random.default_rng(seed)
        am, bm, cm = sample_model(params, n_model_samples, rng=rng)
    else:
        ndim = None
        fit = {}
        am = bm = cm = None

    if have_data:
        b_data = np.asarray(b_data)
        c_data = np.asarray(c_data)
        a_data = None if a_data is None else np.asarray(a_data)

    if show_a_panel is None:
        show_a_panel = (ndim == 7) or (a_data is not None and np.std(a_data) > 0)

    ncols = 3 if show_a_panel else 2
    fig = plt.figure(figsize=(11 if show_a_panel else 8.5, 8))
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.6, 1.0], hspace=0.32, wspace=0.3)

    # --- main panel: B/A vs C/A ------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    if have_data:
        ax.scatter(b_data, c_data, s=8, alpha=0.35, color=color, label=label,
                   rasterized=True)

        mu_b_d, mu_c_d = np.mean(b_data), np.mean(c_data)
        s_b_d, s_c_d = np.std(b_data), np.std(c_data)
        ax.scatter(mu_b_d, mu_c_d, marker='o', s=75, color='k',
                   edgecolors='white', linewidth=1.5, zorder=5)
        if _sensible_ellipse(s_b_d, s_c_d):
            ax.add_patch(Ellipse(xy=(mu_b_d, mu_c_d), width=2 * s_b_d, height=2 * s_c_d,
                                 edgecolor='k', facecolor='none', linestyle=':',
                                 alpha=1, linewidth=2.5))
            ax.plot(-1, -1, c='k', linestyle=':', linewidth=2.5,
                    label=f"Intrinsic Distribution ({label})")

    if have_model:
        H, xe, ye = np.histogram2d(bm, cm, bins=30, range=[[0, 1], [0, 1]], density=True)
        levels = _density_levels(H.T)
        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
        ax.contour(xc, yc, H.T, levels=levels, colors=model_color, linewidths=1.4)
        ax.plot([], [], color=model_color, label=r"model (1/2/3$\sigma$)")
        ax.plot(fit['mu_B'], fit['mu_C'], marker="x", ms=10, mew=2.5, color=model_color)

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("B/A")
    ax.set_ylabel("C/A")
    ax.set_title("Intrinsic shape plane")
    ax.legend(loc="upper left", frameon=False)

    # --- A vs C/A panel (the sigma_ac covariance) -------------------------
    if show_a_panel:
        ax2 = fig.add_subplot(gs[0, 2])
        if a_data is not None:
            ax2.scatter(a_data, c_data, s=8, alpha=0.35, color=color, rasterized=True)
        if have_model:
            H, xe, ye = np.histogram2d(am, cm, bins=60, density=True)
            levels = _density_levels(H.T)
            xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
            ax2.contour(xc, yc, H.T, levels=levels, colors=model_color, linewidths=1.2)
        ax2.set_xlabel("A")
        ax2.set_ylabel("C/A")
        ax2.set_title(r"A - C/A covariance")
        ax2.set_ylim(0, 1)
        if fit.get('sigma_ac') is not None:
            ax2.annotate(rf"$\sigma_{{ac}} = {fit['sigma_ac']:.3f}$",
                         xy=(0.03, 0.95), xycoords='axes fraction',
                         fontsize=11, va='top')

    # --- marginals --------------------------------------------------------
    panels = [("B/A", b_data, bm, (0, 1)),
              ("C/A", c_data, cm, (0, 1))]
    if show_a_panel:
        panels.insert(0, ("A", a_data, am, None))

    for i, (name, d, m, rng_) in enumerate(panels):
        axm = fig.add_subplot(gs[1, i])
        bins = 40
        if d is not None:
            axm.hist(d, bins=bins, range=rng_, density=True, histtype="stepfilled",
                     alpha=0.45, color=color, label=label)
        if m is not None:
            axm.hist(m, bins=bins, range=rng_, density=True, histtype="step",
                     lw=1.8, color=model_color, label="model")
        axm.set_xlabel(name)
        if i == 0:
            axm.set_ylabel("density")
            axm.legend(frameon=False, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=15)

    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"saved figure -> {output_file}")

    return fig


# ===========================================================================
# Projected axis-ratio distributions
# ===========================================================================

def plot_projected_distributions(q_obs_list, labels=None, colors=None, bin_width=0.04,
                                 output_file=None, title=None, kde=True,
                                 precomputed_overlays=None):
    """
    Histograms of projected axis ratios for multiple distributions. Model-free;
    unchanged in behaviour from the original.

    precomputed_overlays : list of dicts with keys
        x_centers, y_values, label, color, style ('step' or 'line'),
        bin_width (default 0.1), linewidth (default 2)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if title is not None:
        fig.suptitle(title)
    if colors is None:
        colors = plt.cm.tab10.colors[:len(q_obs_list)]
    if labels is None:
        labels = [f"Distribution {i + 1}" for i in range(len(q_obs_list))]

    bins = np.arange(0, 1.01, bin_width)

    for q_obs, label, color in zip(q_obs_list, labels, colors):
        ax.hist(q_obs, bins=bins, alpha=1, color=color, label=label,
                density=True, histtype='step', linewidth=2)

        if kde and len(q_obs) > 100:
            x_grid = np.linspace(0, 1, 500)
            ax.plot(x_grid, gaussian_kde(q_obs)(x_grid),
                    color=color, linestyle='-', linewidth=2)

    if precomputed_overlays:
        for overlay in precomputed_overlays:
            x = np.array(overlay['x_centers'])
            y = np.array(overlay['y_values'])
            bw = overlay.get('bin_width', 0.1)
            lw = overlay.get('linewidth', 2)
            col = overlay['color']
            lbl = overlay['label']

            if overlay.get('style') == 'step':
                edges = np.append(x - bw / 2, x[-1] + bw / 2)
                ax.stairs(y, edges, color=col, linewidth=lw, label=lbl, fill=False)
            else:
                ax.plot(x, y, color=col, linewidth=lw, label=lbl, linestyle='-')
                ax.scatter(x, y, color=col, s=50, edgecolor='k', linewidth=1, label=None)

    ax.set_xlabel('Projected Axis Ratio (q = b/a)', fontsize=28)
    ax.set_ylabel('Density', fontsize=28)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_q_comparison(q_data, params, n_model_samples=50000, bin_width=0.04,
                      output_file=None, color="tab:blue", model_color="tab:red",
                      label="data", title=None, run_ks=True):
    """
    Compare an observed projected axis-ratio distribution to the model's, using
    the same bins as the likelihood. `params` may be 4- or 7-dimensional.
    """
    q_model, _ = sample_model_projections(params, n_model_samples)
    bins = np.arange(0, 1.0001, bin_width)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(q_data, bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color=color, label=f"{label} q")
    ax.hist(q_model, bins=bins, density=True, histtype="step", lw=1.8,
            color=model_color, label="model q")
    ax.set_xlabel("projected axis ratio q")
    ax.set_ylabel("density")

    if run_ks:
        d, p = ks_2samp(np.asarray(q_data), q_model)
        ax.annotate(f"KS: D={d:.3f}, p={p:.2e}", xy=(0.02, 0.95),
                    xycoords='axes fraction', fontsize=10, va='top')
        print(f"[plot_q_comparison] data vs model: D={d:.4f}, p={p:.4e}")

    if title:
        ax.set_title(title)
    ax.legend(frameon=False)

    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"saved figure -> {output_file}")

    return fig


def plot_projected_distributions_with_model(q_obs_list, model_params_list=None,
                                            true_params_list=None,
                                            labels=None, colors=None, bin_width=0.04,
                                            output_file=None, title=None, kde=False,
                                            model_samples=10000,
                                            q_model_list=None, q_true_list=None,
                                            verbose=True):
    """
    Observed projected axis-ratio histograms with model and truth overlays.

    Model and truth curves are generated internally with
    `generate_model_projections`, so `model_params_list` / `true_params_list`
    entries can be 4- or 7-parameter vectors. Pass `q_model_list` /
    `q_true_list` instead if you have already drawn the projections yourself.

    KS statistics (observed/model/truth) are printed for each distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if colors is None:
        colors = plt.cm.tab10.colors[:len(q_obs_list)]
    if labels is None:
        labels = [f"Distribution {i + 1}" for i in range(len(q_obs_list))]

    bins = np.arange(0, 1.01, bin_width)
    x_grid = np.linspace(0, 1, 500)
    true_color = 'red'
    ks_results = {}

    for i, (q_obs, label, color) in enumerate(zip(q_obs_list, labels, colors)):
        if color == 'red':
            color = 'purple'

        q_obs = np.asarray(q_obs)

        ax.hist(q_obs, bins=bins, alpha=1, color=color, label=f"{label} (Mock Observed)",
                density=True, histtype='step', linewidth=2.5, linestyle='--')
        if kde and len(q_obs) > 100:
            ax.plot(x_grid, gaussian_kde(q_obs)(x_grid),
                    color=color, linestyle='--', linewidth=2)

        # --- model curve ---
        q_model = None
        if q_model_list is not None and i < len(q_model_list):
            q_model = np.asarray(q_model_list[i])
        elif model_params_list is not None and i < len(model_params_list) \
                and model_params_list[i] is not None:
            q_model, _ = sample_model_projections(model_params_list[i], model_samples)

        if q_model is not None:
            ax.hist(q_model, bins=bins, alpha=1, color='k', label=f"{label} (Model)",
                    density=True, histtype='step', linewidth=2.5, linestyle='-')
            if kde and len(q_model) > 100:
                ax.plot(x_grid, gaussian_kde(q_model)(x_grid),
                        color='k', linestyle='--', linewidth=2)

        # --- truth curve ---
        q_true = None
        if q_true_list is not None and i < len(q_true_list):
            q_true = np.asarray(q_true_list[i])
        elif true_params_list is not None and i < len(true_params_list) \
                and true_params_list[i] is not None:
            q_true, _ = sample_model_projections(true_params_list[i], model_samples)

        if q_true is not None:
            ax.hist(q_true, bins=bins, alpha=1, color=true_color, label=f"{label} (True)",
                    density=True, histtype='step', linewidth=2.5, linestyle='-')
            if kde and len(q_true) > 100:
                ax.plot(x_grid, gaussian_kde(q_true)(x_grid),
                        color=true_color, linestyle=':', linewidth=2)

        # --- KS tests ---
        ks = {}
        for name, x, y in (("Observed vs Model", q_obs, q_model),
                           ("Observed vs True", q_obs, q_true),
                           ("Model vs True", q_model, q_true)):
            if x is not None and y is not None:
                ks[name] = ks_2samp(x, y)
        ks_results[label] = ks

        if verbose and ks:
            print(f"\n{'=' * 60}\nKS Test Results for {label}\n{'=' * 60}")
            for name, (d, p) in ks.items():
                verdict = ("similar (fail to reject H0)" if p > 0.05
                           else "different (reject H0)")
                print(f"{name}: D={d:.4f}, p={p:.4e}  ->  {verdict}")

    ax.set_xlabel(r'Projected Axis Ratio ($q = b/a$)', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(fontsize=16, loc='upper left')
    ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=15)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    fig.ks_results = ks_results
    return fig


# ===========================================================================
# MCMC diagnostics
# ===========================================================================

def plot_chain_evolution(sampler, burn_in=None, output_file=None, title=None, labels=None):
    """
    Evolution of the MCMC chain, one panel per parameter. Panel count and labels
    adapt to a 4- or 7-dimensional chain.
    """
    chain = sampler.get_chain() if hasattr(sampler, 'get_chain') else np.asarray(sampler)
    n_steps, n_walkers, ndim = chain.shape

    if labels is None:
        labels = get_labels(ndim)

    fig, axes = plt.subplots(ndim, figsize=(12, 1.6 * ndim + 2), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(ndim):
        ax = axes[i]
        for j in range(n_walkers):
            ax.plot(chain[:, j, i], alpha=0.1, color='k')
        ax.plot(np.median(chain[:, :, i], axis=1), color='C0', linewidth=2)
        if burn_in is not None:
            ax.axvline(burn_in, color='r', linestyle='--', alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step Number")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


# ===========================================================================
# Summary comparisons
# ===========================================================================

def plot_comparison_grid(results_dict, output_file=None, title=None):
    """
    Comparison grid: projected axis-ratio distributions, the intrinsic shape
    plane with covariance ellipses, and a parameter table.

    results_dict : {shape_name: {'q_obs', 'samples', 'max_prob_params',
                                 optional 'true_params'}}
    Mixed 4-/7-parameter entries are handled; the table columns are taken from
    the first entry's dimensionality.
    """
    n_shapes = len(results_dict)
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

    labels = list(results_dict.keys())
    colors = plt.cm.tab10.colors[:n_shapes]

    # --- panel 1: projected distributions --------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, 1.01, 0.04)
    for (shape_name, results), color in zip(results_dict.items(), colors):
        q_obs = np.asarray(results['q_obs'])
        ax1.hist(q_obs, bins=bins, alpha=0.6, color=color, label=shape_name,
                 density=True, histtype='step', linewidth=2)
        if len(q_obs) > 100:
            x_grid = np.linspace(0, 1, 500)
            ax1.plot(x_grid, gaussian_kde(q_obs)(x_grid), color=color, linewidth=2)

    ax1.set_xlabel('Projected Axis Ratio (q = b/a)')
    ax1.set_ylabel('Density')
    ax1.set_title('Projected Axis Ratio Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- panel 2: intrinsic shape plane ----------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    for (shape_name, results), color in zip(results_dict.items(), colors):
        samples = np.asarray(results['samples'])
        ndim = samples.shape[1]
        iB, iC = param_index(ndim, 'mu_B'), param_index(ndim, 'mu_C')
        B_A, C_A = samples[:, iB], samples[:, iC]

        ax2.scatter(B_A, C_A, alpha=0.01, color=color)

        if results.get('true_params') is not None:
            truth = unpack_params(results['true_params'])
            ax2.scatter(truth['mu_B'], truth['mu_C'], marker='*', s=200, color=color,
                        edgecolors='black', linewidth=1.5, label=f"{shape_name} (True)")

        fit = unpack_params(results['max_prob_params'])
        ax2.scatter(fit['mu_B'], fit['mu_C'], marker='o', s=100, color=color,
                    edgecolors='black', linewidth=1, label=f"{shape_name}")

        cov = np.cov(B_A, C_A)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        for n_sigma in (1, 2):
            ax2.add_patch(Ellipse(
                xy=(np.mean(B_A), np.mean(C_A)),
                width=2 * n_sigma * np.sqrt(max(eigvals[0], 0)),
                height=2 * n_sigma * np.sqrt(max(eigvals[1], 0)),
                angle=angle, facecolor='none', edgecolor=color, alpha=0.8,
                linewidth=2 if n_sigma == 1 else 1,
                linestyle='-' if n_sigma == 1 else '--'))

    ax2.scatter(0.9, 0.1, marker='d', s=100, color='blue', edgecolor='black', label='Disk')
    ax2.scatter(0.9, 0.9, marker='d', s=100, color='red', edgecolor='black', label='Spheroid')
    ax2.scatter(0.1, 0.1, marker='d', s=100, color='green', edgecolor='black', label='Prolate')
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('B/A')
    ax2.set_ylabel('C/A')
    ax2.set_title('Intrinsic Shape Distributions')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax2.grid(True, alpha=0.3)

    # --- panel 3: parameter table ----------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    first = np.asarray(next(iter(results_dict.values()))['samples'])
    ndim = first.shape[1]
    plain = [lab.replace('$', '').replace('\\', '') for lab in get_labels(ndim)]

    table_cols = ['Shape']
    for name in plain:
        table_cols += [f'{name} (True)', f'{name} (Inferred)']

    table_data = []
    for shape_name, results in results_dict.items():
        max_prob = np.asarray(results['max_prob_params']).ravel()
        stds = np.std(np.asarray(results['samples']), axis=0)
        true_params = results.get('true_params')
        true_params = None if true_params is None else np.asarray(true_params).ravel()

        row = [shape_name]
        for k in range(len(max_prob)):
            row.append("N/A" if true_params is None else f"{true_params[k]:.3f}")
            row.append(f"{max_prob[k]:.3f} ± {stds[k]:.3f}")
        table_data.append(row)

    table = ax3.table(cellText=table_data, colLabels=table_cols,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8 if ndim == 7 else 10)
    table.scale(1, 1.5)
    for i in range(len(table_cols)):
        table.auto_set_column_width(i)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_statistics(results_dict, output_file=None, title=None):
    """
    Per-parameter bar chart of posterior means with 1-sigma error bars and true
    values marked. Grid size adapts to the number of parameters (2x2 for 4,
    3x3 for 7).
    """
    first = np.asarray(next(iter(results_dict.values()))['samples'])
    ndim = first.shape[1]
    param_names = get_labels(ndim)

    ncols = 2 if ndim <= 4 else 3
    nrows = int(np.ceil(ndim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.tab10.colors[:len(results_dict)]

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        x_pos, y_vals, y_errs, labels, true_vals = [], [], [], [], []

        for j, (shape_name, results) in enumerate(results_dict.items()):
            samples = np.asarray(results['samples'])
            x_pos.append(j)
            y_vals.append(np.mean(samples[:, i]))
            y_errs.append(np.std(samples[:, i]))
            labels.append(shape_name)
            tp = results.get('true_params')
            true_vals.append(None if tp is None else np.asarray(tp).ravel()[i])

        bars = ax.bar(x_pos, y_vals, yerr=y_errs, alpha=0.7, capsize=5,
                      color=colors[:len(x_pos)], tick_label=labels)

        for j, true_val in enumerate(true_vals):
            if true_val is not None:
                ax.hlines(true_val, j - 0.4, j + 0.4, color='r',
                          linestyle='--', linewidth=2)
                ax.text(j, true_val, f"True: {true_val:.3f}",
                        ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Shape')
        ax.set_ylabel(f'{param_name} Value')
        ax.set_title(f'{param_name} Parameter')
        ax.grid(True, axis='y', alpha=0.3)

        for bar, val, err in zip(bars, y_vals, y_errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                    f"{val:.3f}±{err:.3f}", ha='center', va='bottom',
                    rotation=90, fontsize=8)

    for k in range(ndim, len(axes)):
        axes[k].axis('off')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig