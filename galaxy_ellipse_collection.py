#this contains a bunch of helper functions for MCMC_sims to run and plot MCMC codes from shape_inference and shape_plotting.

import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import pickle
import re
from glob import glob
import io
import contextlib
import logging
from multiprocessing import Pool

from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import is_color_like
from joblib import Memory
from matplotlib.patches import Ellipse

memory = Memory(location='.cache/interpolators', verbose=0)
from fast_likelihood import projected_semi_axes

from shape_inference import (
    # generate_projections,
    generate_ellipsoid_distribution,
    generate_model_projections,
    project_ellipticity_xu,
    infer_intrinsic_shape,
    infer_intrinsic_shape_multivariate,
    attainable,
    random_viewing_angles,
    
)


from shape_plotting import (
    n_params, unpack_params, sample_model_projections,
    plot_corner, plot_chain_evolution, plot_ellipsoid_shapes,
    plot_projected_distributions, plot_projected_distributions_with_model,
    plot_covariant_distribution, plot_q_comparison,
    fit_ellipsoid_params
)

PARAM_NAMES = {
    4: ['mu_B', 'mu_C', 'sigma_B', 'sigma_C'],
    # order taken from the infer_intrinsic_shape_multivariate docstring
    # (std+corr form). Must match what cov_fit() returns for the truth.
    7: ['mu_A', 'mu_B', 'mu_C', 'sigma_A', 'sigma_B', 'sigma_C', 'rho_AC'],
}
SHAPE_IDX = {4: (0, 1), 7: (1, 2)}      # positions of (mu_B, mu_C)
FIT_NDIM = {'fit4': 4, 'fit7': 7}

def get_labels(ndim):
    names = None
    try:
        names = PARAM_NAMES[ndim]
    except KeyError:
        print('invalid dimensions for names?')
    return names


def get_keys(simname, hid, orientation_coords, keys, cachedir='caches',band='i'):

    """
    Pull one or more keys out of a cached face-on/oriented image dict.

    simname            : e.g. 'r431.romulus25.3072g1HsbBH'
    hid                : halo id
    orientation_coords : (theta, phi)
    keys               : str or list of str
    """
    t, p = orientation_coords
    pattern = (f'{cachedir}/{band}band_cache_{simname}_n100/'
               f'*BandStarImages_halo{hid}/*_t{t:+f}_p{p:+f}.pkl')

    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f'No cache file matching {pattern}')

    with open(matches[0], 'rb') as f:
        d = pickle.load(f)

    if isinstance(keys, str):
        return d[keys]
    return [d[k] for k in keys]


_BC_MU_IDX = {4: (0, 1), 7: (1, 2)}
_BC_SIGMA_IDX = {4: (2, 3), 7: (4, 5)}


def plot_bootstrap_ellipses(boot_params, reference_params=None,
                            true_params=None, show_truth=True, intrinsic_data=None,
                            n_ellipses=1000, n_sigma=1.0, seed=0,
                            boot_color='C0', ref_color='C1',
                            boot_label='1 angle / galaxy', ref_label='',#Many angles / galaxy',
                            kf_key='2R_eff', show_contours=False, annotate=True,
                            output_file=None, title=None, **ellipse_kwargs):
    """
    B/A - C/A plane: many-angle reference fit (drawn by plot_ellipsoid_shapes,
    so it matches your other shape plots) with one faint mean +/- n_sigma
    ellipse per bootstrap realisation overlaid.

    boot_params : (n_boot, 4 or 7) point estimates, e.g. boot['fit4']['median']
    reference_params, reference_samples : many-angle max-prob params / samples.
        If reference_params is None, the median of boot_params is used instead.
    true_params : truth vector; drawn only when show_truth=True.
    n_sigma : should match the ellipse size plot_ellipsoid_shapes draws.
    show_contours : False draws the reference as a plain ellipse (also for
        7-param fits); None uses plot_ellipsoid_shapes' default.
    **ellipse_kwargs : passed on to plot_ellipsoid_shapes.
    """
    pts = np.asarray(boot_params, dtype=float)
    ndim = pts.shape[1]
    ib, ic = _BC_MU_IDX[ndim]
    isb, isc = _BC_SIGMA_IDX[ndim]
    #
    reference_params = np.asarray(reference_params, dtype=float)

    fig = plot_ellipsoid_shapes(
        [reference_params], [reference_params],
        true_params_list=([np.asarray(true_params)]
                          if (show_truth and true_params is not None) else None),
        labels=[ref_label], color=[ref_color],
        intrinsic_data=None, kf_key=kf_key,
        show_contours=True, annotate=False,
        focus_on_max_prob=True, show_samples=False, show_ellipses=True,
        **ellipse_kwargs)
    ax = fig.axes[0]

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pts), size=min(n_ellipses, len(pts)), replace=False)
    alpha = float(np.clip(6.0 / len(sel), 0.08, 0.5))  # fainter when there are many
    for k in sel:
        ax.add_patch(Ellipse((pts[k, ib], pts[k, ic]),
                             2 * n_sigma * abs(pts[k, isb]), 2 * n_sigma * abs(pts[k, isc]),
                             fill=False, ec=boot_color, lw=0.8, alpha=alpha, zorder=1.5))
    ax.scatter(pts[:, ib], pts[:, ic], s=15, color=boot_color, alpha=1, lw=0.6, zorder=2.5,edgecolors='white')

    # add the bootstrap entry to the legend plot_ellipsoid_shapes made
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=boot_color, marker='o', ms=4, lw=0.8, alpha=0.8))
    labels.append(f'{boot_label}, {len(pts)} runs'
                  + (f' ({len(sel)} shown)' if len(sel) < len(pts) else ''))
    old = ax.get_legend()
    fontsize = old.get_texts()[0].get_fontsize() if old is not None and old.get_texts() else 9
    ax.legend(handles, labels, loc=old._loc if old is not None else 'best', fontsize=fontsize)

    if title:
        ax.set_title(title)
    print(output_file)
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig

    # ----------------------------------------------------------------------------
    # Method -- paste inside class GalaxyEllipseCollection
    # ----------------------------------------------------------------------------
    def plot_bootstrap_vs_reference(self, boot=None, fit_key='fit4',
                                    reference_dir=None, reference_prefix=None,
                                    show_truth=True, point_estimate='median',
                                    output_file=None, **kwargs):
        """
        Ellipse plot of a bootstrap run (self.bootstrap by default) against the
        many-angle run.

        fit_key : 'fit4' or 'fit7'
        reference_dir, reference_prefix : the output_dir / output_prefix you gave
            run_inference_all_halos. The 7-param reference is read from
            <reference_dir>_covariance, as that function writes it. If omitted,
            the reference stored in `boot` is used (params only, no samples).
        **kwargs : forwarded to plot_bootstrap_ellipses (n_ellipses, n_sigma,
            ref_color, boot_label, ref_label, annotate, show_contours, ...).
        """
        boot = boot or self.bootstrap
        fit = boot[fit_key]
        if fit is None:
            raise ValueError(f'no {fit_key} results in this bootstrap run')

        ref_params = boot.get('reference_params', {}).get(fit_key)
        ref_samples = boot.get('reference_samples', {}).get(fit_key)
        if reference_dir is not None and reference_prefix is not None:
            ref_dir = Path(reference_dir)
            if fit_key == 'fit7':
                ref_dir = ref_dir.with_name(ref_dir.name + '_covariance')
            f = ref_dir / f'{reference_prefix}_results.pkl'
            with open(f, 'rb') as fh:
                res = pickle.load(fh)
            ref_params = np.asarray(res['max_prob_params'], dtype=float).ravel()
            ref_samples = res.get('samples')

        truth = boot['truth']
        if output_file is None:
            output_file = str(Path(boot['output_dir']) /
                              f'{boot["output_prefix"]}{self.band_key}_bootstrap_'
                              f'{fit_key[-1]}p_ellipses_vs_reference.png')

        return plot_bootstrap_ellipses(
            fit[point_estimate], reference_params=ref_params,
            reference_samples=ref_samples,
            true_params=boot['true_params'][fit_key], show_truth=show_truth,
            intrinsic_data=(truth['a_s'], truth['ba_s'], truth['ca_s']),
            kf_key=f'{self.radius}R_eff', output_file=output_file, **kwargs)


# ===========================================================================
# Worker (module level so multiprocessing can pickle it)
# ===========================================================================
def _summarize_chain(samples, max_prob_params, sampler, burn_in, n_keep, rng):
    samples = np.asarray(samples)
    keep = rng.choice(len(samples), size=min(n_keep, len(samples)), replace=False)
    p2_5, p16, p50, p84, p97_5 = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)
    emcee_log = logging.getLogger('emcee')
    old_level = emcee_log.level
    emcee_log.setLevel(logging.ERROR)  # autocorr warnings go via logging, not stdout
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tau = sampler.get_autocorr_time(quiet=True, discard=burn_in)
    except Exception:
        tau = np.full(samples.shape[1], np.nan)
    finally:
        emcee_log.setLevel(old_level)
    return dict(
        map=np.asarray(max_prob_params, dtype=float),
        median=p50, p16=p16, p84=p84, p2_5=p2_5, p97_5=p97_5,
        thinned=samples[keep],
        acceptance=float(np.mean(sampler.acceptance_fraction)),
        tau=np.asarray(tau, dtype=float),
    )


def _bootstrap_worker(task):
    """Run the 4-param (and optionally 7-param) fit on one single-angle draw."""
    i, mcmc_seed, q, a, cfg = task
    np.random.seed(mcmc_seed)  # walker init + emcee use global RNG
    rng = np.random.default_rng(mcmc_seed)
    out = {'i': i, 'seed': mcmc_seed, 'q': q, 'a': a, 'error': None}
    log = io.StringIO()  # swallow prints / progress bars
    try:
        t0 = time.time()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            s4, m4, samp4 = infer_intrinsic_shape(
                q, n_cores=None, output_prefix=None, contain=cfg['contain'],
                **cfg['mcmc_4'])
        out['fit4'] = _summarize_chain(s4, m4, samp4, cfg['mcmc_4']['burn_in'],
                                       cfg['n_keep'], rng)
        out['fit4']['time'] = time.time() - t0
        out['s4'],out['m4'],out['samp4'] = s4, m4, samp4

        if cfg['multivariate']:
            t0 = time.time()
            init = out['fit4'][cfg['init_from']]
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                s7, m7, samp7 = infer_intrinsic_shape_multivariate(
                    q, init, a, n_cores=None, rng=rng, output_prefix=None,
                    contain=cfg['contain'], **cfg['mcmc_7'])
            out['fit7'] = _summarize_chain(s7, m7, samp7, cfg['mcmc_7']['burn_in'],
                                           cfg['n_keep'], rng)
            out['fit7']['time'] = time.time() - t0
            out['s7'],out['m7'],out['samp7'] = s7, m7, samp7
    except Exception as ex:
        out['error'] = f'{type(ex).__name__}: {ex}'
        out['log_tail'] = log.getvalue()[-2000:]
    return out


# ===========================================================================
# Plotting helpers
# ===========================================================================
def plot_bootstrap_marginals(fit, point_key='median', truth=None, reference=None,
                             color='C0', output_file=None, title=None):
    """Histogram of point estimates per parameter vs the typical single-run width."""
    pts = fit[point_key]
    ndim = pts.shape[1]
    names = get_labels(ndim)
    hw = np.median(0.5 * (fit['p84'] - fit['p16']), axis=0)
    ncol = 4
    nrow = int(np.ceil(ndim / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.3 * nrow), squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j >= ndim:
            ax.axis('off')
            continue
        x = pts[:, j]
        m = np.median(x)
        ax.hist(x, bins='auto', density=True, color=color, alpha=0.6,
                label=f'bootstrap {point_key}')
        ax.axvspan(m - hw[j], m + hw[j], color='gray', alpha=0.2,
                   label='typical single-run 68% width')
        if truth is not None:
            ax.axvline(truth[j], color='k', lw=2, label='truth (3D shapes)')
        if reference is not None:
            ax.axvline(reference[j], color='C1', ls='--', lw=2, label='reference (many angles)')
        ax.set_xlabel(names[j])
        ax.set_title(f'scatter σ={np.std(x):.3f} | posterior σ≈{hw[j]:.3f}', fontsize=9)
    axes.flat[0].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    return fig


def plot_bootstrap_calibration(fit, target, target_name, point_key='median',
                               output_file=None, title=None):
    """z-score distribution and credible-interval coverage against a target vector."""
    ndim = fit['median'].shape[1]
    names = get_labels(ndim)
    target = np.asarray(target, dtype=float).ravel()
    hw = 0.5 * (fit['p84'] - fit['p16'])
    z = (fit[point_key] - target) / hw
    cov68 = np.mean((fit['p16'] <= target) & (target <= fit['p84']), axis=0)
    cov95 = np.mean((fit['p2_5'] <= target) & (target <= fit['p97_5']), axis=0)
    n = len(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-5, 5, 41)
    for j in range(ndim):
        ax1.hist(np.clip(z[:, j], -5, 5), bins=bins, density=True,
                 histtype='step', lw=1.5, label=names[j])
    xx = np.linspace(-5, 5, 300)
    ax1.plot(xx, norm.pdf(xx), 'k--', label='N(0,1)')
    ax1.set_xlabel(f'({point_key} − {target_name}) / posterior half-width')
    ax1.set_ylabel('density')
    ax1.legend(fontsize=8)

    idx = np.arange(ndim)
    w = 0.38
    for off, cov, nominal, lab, ls in [(-w / 2, cov68, 0.68, '68%', ':'),
                                       (w / 2, cov95, 0.95, '95%', '--')]:
        err = np.sqrt(np.clip(cov * (1 - cov), 1e-12, None) / n)
        ax2.bar(idx + off, cov, w, yerr=err, capsize=3, label=f'{lab} interval')
        ax2.axhline(nominal, color='k', ls=ls, lw=1)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel(f'fraction of runs containing {target_name}')
    ax2.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    stats = dict(cov68=cov68, cov95=cov95,
                 z_mean=np.mean(z, axis=0), z_std=np.std(z, axis=0))
    return fig, stats


def plot_bootstrap_shape_plane(fit, point_key='median', truth=None, reference=None,
                               intrinsic_data=None, color='C0', output_file=None, title=None):
    """Cloud of (mu_B, mu_C) estimates in the B/A - C/A plane."""
    pts = fit[point_key]
    ib, ic = SHAPE_IDX[pts.shape[1]]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color='gray', lw=0.8)
    if intrinsic_data is not None:
        _, ba, ca = intrinsic_data
        ax.scatter(ba, ca, s=10, color='gray', alpha=0.5, label='galaxies (intrinsic)')
    ax.scatter(pts[:, ib], pts[:, ic], s=12, color=color, alpha=0.6,
               label=f'bootstrap {point_key}s')
    if truth is not None:
        ax.scatter(truth[ib], truth[ic], marker='*', s=250, color='k', zorder=5,
                   label='truth (3D shapes)')
    if reference is not None:
        ax.scatter(reference[ib], reference[ic], marker='s', s=80, color='C1',
                   edgecolor='k', zorder=5, label='reference (many angles)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\mu_B$  (B/A)')
    ax.set_ylabel(r'$\mu_C$  (C/A)')
    ax.legend(fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    return fig


def plot_bootstrap_hist_band(obs_matrix, model_draws, bins, reference=None,
                             xlabel='q', color='C0', ref_label='Projections of intrinsic 3D shapes',
                             output_file=None, title=None):
    """
    16-84% histogram bands: single-angle observed samples (gray) and model
    realisations from each bootstrap's point estimate (colour).
    """
    centers = 0.5 * (bins[1:] + bins[:-1])

    def band(arrs):
        H = np.array([np.histogram(a, bins=bins, density=True)[0] for a in arrs])
        return np.percentile(H, [16, 50, 84], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, _, hi = band(obs_matrix)
    ax.fill_between(centers, lo, hi, step='mid', color='gray', alpha=0.3,
                    label='single-angle mock samples (16–84%)')
    ax.step(centers, np.histogram(np.ravel(obs_matrix), bins, density=True)[0],
            where='mid', color='k', lw=1.5, label='all mock draws pooled')
    lo, med, hi = band(model_draws)
    ax.fill_between(centers, lo, hi, step='mid', color=color, alpha=0.3,
                    label='bootstrap models (16–84%)')
    ax.step(centers, med, where='mid', color=color, lw=1.5, label='bootstrap models (median)')
    if reference is not None:
        ax.step(centers, np.histogram(reference, bins, density=True)[0],
                where='mid', color='C1', ls='--', lw=1.5, label=ref_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    return fig


def plot_bootstrap_diagnostics(fit, n_steps, output_file=None, title=None):
    """Acceptance fraction, autocorrelation time and runtime across iterations."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(fit['acceptance'], bins=20, color='C0')
    axes[0].axvspan(0.2, 0.5, color='g', alpha=0.1)
    axes[0].set_xlabel('mean acceptance fraction')

    tau = fit['tau']
    finite = np.isfinite(tau).any(axis=1)
    tau_max = np.full(len(tau), np.nan)
    tau_max[finite] = np.nanmax(tau[finite], axis=1)
    if np.any(finite):
        axes[1].hist(tau_max[finite], bins=20, color='C0')
    axes[1].axvline(n_steps / 50, color='r', ls='--', label='n_steps / 50')
    axes[1].set_xlabel(f'max autocorrelation time ({np.sum(~finite)} undefined)')
    axes[1].legend(fontsize=8)

    axes[2].hist(fit['time'] / 60, bins=20, color='C0')
    axes[2].set_xlabel('minutes per fit')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
    return fig


def bootstrap_summary_text(fit, point_key, truth=None, reference=None, calib=None):
    ndim = fit[point_key].shape[1]
    pts = fit[point_key]
    hw = np.median(0.5 * (fit['p84'] - fit['p16']), axis=0)
    cols = ['param', 'truth', 'reference', 'median_est', 'scatter', 'post_hw',
            'bias_truth', 'bias/scat', 'bias_ref', 'cov68_t', 'cov95_t', 'cov68_r', 'cov95_r']
    lines = [f'{len(pts)} successful iterations, point estimate = {point_key}',
             ''.join(f'{c:>12s}' for c in cols)]
    fmt = lambda v: f'{v:12.4f}' if v is not None and np.isfinite(v) else f'{"—":>12s}'
    for j, name in enumerate(PARAM_NAMES[ndim]):
        med, sc = np.median(pts[:, j]), np.std(pts[:, j])
        t = None if truth is None else truth[j]
        r = None if reference is None else reference[j]
        ct = (calib or {}).get('truth')
        cr = (calib or {}).get('reference')
        row = [t, r, med, sc, hw[j],
               None if t is None else med - t,
               None if t is None else (med - t) / sc,
               None if r is None else med - r,
               None if ct is None else ct['cov68'][j],
               None if ct is None else ct['cov95'][j],
               None if cr is None else cr['cov68'][j],
               None if cr is None else cr['cov95'][j]]
        lines.append(f'{name:>12s}' + ''.join(fmt(v) for v in row))
    return '\n'.join(lines)


# @memory.cache
# def _build_interpolators_cached(sim_name, halo_id, halo_data, reff_index,
#                                  interpolation_method, coordinate_system):
#     def parse_orientation(orientation):
#         try:
#             x_angle = int(orientation[1:4])
#             y_angle = int(orientation[5:8])
#             return x_angle, y_angle
#         except (ValueError, IndexError):
#             print(f"Invalid orientation format: {orientation}")
#             return None, None
#
#     original_points = []
#     original_values = []
#
#     for orientation, ellipticities in halo_data.items():
#         if not (orientation.startswith('x') and 'y' in orientation):
#             continue
#         x_angle, y_angle = parse_orientation(orientation)
#         original_points.append([x_angle, y_angle])
#         if reff_index < len(ellipticities):
#             original_values.append(float(ellipticities[reff_index]))
#
#     extended_points = original_points.copy()
#     extended_values = original_values.copy()
#
#     if coordinate_system == 'angles':
#         for idx, point in enumerate(original_points):
#             x_angle, y_angle = point
#             if x_angle == 0:
#                 extended_points.append([180, y_angle])
#                 extended_values.append(original_values[idx])
#             elif x_angle == 180:
#                 extended_points.append([0, y_angle])
#                 extended_values.append(original_values[idx])
#             if y_angle == 0:
#                 extended_points.append([x_angle, 360])
#                 extended_values.append(original_values[idx])
#             elif y_angle in (360, 359):
#                 extended_points.append([x_angle, 0])
#                 extended_values.append(original_values[idx])
#             if (x_angle == 0 and y_angle == 0):
#                 extended_points.append([180, 360])
#                 extended_values.append(original_values[idx])
#             elif (x_angle == 180 and y_angle == 0):
#                 extended_points.append([0, 360])
#                 extended_values.append(original_values[idx])
#             elif (x_angle == 0 and y_angle in (360, 359)):
#                 extended_points.append([180, 0])
#                 extended_values.append(original_values[idx])
#             elif (x_angle == 180 and y_angle in (360, 359)):
#                 extended_points.append([0, 0])
#                 extended_values.append(original_values[idx])
#
#     points_array = np.array(extended_points)
#     values_array = np.array(extended_values)
#     valid_mask = ~np.isnan(values_array)
#     points_array = points_array[valid_mask]
#     values_array = values_array[valid_mask]
#
#     interpolator = None
#     fallback_interpolator = None
#
#     if len(values_array) > 0:
#         if interpolation_method == 'rbf':
#             if coordinate_system == 'vectors':
#                 interpolator = Rbf(points_array[:, 0], points_array[:, 1],
#                                    points_array[:, 2], values_array, function='multiquadric')
#             else:
#                 interpolator = Rbf(points_array[:, 0], points_array[:, 1],
#                                    values_array, function='multiquadric')
#         elif interpolation_method == 'nearest':
#             interpolator = NearestNDInterpolator(points_array, values_array)
#         else:
#             interpolator = LinearNDInterpolator(points_array, values_array)
#             fallback_interpolator = NearestNDInterpolator(points_array, values_array)
#
#     return interpolator, fallback_interpolator


#
# class HaloInterpolator:
#     def __init__(self, sim_name, halo_id, halo_data, reff_index,
#                  interpolation_method='linear', coordinate_system='angles'):
#         self._interpolators, self._fallback_interpolators = \
#             _build_interpolators_cached(
#                 sim_name, halo_id, halo_data,
#                 reff_index, interpolation_method, coordinate_system
#             )
#         self.interpolation_method = interpolation_method
#         self.coordinate_system = coordinate_system
#
#     @staticmethod
#     def angles_to_vector(x_angle, y_angle):
#         x_rad = np.radians(x_angle)
#         y_rad = np.radians(y_angle)
#         vx = np.sin(y_rad)
#         vy = 0
#         vz = -np.cos(y_rad)
#         new_vy = vy * np.cos(x_rad) - vz * np.sin(x_rad)
#         new_vz = vy * np.sin(x_rad) + vz * np.cos(x_rad)
#         return [vx, new_vy, new_vz]
#
#     def __call__(self, x_angle, y_angle, reff_index=0):
#
#         if x_angle > 180:
#             x_angle = x_angle % 180
#         if y_angle > 360:
#             y_angle = y_angle % 360
#
#         if self.interpolation_method == 'rbf':
#             if self.coordinate_system == 'vectors':
#                 vx, vy, vz = self.angles_to_vector(x_angle, y_angle)
#                 return float(self._interpolators(vx, vy, vz))
#             else:
#                 return float(self._interpolators(x_angle, y_angle))
#         else:
#             if self.coordinate_system == 'vectors':
#                 point = self.angles_to_vector(x_angle, y_angle)
#             else:
#                 point = [x_angle, y_angle]
#
#             result = self._interpolators(point)
#
#             if self.interpolation_method == 'linear' and np.isnan(result):
#                 result = self._fallback_interpolators(point)
#
#             if len(result) > 1:
#                 print(result)
#             return float(result[0])


def extract_floats(data):
    """
    Parse theta/phi from dict keys in either format:
      - '(np.float64(t), np.float64(p))'
      - '0000_t+0.174533_p+0.000000'
    Accepts a dict (uses its keys), a list of strings, or a single string.
    Returns (thetas, phis) as lists of floats, in the same order encountered.
    If a dict was passed, also returns the corresponding values as a third list,
    so you can zip(thetas, phis, values) safely.
    """
    old_pattern = r'np\.float64\(([^)]+)\)\s*,\s*np\.float64\(([^)]+)\)'
    new_pattern = r't([+-]?[\d.]+)_p([+-]?[\d.]+)'

    is_dict = isinstance(data, dict)
    if isinstance(data, str):
        keys = [data]
    elif is_dict:
        keys = list(data.keys())
    else:
        keys = list(data)

    thetas, phis, values = [], [], []

    for k in keys:
        m = re.search(old_pattern, k)
        if not m:
            m = re.search(new_pattern, k)
        if not m:
            continue  # skip keys that don't match either format
        t, p = m.groups()
        thetas.append(float(t))
        phis.append(float(p))
        if is_dict:
            values.append(data[k])

    if is_dict:
        return thetas, phis, values
    return thetas, phis

# def parse_orientation(orientation):
#     try:
#         x_angle = int(orientation[1:4])
#         y_angle = int(orientation[5:8])
#         return x_angle, y_angle
#     except (ValueError, IndexError):
#         print(f"Invalid orientation format: {orientation}")
#         return None, None

def _sph_to_cart(theta, phi):
    """(theta, phi) on the unit sphere -> (N, 3) Cartesian points."""
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi, dtype=float)
    st = np.sin(theta)
    return np.column_stack((st * np.cos(phi),
                            st * np.sin(phi),
                            np.cos(theta)))

class SphereInterpolator:
    """
    Interpolate values sampled at scattered points on a sphere.

    Parameters
    ----------
    data : dict
        Keys are strings like '(np.float64(theta), np.float64(phi))',
        values are scalars or 1-D arrays (e.g. shape (3,)). All values
        must share the same shape; vector values are interpolated
        component-wise.
    kernel : str
        RBF kernel passed to scipy's RBFInterpolator. 'thin_plate_spline'
        is a good smooth default for sphere data.
    smoothing : float
        0.0 -> exact interpolation through the samples.
        > 0  -> smoothed fit (useful for noisy data).

    Call with f(theta, phi); accepts scalars or arrays (broadcast together).
    """

    def __init__(self, halo_data, kernel="thin_plate_spline", smoothing=0.0):


        thetas, phis = extract_floats(list(halo_data.keys()))
        thetas = np.asarray(thetas)
        phis   = np.asarray(phis)
        n_values = len(thetas)

        values = np.array([np.atleast_1d(v) for v in halo_data.values()], dtype=float)

        # Drop samples containing NaNs (e.g. failed ellipticity fits at some
        # viewing angles) so they don't poison the interpolant.
        good = ~np.isnan(values).any(axis=1)
        self.n_dropped = int((~good).sum())
        if self.n_dropped:
            thetas, phis, values = thetas[good], phis[good], values[good]
            if self.n_dropped > 0.4*n_values:
                raise ValueError(f'(Too many nans {self.n_dropped}/{n_values})')
        if len(values) == 0:
            raise ValueError("all samples are NaN; nothing to interpolate")

        self._scalar = (values.shape[1] == 1)

        pts = _sph_to_cart(thetas, phis)

        # Deduplicate points that coincide on the sphere (e.g. the poles,
        # or phi = 0 vs phi = 2*pi given at the same theta). RBFInterpolator
        # requires strictly distinct nodes; duplicates are averaged.
        pts_r = np.round(pts, 12)
        _, idx, inv = np.unique(pts_r, axis=0, return_index=True,
                                return_inverse=True)
        if len(idx) < len(pts):
            vals_u = np.zeros((len(idx), values.shape[1]))
            counts = np.zeros(len(idx))
            for i, g in enumerate(inv):
                vals_u[g] += values[i]
                counts[g] += 1
            values = vals_u / counts[:, None]
            pts = pts[idx]

        self._rbf = RBFInterpolator(pts, values, kernel=kernel,
                                    smoothing=smoothing)

    def __call__(self, theta, phi):
        theta = np.asarray(theta, dtype=float)
        phi   = np.asarray(phi, dtype=float)
        theta, phi = np.broadcast_arrays(theta, phi)
        shape = theta.shape

        if np.any((theta < 0) | (theta > np.pi)):
            raise ValueError("theta must lie in [0, pi]")
        phi = np.mod(phi, 2 * np.pi)  # enforce periodicity on input too

        out = self._rbf(_sph_to_cart(theta.ravel(), phi.ravel()))

        if self._scalar:
            out = out[:, 0].reshape(shape)
            return out.item() if shape == () else out
        return out.reshape(shape + (out.shape[1],)) if shape else out[0]




def inverse_transform_intrinsic(theta, phi):
    """Given spherical angles, compute required intrinsic X and Y rotations

    This function is already vectorized - it works with arrays or scalars
    """
    Y = np.arcsin(np.clip(np.sin(theta) * np.cos(phi), -1, 1))
    X = np.arctan2(-np.sin(theta) * np.sin(phi), np.cos(theta))
    return X, Y


def spherical_to_rotation_angles(theta, phi):
    """
    Convert spherical coordinates to intrinsic X and Y rotation angles.
    Fully vectorized to handle arrays of inputs.

    Parameters:
        theta: Polar angle from z-axis (0 to π) in radians
               Can be scalar or array
        phi: Azimuthal angle in xy-plane from x-axis (0 to 2π) in radians
             Can be scalar or array

    Returns:
        tuple: (X_deg, Y_deg) rotation angles in degrees
               Same shape as input
    """
    # Use the inverse transform to get rotation angles
    X_rad, Y_rad = inverse_transform_intrinsic(theta, phi)

    # Convert to numpy arrays to ensure consistent behavior
    X_rad = np.asarray(X_rad)
    Y_rad = np.asarray(Y_rad)

    # Apply angle adjustments using vectorized operations
    # First condition: X_rad < -0.01
    mask1 = X_rad < -0.01
    X_rad = np.where(mask1, X_rad + np.pi, X_rad)
    Y_rad = np.where(mask1, np.pi - Y_rad, Y_rad)

    # Second condition: X_rad >= π (only apply if first condition was False)
    mask2 = (~mask1) & (X_rad >= np.pi)
    X_rad = np.where(mask2, X_rad - np.pi, X_rad)
    Y_rad = np.where(mask2, np.pi - Y_rad, Y_rad)

    # Special case: if Y_rad is negative, add 2π
    mask3 = Y_rad < 0
    Y_rad = np.where(mask3, Y_rad + 2 * np.pi, Y_rad)

    # Convert to degrees
    X_deg = np.degrees(X_rad)
    Y_deg = np.degrees(Y_rad)

    return X_deg, Y_deg


# def random_viewing_angles(n):
#     """
#     Generate n random viewing angles uniformly distributed on a sphere.
#
#     This function is already vectorized.
#
#     Parameters:
#         n (int): Number of viewing angles to generate
#
#     Returns:
#         tuple: (X, Y) rotation angles in degrees for n viewing positions
#     """
#     phi = np.random.uniform(0, 2 * np.pi, n)
#     nu = np.random.uniform(0, 1, n)
#     theta = np.arccos(2 * nu - 1)
#     X, Y = spherical_to_rotation_angles(theta, phi)
#     return X, Y




def _in_support(a, b, c):
    """Boolean mask for the physical region A>0, 0<C<B<1."""
    return (a > 0) & (c > 0) & (b > c) & (b < 1)

def cov_fit(a, b, c, verbose=True):
    """
    Moment-matched fit returning theta in the (sigma, rho) convention.

    Fast and exact for an *untruncated* Gaussian; slightly biased when the
    truncation at the (0, 1) boundaries clips real probability mass.  Use
    fit_truncated_mle() afterwards if that bias matters.
    """
    a, b, c = map(np.asarray, (a, b, c))
    mask = _in_support(a, b, c)
    if not mask.all():
        if verbose:
            print(f"cov_fit(): dropping {(~mask).sum()} points outside the physical region")
        a, b, c = a[mask], b[mask], c[mask]

    mu_a, mu_b, mu_c = a.mean(), b.mean(), c.mean()
    sig_a = a.std(ddof=1)
    sig_b = b.std(ddof=1)
    sig_c = c.std(ddof=1)

    s_ac = np.cov(a, c, ddof=1)[0, 1]
    rho_ac = s_ac / (sig_a * sig_c)
    # sample correlations can land on +/-1 for tiny n; clip to stay PD
    rho_ac = float(np.clip(rho_ac, -1.0 + 1e-6, 1.0 - 1e-6))
    params = np.array([mu_a, mu_b, mu_c, sig_a, sig_b, sig_c, rho_ac])
    #check if measured params are attainable
    # if attainable(params):
    return params
    # else:
    #     raise ValueError(f"cov_fit(): cov_fit failed for {params}")



# Loading function remains the same
def load_and_process_halo_data(sim_name=None, halo_id=None, pickle_filename='ellipse_data.pickle'):
    """Load ellipse data from the pickle file for a specific halo."""
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)

    if sim_name is None:
        sim_name = list(ellipse_dict.keys())[0]
    if sim_name not in ellipse_dict:
        raise ValueError(f"Simulation '{sim_name}' not found")

    if halo_id is None:
        halo_id = list(ellipse_dict[sim_name].keys())[0]
    if halo_id not in ellipse_dict[sim_name]:
        print(ellipse_dict[sim_name].keys())
        raise ValueError(f"Halo ID {halo_id} not found")

    return ellipse_dict[sim_name][halo_id]


def load_results(output_prefix, output_dir="results"):
    """
    Load saved inference results.

    Parameters:
        output_prefix (str): Prefix of saved files
        output_dir (str): Directory containing results

    Returns:
        dict: Dictionary of loaded results
    """
    try:
        with open(f"{output_dir}/{output_prefix}_results.pkl", 'rb') as f:
            results = pickle.load(f)
        return results
    except FileNotFoundError:
        # Try loading individual files if the pickle doesn't exist
        results = {}

        try:
            results['samples'] = np.load(f"{output_dir}/{output_prefix}_samples.npy")
            results['max_prob_params'] = np.load(f"{output_dir}/{output_prefix}_max_prob_params.npy")
            results['log_probs'] = np.load(f"{output_dir}/{output_prefix}_log_probs.npy")
            results['q_obs'] = np.load(f"{output_dir}/{output_prefix}_q_obs.npy")

            # Try to load the full chain if it exists
            try:
                results['full_chain'] = np.load(f"{output_dir}/{output_prefix}_full_chain.npy")
            except FileNotFoundError:
                pass

            return results
        except Exception as e:
            raise Exception(f"Error loading results for {output_prefix}: {e}")

# Add the check_existing_results function from your test code



def assess_parameter_recovery(max_prob_params, true_params,
                              param_names=['μ_B', 'μ_C', 'σ_B', 'σ_C'],
                              param_errors=None,
                              q_obs=None, q_model=None, q_true=None):
    """
    Assess how well the maximum probability MCMC estimates recover true parameter values.

    Parameters:
        max_prob_params: Array of maximum probability parameter estimates [mu_B, mu_C, sigma_B, sigma_C]
        true_params: Array of true parameter values [B_true, C_true, B_err_true, C_err_true]
        param_names: Names of parameters for display
        param_errors: Optional array of parameter uncertainties (e.g., from posterior std)
        q_obs: Optional array of observed distribution values
        q_model: Optional array of model distribution values (from estimated parameters)
        q_true: Optional array of true distribution values (from true parameters)

    Returns:
        dict: Recovery statistics including 'output_text' for saving to file
    """
    import numpy as np
    from scipy import stats

    results = {
        'param_names': param_names,
        'true_values': true_params,
        'estimated_values': max_prob_params,
        'bias': [],
        'fractional_bias': [],
        'absolute_error': [],
    }

    if param_errors is not None:
        results['param_errors'] = param_errors
        results['sigma_from_true'] = []
        results['within_1sigma'] = []
        results['within_2sigma'] = []

    # Build output string
    lines = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("PARAMETER RECOVERY ASSESSMENT (Maximum Probability Estimates)")
    lines.append("=" * 80)

    if param_errors is not None:
        lines.append(f"{'Parameter':<10} {'True':<10} {'Estimated':<12} {'Error':<10} {'Bias':<12} {'σ away':<10}")
        lines.append("-" * 80)
    else:
        lines.append(f"{'Parameter':<10} {'True':<10} {'Estimated':<12} {'Bias':<15} {'Frac Bias':<12}")
        lines.append("-" * 80)

    for i, (name, true_val, est_val) in enumerate(zip(param_names, true_params, max_prob_params)):
        bias = est_val - true_val
        frac_bias = (bias / true_val * 100) if true_val != 0 else np.nan
        abs_error = abs(bias)

        results['bias'].append(bias)
        results['fractional_bias'].append(frac_bias)
        results['absolute_error'].append(abs_error)

        if param_errors is not None:
            sigma_away = abs(bias) / param_errors[i] if param_errors[i] > 0 else np.nan
            results['sigma_from_true'].append(sigma_away)
            results['within_1sigma'].append(sigma_away <= 1)
            results['within_2sigma'].append(sigma_away <= 2)

            lines.append(
                f"{name:<10} {true_val:<10.4f} {est_val:<12.4f} {param_errors[i]:<10.4f} {bias:+.4f}      {sigma_away:<10.2f}")
        else:
            lines.append(f"{name:<10} {true_val:<10.4f} {est_val:<12.4f} {bias:+.4f}         {frac_bias:+.1f}%")

    lines.append("-" * 80)

    # Summary statistics
    avg_abs_error = np.mean(results['absolute_error'])
    avg_frac_bias = np.nanmean(np.abs(results['fractional_bias']))

    lines.append("")
    lines.append("Summary:")
    lines.append(f"  Mean absolute error: {avg_abs_error:.4f}")
    lines.append(f"  Mean |fractional bias|: {avg_frac_bias:.1f}%")

    if param_errors is not None:
        avg_sigma_away = np.nanmean(results['sigma_from_true'])
        n_within_1sigma = sum(results['within_1sigma'])
        n_within_2sigma = sum(results['within_2sigma'])
        lines.append(f"  Average σ from true: {avg_sigma_away:.2f}")
        lines.append(f"  Within 1σ: {n_within_1sigma}/{len(param_names)}")
        lines.append(f"  Within 2σ: {n_within_2sigma}/{len(param_names)}")

    # Interpretation
    lines.append("")
    lines.append("Interpretation:")
    if avg_frac_bias < 5:
        lines.append("  ✓ Excellent recovery: < 5% average bias")
    elif avg_frac_bias < 10:
        lines.append("  ✓ Good recovery: < 10% average bias")
    elif avg_frac_bias < 20:
        lines.append("  ~ Moderate recovery: 10-20% average bias")
    else:
        lines.append("  ✗ Poor recovery: > 20% average bias")

    # ========================================================================
    # DISTRIBUTION COMPARISON TESTS
    # ========================================================================
    if q_obs is not None and q_model is not None:
        lines.append("")
        lines.append("=" * 80)
        lines.append("DISTRIBUTION COMPARISON TESTS")
        lines.append("=" * 80)

        results['distribution_tests'] = {}

        # Test 1: Observed vs Model
        lines.append("")
        lines.append("1. Observed vs Model Distribution")
        lines.append("-" * 80)

        ks_stat_om, ks_pval_om = stats.ks_2samp(q_obs, q_model)
        emd_om = stats.wasserstein_distance(q_obs, q_model)

        results['distribution_tests']['obs_vs_model'] = {
            'ks_statistic': ks_stat_om,
            'ks_pvalue': ks_pval_om,
            'earth_movers_distance': emd_om
        }

        lines.append(f"  Kolmogorov-Smirnov Test:")
        lines.append(f"    Statistic: {ks_stat_om:.6f}")
        lines.append(f"    p-value:   {ks_pval_om:.6f}")
        if ks_pval_om > 0.05:
            lines.append(f"    → Cannot reject null (p > 0.05): distributions are similar")
        else:
            lines.append(f"    → Reject null (p ≤ 0.05): distributions are different")

        lines.append(f"  Earth Mover's Distance: {emd_om:.6f}")

        # Test 2: Observed vs True (if q_true provided)
        if q_true is not None:
            lines.append("")
            lines.append("2. Observed vs True Distribution")
            lines.append("-" * 80)

            ks_stat_ot, ks_pval_ot = stats.ks_2samp(q_obs, q_true)
            emd_ot = stats.wasserstein_distance(q_obs, q_true)

            results['distribution_tests']['obs_vs_true'] = {
                'ks_statistic': ks_stat_ot,
                'ks_pvalue': ks_pval_ot,
                'earth_movers_distance': emd_ot
            }

            lines.append(f"  Kolmogorov-Smirnov Test:")
            lines.append(f"    Statistic: {ks_stat_ot:.6f}")
            lines.append(f"    p-value:   {ks_pval_ot:.6f}")
            if ks_pval_ot > 0.05:
                lines.append(f"    → Cannot reject null (p > 0.05): distributions are similar")
            else:
                lines.append(f"    → Reject null (p ≤ 0.05): distributions are different")

            lines.append(f"  Earth Mover's Distance: {emd_ot:.6f}")

            # Test 3: Model vs True
            lines.append("")
            lines.append("3. Model vs True Distribution")
            lines.append("-" * 80)

            ks_stat_mt, ks_pval_mt = stats.ks_2samp(q_model, q_true)
            emd_mt = stats.wasserstein_distance(q_model, q_true)

            results['distribution_tests']['model_vs_true'] = {
                'ks_statistic': ks_stat_mt,
                'ks_pvalue': ks_pval_mt,
                'earth_movers_distance': emd_mt
            }

            lines.append(f"  Kolmogorov-Smirnov Test:")
            lines.append(f"    Statistic: {ks_stat_mt:.6f}")
            lines.append(f"    p-value:   {ks_pval_mt:.6f}")
            if ks_pval_mt > 0.05:
                lines.append(f"    → Cannot reject null (p > 0.05): distributions are similar")
            else:
                lines.append(f"    → Reject null (p ≤ 0.05): distributions are different")

            lines.append(f"  Earth Mover's Distance: {emd_mt:.6f}")

            # Comparative summary
            lines.append("")
            lines.append("Summary of Distribution Comparisons:")
            lines.append("-" * 80)
            lines.append(f"  {'Comparison':<25} {'KS Stat':<12} {'KS p-value':<12} {'EMD':<12}")
            lines.append(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
            lines.append(f"  {'Observed vs Model':<25} {ks_stat_om:<12.6f} {ks_pval_om:<12.6f} {emd_om:<12.6f}")
            lines.append(f"  {'Observed vs True':<25} {ks_stat_ot:<12.6f} {ks_pval_ot:<12.6f} {emd_ot:<12.6f}")
            lines.append(f"  {'Model vs True':<25} {ks_stat_mt:<12.6f} {ks_pval_mt:<12.6f} {emd_mt:<12.6f}")

            lines.append("")
            lines.append("Interpretation:")
            if emd_mt < emd_ot:
                lines.append(f"  ✓ Model is closer to truth than observations (EMD: {emd_mt:.6f} < {emd_ot:.6f})")
            else:
                lines.append(f"  → Model is farther from truth than observations (EMD: {emd_mt:.6f} ≥ {emd_ot:.6f})")

            if ks_pval_om > 0.05:
                lines.append(f"  ✓ Model adequately represents observed data (KS p = {ks_pval_om:.4f})")
            else:
                lines.append(f"  ✗ Model differs significantly from observed data (KS p = {ks_pval_om:.4f})")

    # Convert lists to arrays for easier use
    results['bias'] = np.array(results['bias'])
    results['fractional_bias'] = np.array(results['fractional_bias'])
    results['absolute_error'] = np.array(results['absolute_error'])
    results['avg_fractional_bias'] = avg_frac_bias
    results['avg_absolute_error'] = avg_abs_error

    # Store the full output text
    output_text = '\n'.join(lines)
    results['output_text'] = output_text

    # Print to terminal
    print(output_text)

    return results

def check_existing_results(output_dir, prefix):
    """
    Check if results already exist for the given prefix.

    Parameters:
        output_dir (str): Directory to check
        prefix (str): Prefix for result files

    Returns:
        tuple or None: (samples, max_prob_params, chain) if results exist, None otherwise
    """
    results_path = Path(output_dir) / f"{prefix}_results.pkl"

    if results_path.exists():
        print(f"Found existing results at {results_path}")
        try:
            results = load_results(prefix, output_dir)

            # Check if we have all required components
            if all(k in results for k in ['samples', 'max_prob_params', 'full_chain', 'q_obs', 'a_obs', 'sampler']):
                print("Loaded existing MCMC results.")
                return results['samples'], results['max_prob_params'], results['full_chain'],results['q_obs'], results['a_obs'], results['sampler']
            else:
                print("Existing results incomplete. Will rerun analysis.")
                return None

        except Exception as e:
            print(f"Error loading existing results: {e}")
            return None

    return None


def shape_at_semi_major(a_s_func, b_s_func, c_s_func, a_eval, rbins,
                        n_grid=2048):
    """Axis ratios of the shell whose semi-major axis is `a_eval` (kpc).

    Inverse of shape_semi_major. a_s_func(r) is sampled on a dense grid over
    [r_min, r_max] and interpolated the other way, so `a_eval` should be an
    isophote sma measured in kpc. Values outside the sampled range of a give NaN
    rather than being clamped.

    Returns (rbins, ba, ca), floats if `a_eval` is scalar.
    """
    a_eval_arr = np.asarray(a_eval, dtype=float)
    #a_flat = np.atleast_1d(a_eval_arr)
    r_min = np.min(rbins)
    r_max = np.max(rbins)

    r = np.linspace(r_min, r_max, n_grid)
    a = np.asarray(a_s_func(r), dtype=float)
    b = np.asarray(b_s_func(r), dtype=float)
    c = np.asarray(c_s_func(r), dtype=float)

    i = np.argmin(np.abs(a-a_eval))
    error = np.abs(a[i]-a_eval)
    # if error/a_eval > 0.3:
    #     print(f'error in a = {error/a_eval:.2f},r_bin={r[i]:.2f},a={a[i]:.2f},a_eval={a_eval:.2f},r_max={r[-1]:.2f},i={i}')
    return r[i],a[i],b[i],c[i]

    # with np.errstate(invalid='ignore'):
    #     ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 0)
    # r, a, b, c = r[ok], a[ok], b[ok], c[ok]
    #
    # # a -> r must be single valued
    # if a.size:
    #     keep = np.concatenate(([True], np.diff(a) > 0))
    #     r, a, b, c = r[keep], a[keep], b[keep], c[keep]
    #
    # if a.size < 2:
    #     out = [np.full(a_flat.shape, np.nan)] * 3
    # else:
    #     inside = (a_flat >= a[0]) & (a_flat <= a[-1]) & np.isfinite(a_flat)
    #     with np.errstate(invalid='ignore'):
    #         out = [np.where(inside, np.interp(a_flat, a, v), np.nan)
    #                for v in (r, b / a, c / a)]
    # if scalar_input:
    #     return tuple(float(v.reshape(-1)[0]) for v in out)
    # return tuple(v.reshape(a_eval_arr.shape) for v in out)


class GalaxyEllipseCollection:
    """
    A collection of galaxy ellipse data with methods to generate observed axis ratio distributions
    and run shape inference using MCMC.
    """

    def __init__(self,radius,reff_index,run_covariances=False,band_key=''):
        """Initialize an empty collection of galaxy ellipses."""
        #self.halos = {}  # Dictionary to store halo data {(sim_name, halo_id): {data}}
        self.radius = radius
        self.interpolators = {}  # Dictionary to store interpolation functions
        self.reff_interpolators = {}
        self.n_steps = 30000
        self.halo_data = {}  # Store raw halo data for reference
        self.reff_index = reff_index
        self.run_covariances = run_covariances
        self.band_key=band_key


    def add_halo(self, sim_name, halo_id, halo_data):
        halo_key = (sim_name, halo_id)
        #self.halo_data[halo_key] = halo_data  # Store raw data for reference
        #self.halos[halo_key] = halo_data


        data = halo_data.copy()
        if 'a_s' in data.keys():
            del data['a_s']
            del data['b_s']
            del data['c_s']
            del data['rbins']
            del data['a_s_i']
            del data['b_s_i']
            del data['c_s_i']
            del data['rbins_i']
        eps_dict={}
        for orientation in data:
            eps_dict[orientation] = data[orientation]['eps'][self.reff_index]

        # a_eval = data['0099_t+0.000000_p+0.000000']['r_kpc'][self.reff_index]



        interpolator = SphereInterpolator(eps_dict)
        self.interpolators[halo_key] = interpolator
        reff_dict = self.create_reff_dict(data,sim_name,halo_id)
        self.reff_interpolators[halo_key] = SphereInterpolator(reff_dict)
        self.face_on_reff = self.reff_interpolators[halo_key](0,0)
        #print(self.face_on_reff)
        a_eval = 3*self.face_on_reff
        assert np.isfinite(a_eval), f'invalid face-on reff for {halo_key}: {a_eval}'


        rbins = halo_data['rbins']
        a_s_func = halo_data['a_s']
        b_s_func = halo_data['b_s']
        c_s_func = halo_data['c_s']
        # if a_eval > rbins[-1]:
        #     #check if they are somewhat close, we can keep if so.
        #     if a_eval * 0.9 < rbins[-1]:
        #         a_eval *= 0.9
        # assert rbins[0] < a_eval < rbins[-1], f'a_eval not in range of rbins r_min={rbins[0]:.2f},r_max={rbins[-1]:.2f}, a_eval={a_eval:.2f}, {halo_key}'

        r, a, b, c = shape_at_semi_major(a_s_func, b_s_func, c_s_func, a_eval,rbins)
        halo_data['ba_s'] = b/a
        halo_data['ca_s'] = c/a
        halo_data['a_s'] = a


        rbins_i = halo_data['rbins_i']
        a_s_func_i = halo_data['a_s_i']
        b_s_func_i = halo_data['b_s_i']
        c_s_func_i = halo_data['c_s_i']
        # if a_eval > rbins[-1]:
        #     #check if they are somewhat close, we can keep if so.
        #     if a_eval * 0.9 < rbins[-1]:
        #         a_eval *= 0.9
        # assert rbins[0] < a_eval < rbins[-1], f'a_eval not in range of rbins r_min={rbins[0]:.2f},r_max={rbins[-1]:.2f}, a_eval={a_eval:.2f}, {halo_key}'

        r, a, b, c = shape_at_semi_major(a_s_func_i, b_s_func_i, c_s_func_i, a_eval, rbins_i)


        halo_data['ba_s_i'] = b/a
        halo_data['ca_s_i'] = c/a
        halo_data['a_s_i'] = a
        assert np.abs(a-a_eval)/a_eval < 0.4, f'error between requested and available semi-major axis too large: {np.abs(a-a_eval)/a_eval}'

            #print(f'Warning, large error in a_eval {a:.2f}, {a_eval:.2f} {halo_key}')
        assert np.isfinite(b/a), f'invalid interpolated shape for {halo_key}: {rbins, b/a, c/a, a_eval}'
        self.halo_data[halo_key] = halo_data



    def create_reff_dict(self,halo_data,sim_name,halo_id):
        thetas, phis = extract_floats(list(halo_data.keys()))
        reff_dict = {}
        for theta,phi,key in zip(thetas,phis,halo_data.keys()):
            reff = get_keys(sim_name, halo_id, (theta, phi), 'Reff')
            reff_dict[key] = reff
        return reff_dict



    def copy_halo_from(self, other_collection, sim_name, halo_id):
        """Copy a halo and its prebuilt interpolator from another collection."""
        halo_key = (sim_name, halo_id)
        #self.halos[halo_key] = other_collection.halos[halo_key]
        self.interpolators[halo_key] = other_collection.interpolators[halo_key]
        self.halo_data[halo_key] = other_collection.halo_data[halo_key]
        self.reff_interpolators[halo_key] = other_collection.reff_interpolators[halo_key]

    def generate_q_distribution_single_halo(self, sim_name, halo_id, n_angles):
        """
        Generate q (axis ratio) distribution for a single halo with random viewing angles.

        Parameters:
            sim_name (str): Simulation name
            halo_id (int/str): Halo identifier
            n_angles (int): Number of random viewing angles

        Returns:
            array: q values (axis ratios) for the specified halo
        """
        halo_key = (sim_name, halo_id)
        if halo_key not in self.halo_data:
            raise KeyError(f"Halo {halo_id} from simulation {sim_name} not found in collection")

        interpolator = self.interpolators[halo_key]
        reff_interpolator = self.reff_interpolators[halo_key]
        phi,theta = random_viewing_angles(n_angles)
        e = interpolator(theta,phi)#[:,self.reff_index]
        q_values = 1 - e
        reff_values = reff_interpolator(theta, phi)

        return q_values, reff_values

    def generate_q_distribution_all_halos(self, angles_per_halo):
        """
        Generate q distribution from all halos using random viewing angles.

        Parameters:
            n_total_angles (int): Total number of random viewing angles across all halos
            reff_index (int): Index into reff_multipliers to use
            weighted (bool): If True, sample each halo with equal probability
                            If False, allocate angles proportionally to number of halos

        Returns:
            array: q values (axis ratios) from all halos
        """
        if not self.halo_data:
            raise ValueError("No halos in collection")

        q_values = []
        reff_values = []
        i = 0
        j = 0
        
        for halo_key in self.halo_data.keys():
            interpolator = self.interpolators[halo_key]
            reff_interpolator = self.reff_interpolators[halo_key]
            # Generate random viewing angles for this halo
            phi, theta = random_viewing_angles(angles_per_halo)
            # Get q values for each angle
            try:
                e = interpolator(theta,phi)#[:,self.reff_index]
                q = 1 - e
                q_values.extend(q)
                reffs = reff_interpolator(theta, phi)
                reff_values.extend(reffs)
                i = i + 1
            except Exception as ex:
                print(f"Error generating q values for halo {halo_key}: {ex}")
                j= j + 1
                #count number of successful and failed halos


        print(f'Generated q values from {i} halos, failed for {j} halos.')
        #make sure we don't have anything unusal here like nans?
        if np.any(np.isnan(q_values)):
            raise ValueError(f"q values for halo {halo_key} are not nan")
        if np.any(np.isnan(reff_values)):
            raise ValueError(f"reff_values for halo {halo_key} are not nan")
        a_obs=np.array(reff_values)*self.radius

        return np.array(q_values), np.log10(a_obs)+3 #get radius we want, then add 2 to make sure all positive

    def generate_q_distribution_all_halos_sideon(self):
        if not self.halo_data:
            raise ValueError("No halos in collection")

        q_values = []
        i = 0
        j = 0
        sideon_orientations = [f'x090y{i:03d}' for i in np.arange(0, 330, 30)]
        for halo_key in self.halo_data.keys():
            halo_data = self.halo_data[halo_key]
            e = np.nanmax([halo_data[orientation][self.reff_index] for orientation in sideon_orientations])
            q = 1 - e
            i = i + 1
            q_values.append(q)



        print(f'Generated q values from {i} halos, failed for {j} halos.')

        return np.array(q_values)
    

    def run_or_load_inference(self, output_dir, output_prefix,
                              q_input=None, a_input=None,
                              generate_q=False, n_angles_per_halo=100,
                              multivariate=False, init_params=None,
                              n_walkers=32, n_steps=3000, burn_in=500,
                              n_cores=None, contain=True,
                              force_rerun=False, save_q_obs=False,
                              plot=True, plot_dir=None, plot_prefix=None,
                              plot_burn_in=None,
                              q_plot_key='q_obs', a_plot_key=None,
                              q_model_from_fit=False, q_model_n=None,
                              **plot_kwargs):
        """
        Load cached inference results if they exist, otherwise run the MCMC and
        cache them, then plot.

        Parameters:
            output_dir (str/Path): Where the .pkl / .npy live and are written.
            output_prefix (str): Prefix used by check_existing_results and the pickle.
            q_input, a_input (array): Arrays to fit. Ignored when generate_q=True
                or when a cached result is found.
            generate_q (bool): Call self.generate_q_distribution_all_halos instead
                of using q_input/a_input.
            n_angles_per_halo (int): Only used when generate_q=True.
            multivariate (bool): Use infer_intrinsic_shape_multivariate (7-param)
                instead of infer_intrinsic_shape (4-param).
            init_params (array): Starting parameters for the multivariate fit.
            save_q_obs (bool): Also write "<prefix>_q_obs.npy".
            plot (bool): Call self.plot_results at the end.
            plot_dir, plot_prefix, plot_burn_in: Override output_dir / output_prefix
                / burn_in for the plots only. Default to the run values.
            q_plot_key, a_plot_key (str): Which plot_results argument the fitted
                q / a arrays should be passed as ('q_obs' vs 'q_true', 'a_obs').
            q_model_from_fit (bool): Regenerate q_model from the fitted
                max-probability parameters before plotting.
            q_model_n (int): Number of projections for q_model_from_fit. Defaults
                to len(q_obs) as passed to the plot.
            **plot_kwargs: Forwarded to plot_results (true_params, q_true, a_true,
                label, color, intrinsic_data, labels, ...).

        Returns:
            (samples, max_prob_params, chain, q, a, sampler)
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        q, a = q_input, a_input

        existing_results = None
        if not force_rerun:
            existing_results = check_existing_results(str(out_dir), output_prefix)

        if existing_results:
            samples, max_prob_params, chain, q, a, sampler = existing_results
            print(f"Using existing inference results: {out_dir}/{output_prefix}")
        else:
            start_time = time.time()

            if generate_q:
                q, a = self.generate_q_distribution_all_halos(n_angles_per_halo)
            assert len(q) == len(a), (
                f'lengths of observed arrays do not match, q {len(q)}, a {len(a)}')

            if save_q_obs:
                np.save(out_dir / f"{output_prefix}_q_obs.npy", q)

            sampler_args = (q, init_params, a) if multivariate else (q,)
            sampler_fn = (infer_intrinsic_shape_multivariate if multivariate
                          else infer_intrinsic_shape)
            samples, max_prob_params, sampler = sampler_fn(
                *sampler_args,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                n_cores=n_cores,
                output_prefix=output_prefix,
                output_dir=str(out_dir),
                contain=contain
            )

            print(f"Inference completed in {time.time() - start_time:.2f} seconds")
            chain = sampler.get_chain()
            print(f"[{output_prefix}] max-probability parameters: {max_prob_params}")

            results = {
                'samples': samples,
                'max_prob_params': max_prob_params,
                'full_chain': chain,
                'q_obs': q,
                'a_obs': a,
                'sampler': sampler,
            }
            with open(out_dir / f"{output_prefix}_results.pkl", 'wb') as f:
                pickle.dump(results, f)

        if plot:
            plot_kwargs[q_plot_key] = q
            if a_plot_key is not None:
                plot_kwargs[a_plot_key] = a
            #if q_model_from_fit:
            n_model = q_model_n if q_model_n is not None else len(plot_kwargs['q_obs'])
            q_model,_= sample_model_projections(max_prob_params, n_model)
            plot_kwargs['q_model']=q_model
            self.q_model=q_model


            self.plot_results(
                samples=samples,
                max_prob_params=max_prob_params,
                chain=chain,
                burn_in=burn_in if plot_burn_in is None else plot_burn_in,
                output_prefix=output_prefix if plot_prefix is None else plot_prefix,
                output_dir=str(out_dir if plot_dir is None else plot_dir),
                **plot_kwargs
            )

        return samples, max_prob_params, chain, q, a, sampler

    # ------------------------------------------------------------------
    def run_inference_all_halos(self, n_angles_per_halo=100, weighted=False,
                                n_walkers=32, n_steps=3000, burn_in=500, n_cores=None,
                                output_prefix="all_halos", output_dir="results",
                                force_rerun=False,
                                label="All Halos Combined", color="blue",
                                contain=True):
        """
        Run shape inference on a distribution from all halos.

        (docstring unchanged)
        """
        full_output_dir = Path(output_dir)
        full_output_dir.mkdir(parents=True, exist_ok=True)

        # --- intrinsic shapes: true_params by averaging over all halos ---
        ba_s = []
        ca_s = []
        a_s = []
        ba_s_i = []
        ca_s_i = []
        a_s_i = []

        for halo_key in self.halo_data.keys():
            halo_data = self.halo_data[halo_key]
            ba_s.append(halo_data['ba_s'])
            ca_s.append(halo_data['ca_s'])
            a_s.append(halo_data['a_s'])
            ba_s_i.append(halo_data['ba_s_i'])
            ca_s_i.append(halo_data['ca_s_i'])
            a_s_i.append(halo_data['a_s_i'])
        ba_s = np.array(ba_s)
        ca_s = np.array(ca_s)
        a_s = np.log10(np.array(a_s)) + 3
        ba_s_i = np.array(ba_s_i)
        ca_s_i = np.array(ca_s_i)
        a_s_i = np.array(a_s_i) + 3

        true_params, _ = fit_ellipsoid_params(ba_s, ca_s)

        q_true = []
        a_true = []
        q_i_true = []
        a_i_true = []
        for i in range(len(ba_s)):
            phi, theta = random_viewing_angles(n_angles_per_halo)
            alpha, beta = projected_semi_axes(phi, theta, ba_s[i], ca_s[i])
            q = beta / alpha
            a = np.log10((10 ** (a_s[i] - 3) * np.sqrt(alpha * beta))) + 3
            # q = project_ellipticity_xu(theta,phi)
            q_true.extend(q)
            a_true.extend(a)
            alpha, beta = projected_semi_axes(phi, theta, ba_s[i], ca_s[i])
            q = beta / alpha
            a = np.log10((10 ** (a_s[i] - 3) * np.sqrt(alpha * beta))) + 3
            q_i_true.extend(q)
            a_i_true.extend(a)

        if np.any(np.isnan(ba_s)):
            print(f' nans found in shape array')
            print(a_s)
            print(ba_s)
            print(ca_s)

        if self.run_covariances:
            theta_data = cov_fit(a_s, ba_s, ca_s)  # intrinsic shapes from the sim
            from shape_inference import attainable
            print(theta_data, attainable(theta_data))

        if not is_color_like(color):
            color = None

        # NOTE: in the original this name was read before ever being assigned
        # (UnboundLocalError). None makes plot_results generate it internally.
        q_model = None

        # settings shared by all three runs / all three plots
        mcmc = dict(n_walkers=n_walkers, n_steps=n_steps, burn_in=burn_in,
                    n_cores=n_cores, contain=contain, force_rerun=force_rerun)
        style = dict(label=label, color=color,intrinsic_data=(a_s, ba_s, ca_s))

        # --- 1. main 4-parameter run on the image q distribution ---
        samples, max_prob_params, chain, q_obs, reff_values, sampler = self.run_or_load_inference(
            full_output_dir, output_prefix,
            generate_q=True, n_angles_per_halo=n_angles_per_halo, save_q_obs=True,
            true_params=true_params, q_model=q_model, q_true=q_true,
            **mcmc, **style,
        )
        recovery_results = assess_parameter_recovery(
            max_prob_params, true_params, q_obs=q_obs, q_model=self.q_model, q_true=q_true
        )
        with open(full_output_dir / f"{output_prefix}_recovery.txt", 'w') as f:
            f.write(recovery_results['output_text'])

        # --- 2. 7-parameter covariance run ---
        if self.run_covariances:
            cov_output_dir = full_output_dir.with_name(full_output_dir.name + "_covariance")
            true_params_cov = np.array(cov_fit(a_s, ba_s, ca_s))

            (samples_cov, max_prob_params_cov, chain_cov,
             q_obs, reff_values, sampler_cov) = self.run_or_load_inference(
                cov_output_dir, output_prefix,
                generate_q=True, n_angles_per_halo=n_angles_per_halo,
                multivariate=True, init_params=max_prob_params,
                plot_dir=cov_output_dir, plot_prefix=output_prefix + '_covariance',
                plot_burn_in=int(1 / 16 * burn_in),
                a_plot_key='a_obs', q_model_from_fit=True, q_model_n=10000,
                true_params=true_params_cov, q_true=q_true, a_true=a_true,
                **{**mcmc, 'n_steps': int(n_steps / 10), 'burn_in': int(burn_in / 16)},
                **style,
            )
            #recreate plot but use the luminosity based shapes as the true params!
            true_params_cov_i = np.array(cov_fit(a_s_i, ba_s_i, ca_s_i))
            style_i = dict(label=label, color=color, intrinsic_data=(a_s_i, ba_s_i, ca_s_i))
            (samples_cov, max_prob_params_cov, chain_cov,
             q_obs, reff_values, sampler_cov) = self.run_or_load_inference(
                cov_output_dir, output_prefix, q_input=q_obs,
                generate_q=False, n_angles_per_halo=n_angles_per_halo,
                multivariate=True, init_params=max_prob_params,
                plot_dir=cov_output_dir, plot_prefix=output_prefix+f'{self.band_key}3D_Shape_projections'+'_covariance',
                plot_burn_in=int(1 / 16 * burn_in),
                a_plot_key='a_obs', q_model_from_fit=True, q_model_n=10000,
                true_params=true_params_cov_i, q_true=q_i_true, a_true=a_i_true,
                **{**mcmc, 'n_steps': int(n_steps / 10), 'burn_in': int(burn_in / 16)},
                **style_i,
            )

        # --- 3. what the model predicts when directly deprojecting the intrinsic 3D shape ---
        # true_params here are what the MCMC returns when applied to the images, and
        # max_prob_params are what it returns when applied directly to projections of
        # the intrinsic shape, so the default legend text is wrong for this panel.
        q_true = np.array(q_true)
        prefix_3d = output_prefix + f'{self.band_key}3D_Shape_projections'
        labels_3d = {
            'obs': 'Synthetic Observed',
            'model': 'Model from 3D-projection fit',
            'true': 'Projections of intrinsic 3D shapes',
            'q_true': 'Projections of intrinsic 3D shapes',
            'intrinsic': 'Image Model',
            'inferred': '3D Shape Model',
        }

        # (samples_true_q, max_prob_params_true, chain_true,
        #  q_true, a_true, sampler_true) = self.run_or_load_inference(
        #     full_output_dir, prefix_3d,
        #     q_input=q_true, a_input=a_true,
        #     q_plot_key='q_true', q_model_from_fit=True,
        #     true_params=max_prob_params, q_obs=q_obs, labels=labels_3d,
        #     **mcmc, **style,
        # )

        return true_params,true_params_cov, max_prob_params, max_prob_params_cov,

    def plot_results(self, samples, max_prob_params, q_obs, q_model=None, q_true=None,
                     true_params=None, chain=None, burn_in=500, label="Model",
                     color="blue", output_prefix=None, output_dir="results",
                     a_obs=None, a_true=None, intrinsic_data=None, bin_width=0.04,
                     labels=None):
        """
        Plot inference results with model comparison.

        Works for 4- or 7-parameter fits; the extra panels are produced only when
        `max_prob_params` has seven entries.

        Parameters:
            samples (array): MCMC samples, (n_samples, ndim)
            max_prob_params (array): Highest-probability parameters, length 4 or 7
            q_obs (array): Observed projected axis ratios
            q_model, q_true (array): Precomputed projections. If None they are
                generated from the parameter vectors.
            true_params (array): True parameters, same length as max_prob_params
            chain (array): MCMC chain, (n_steps, n_walkers, ndim)
            burn_in (int): Number of burn-in steps
            label (str): Label for plots
            color (str): Color for plots
            output_prefix (str): Prefix for output files
            output_dir (str): Directory to save results
            a_obs (array): Observed projected sizes. 7-parameter runs only.
            intrinsic_data (tuple): (a, b, c) intrinsic arrays to compare against
                the model shape distribution. 7-parameter runs only.
            bin_width (float): Histogram bin width for the projection plots
            labels (dict): Overrides for the legend/axis text. Recognised keys:
                'obs', 'model', 'true', 'q_true', 'intrinsic'. Anything not given
                falls back to the defaults below.
        """
        # =====================================================================
        default_labels = {
            'obs': 'Synthetic Observed',
            'model': 'Model',
            'true': 'True',
            'intrinsic': 'Intrinsic Distribution',
            'inferred': 'Model Distribution',
        }

        labels = {**default_labels, **(labels or {})}

        if output_prefix is None:
            output_prefix = label.lower().replace(" ", "_")

        ndim = n_params(max_prob_params)
        if true_params is not None and n_params(true_params) != ndim:
            raise ValueError(
                f"true_params has {n_params(true_params)} entries but max_prob_params "
                f"has {ndim}. (A 7-parameter fit needs sigma_ac included.)"
            )

        print(f"[{label}] {ndim}-parameter model")
        print("  True params:    ", true_params)
        print("  Max prob params:", max_prob_params)

        os.makedirs(output_dir, exist_ok=True)
        path = lambda name: os.path.join(output_dir, f"{output_prefix}_{name}.png")

        # --- projections: observed vs model vs truth --------------------------
        fig = plot_projected_distributions_with_model(
            [q_obs], [max_prob_params], [true_params], [label], [color],
            q_model_list=None if q_model is None else [q_model],
            q_true_list=None if q_true is None else [q_true],
            output_file=path("projections_comparison"),
            # title=f"Projected Axis Ratios: Observed vs Model for {label}",
            bin_width=bin_width,
            obs_label=labels['obs'],
            model_label=labels['model'],
            true_label=labels['true'],
        )
        plt.close(fig)

        # Pass colour as a LIST. A bare string used to be zipped character-wise.
        fig = plot_ellipsoid_shapes(
            [samples], [max_prob_params], [true_params], [label], [color],
            output_file=path("ellipsoid_shapes"),
            # title=f"Intrinsic Shapes: {label}",
            kf_key=f'{self.radius}R_eff',
            focus_on_max_prob=True,
            intrinsic_data=intrinsic_data,
            intrinsic_label=labels['intrinsic'],
            inferred_label=labels['inferred'],
        )
        plt.close(fig)

        # fig = plot_projected_distributions(
        #     [q_obs], [label], [color],
        #     output_file=path("projections_hist"),
        #     #title=f"Projected Axis Ratios for {label}",
        #     bin_width=bin_width, kde=False,
        # )
        # plt.close(fig)

        # --- corner -----------------------------------------------------------
        fig = plot_corner(
            samples, max_prob_params, true_params,
            output_file=path("corner"),
            #title=f"Parameter Inference for {label}",
        )
        plt.close(fig)

        # --- chain ------------------------------------------------------------
        if chain is not None:
            fig = plot_chain_evolution(
                chain, burn_in=burn_in,
                output_file=path("chain"),
                #title=f"MCMC Chain Evolution for {label}",
            )
            plt.close(fig)

        # --- intrinsic shape plane -------------------------------------------


        # fig = plot_ellipsoid_shapes(
        #     [samples], [max_prob_params], [true_params], [label], ["blue"],
        #     output_file=path("ellipsoid_shapes_all"),
        #     #title=f"Intrinsic Shapes: {label}",
        #     focus_on_max_prob=False, show_samples=True, show_ellipses=False,
        # )
        # plt.close(fig)

        # --- 7-parameter extras ----------------------------------------------
        if ndim == 7:
            a_data = b_data = c_data = None
            if intrinsic_data is not None:
                a_data, b_data, c_data = intrinsic_data

            # Intrinsic A / B/A / C/A marginals, the shape plane, and the
            # A - C/A covariance panel where sigma_ac shows up.
            fig = plot_covariant_distribution(
                params=max_prob_params,
                a_data=a_data, b_data=b_data, c_data=c_data,
                label=label, color=color,
                output_file=path("shape_distribution"),
                #title=f"Intrinsic Shape Distribution: {label}",
            )
            plt.close(fig)

            # Observed q against a fresh model realisation, with KS.
            fig = plot_q_comparison(
                q_obs, max_prob_params,
                bin_width=bin_width, color=color, label=label,
                output_file=path("q_comparison"),
                #title=f"Projected Axis Ratio: {label}",
            )
            plt.close(fig)
            q_model_cov,a_model_cov = sample_model_projections(max_prob_params)
            fig = plot_projected_distributions_with_model(
                [a_obs], None, None, [label], [color],
                q_model_list=[a_model_cov],
                q_true_list=[a_true],
                output_file=path("projections_comparison_a_obs_cov"),
                # title=f"Projected Axis Ratios: Observed vs Model for {label}",
                bin_width=bin_width,
                type='a'
            )
            plt.close(fig)

            p = unpack_params(max_prob_params)
            rho = p['sigma_ac'] / (p['sigma_A'] * p['sigma_C'])
            print(f"  sigma_ac = {p['sigma_ac']:.4f}  ->  rho_ac = {rho:+.3f} "
                  f"(bound |sigma_ac| < {p['sigma_A'] * p['sigma_C']:.4f})")
            # if abs(rho) > 0.9:
            #     print("  WARNING: correlation is close to degenerate; the rejection "
            #           "sampler will have a low acceptance rate.")

            if a_obs is not None:
                print(f"  observed sizes: median a = {np.median(a_obs):.3f}")

    def run_inference_single_halo(self, sim_name, halo_id, n_angles,
                                  n_walkers=32, n_steps=3000, burn_in=500, n_cores=None,
                                  output_prefix=None, output_dir="results", force_rerun=False,
                                  color='blue'):
        """
        Run shape inference on a single halo.

        Parameters:
            sim_name (str): Simulation name
            halo_id (int/str): Halo identifier
            n_angles (int): Number of random viewing angles
            reff_index (int): Index into reff_multipliers to use
            n_walkers (int): Number of MCMC walkers
            n_steps (int): Number of MCMC steps
            burn_in (int): Number of burn-in steps
            n_cores (int): Number of CPU cores to use
            output_prefix (str): Prefix for output files
            output_dir (str): Directory to save results
            force_rerun (bool): Whether to force rerunning the analysis

        Returns:
            tuple: (samples, max_prob_params, sampler, q_obs) from inference
        """
        # Set up output directory and prefix
        if output_prefix is None:
            output_prefix = f"{sim_name}_{halo_id}"

        # Create full output directory path
        full_output_dir = Path(output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        # Check for existing results
        existing_results = None
        if not force_rerun:
            existing_results = check_existing_results(str(full_output_dir), output_prefix)
            if existing_results:
                samples, max_prob_params, chain, q_obs, a_obs, sampler = existing_results
                print(f"Using existing inference results for {sim_name} halo {halo_id}")

        if force_rerun or not existing_results:
            # Record start time
            start_time = time.time()

            # Generate q distribution
            q_obs,a_obs = self.generate_q_distribution_single_halo(
                sim_name, halo_id, n_angles,
            )

            # Save q distribution
            np.save(full_output_dir / f"{output_prefix}_q_obs.npy", q_obs)

            # Run inference
            samples, max_prob_params, sampler = infer_intrinsic_shape(
                q_obs,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                n_cores=n_cores,
                output_prefix=output_prefix,
                output_dir=str(full_output_dir)
            )

            # Record end time
            end_time = time.time()
            print(f"Inference completed in {end_time - start_time:.2f} seconds")

            # Extract the full chain for plotting
            chain = sampler.get_chain()

            # Save full results in the format expected by check_existing_results
            results = {
                'samples': samples,
                'max_prob_params': max_prob_params,
                'full_chain': chain,
                'a_obs': a_obs,
                'q_obs': q_obs,
                'sampler': sampler
            }

            with open(full_output_dir / f"{output_prefix}_results.pkl", 'wb') as f:
                pickle.dump(results, f)

            # Get the autocorrelation time
            try:
                tau = sampler.get_autocorr_time(quiet=True)
                print(f"Autocorrelation times: {tau}")
                print(f"Number of effective samples: {n_steps / np.max(tau)}")
            except Exception as e:
                print(f"Could not compute autocorrelation time (might need more samples): {e}")

        # get true_params from halo_data
        halo_data = self.halo_data[(sim_name, halo_id)]
        ba_s = halo_data['ba_s']
        ca_s = halo_data['ca_s']
        a_s = halo_data['a_s']
        true_params = np.array([ba_s, ca_s, 0, 0])
        model_samples = 10000
        #
        #q_true, _, _ = generate_model_projections([ba_s, ca_s, 0.001, 0.001], model_samples)
        phi, theta = random_viewing_angles(n_angles)
        alpha, beta = projected_semi_axes(phi, theta, ba_s, ca_s)
        q_true = beta / alpha
        a_true = a_s* np.sqrt(alpha * beta)
        q_model, _, _ = generate_model_projections(max_prob_params, model_samples)

        # Plot results
        self.plot_results(
            samples=samples,
            max_prob_params=max_prob_params,
            true_params=true_params,
            q_obs=q_obs,
            q_model=q_model,
            q_true=q_true,
            chain=chain,
            burn_in=burn_in,
            label=f"{sim_name} Halo {halo_id}",
            color=color,
            output_prefix=output_prefix,
            output_dir=str(full_output_dir)
        )

        return samples, max_prob_params, sampler, q_obs, a_s, ba_s, ca_s

    def get_all_halo_keys(self):
        """Return a list of all (sim_name, halo_id) keys in the collection."""
        return list(self.halo_data.keys())

    def get_halo_count(self):
        """Return the number of halos in the collection."""
        return len(self.halo_data)

    # ------------------------------------------------------------------
    def _intrinsic_truth(self, n_angles_per_halo=200, seed=None, log_offset=3):
        """
        Intrinsic shapes, the 4-param truth, and projections of the 3D shapes.
        Same logic as the start of run_inference_all_halos (which could call
        this instead), except a_true uses the same log offset as a_s.
        """
        keys = list(self.halo_data.keys())
        ba_s = np.array([self.halo_data[k]['ba_s'] for k in keys], dtype=float)
        ca_s = np.array([self.halo_data[k]['ca_s'] for k in keys], dtype=float)
        a_lin = np.array([self.halo_data[k]['a_s'] for k in keys], dtype=float)
        a_s = np.log10(a_lin) + log_offset
        true_params, _ = fit_ellipsoid_params(ba_s, ca_s)

        state = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)
        try:
            q_true, a_true = [], []
            for b, c, al in zip(ba_s, ca_s, a_lin):
                phi, theta = random_viewing_angles(n_angles_per_halo)
                alpha, beta = projected_semi_axes(phi, theta, b, c)
                q_true.append(beta / alpha)
                a_true.append(np.log10(al * np.sqrt(alpha * beta)) + log_offset)
        finally:
            if seed is not None:
                np.random.set_state(state)

        return dict(ba_s=ba_s, ca_s=ca_s, a_s=a_s, true_params=true_params,
                    q_true=np.concatenate(q_true), a_true=np.concatenate(a_true))

    # ------------------------------------------------------------------
    def _draw_single_angle_sample(self, angle_seed, halo_keys):
        """One random viewing angle per halo -> (q, log10(a)+3), each of length N_halos."""
        state = np.random.get_state()
        np.random.seed(angle_seed)

        phi, theta = random_viewing_angles(len(halo_keys))

        np.random.set_state(state)

        q = np.empty(len(halo_keys))
        reff = np.empty(len(halo_keys))
        for k, key in enumerate(halo_keys):
            th, ph = theta[k:k + 1], phi[k:k + 1]
            q[k] = 1 - np.atleast_1d(self.interpolators[key](th, ph))[self.reff_index]
            reff[k] = np.atleast_1d(self.reff_interpolators[key](th, ph))[self.reff_index]
        #
        # bad = ~np.isfinite(q) | ~np.isfinite(reff)
        # if np.any(bad):
        #     raise ValueError(f'non-finite q/reff for halos '
        #                      f'{[halo_keys[k] for k in np.where(bad)[0]]} (seed {angle_seed})')
        return q, np.log10(reff * self.radius) + 3

    # ------------------------------------------------------------------
    @staticmethod
    def _save_bootstrap(store, path):
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(store, f)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written file

    @staticmethod
    def _load_bootstrap(path, config):
        with open(path, 'rb') as f:
            store = pickle.load(f)
        old = store['config']
        diffs = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
        if diffs:
            raise ValueError(
                f'{path} was made with different settings ({diffs}). '
                f'Use force_rerun=True or a new output_prefix.')
        print(f'Loaded {len(store["results"])} existing bootstrap iterations from {path}')
        return store

    @staticmethod
    def _collect_bootstrap(store, n_bootstrap, fit_key):
        res = [store['results'][i] for i in sorted(store['results']) if i < n_bootstrap]
        ok = [r for r in res if fit_key in r]
        if not ok:
            return None
        fit = {k: np.array([r[fit_key][k] for r in ok]) for k in ok[0][fit_key]}
        fit['q_obs'] = np.array([r['q'] for r in ok])
        fit['a_obs'] = np.array([r['a'] for r in ok])
        fit['index'] = np.array([r['i'] for r in ok])
        fit['n_failed'] = len(res) - len(ok)
        return fit


    def plot_bootstrap_vs_reference(self,
                                    fit_key='fit4',
                                    show_truth=True,
                                    point_estimate='map',
                                    **kwargs):
        """
        Ellipse plot of a bootstrap run (self.bootstrap by default) against the
        many-angle run.

        fit_key : 'fit4' or 'fit7'
        reference_dir, reference_prefix : the output_dir / output_prefix you gave
            run_inference_all_halos. The 7-param reference is read from
            <reference_dir>_covariance, as that function writes it. If omitted,
            the reference stored in `boot` is used (params only, no samples).
        **kwargs : forwarded to plot_bootstrap_ellipses (n_ellipses, n_sigma,
            ref_color, boot_label, ref_label, annotate, show_contours, ...).
        """
        boot = self.bootstrap
        output_dir_boot = boot['output_dir']
        output_file = output_dir_boot + f'/bootstrap{fit_key}_ellipses.png'
        fit = boot[fit_key]
        if fit is None:
            raise ValueError(f'no {fit_key} results in this bootstrap run')

        ref_params = boot['reference_params'][fit_key]

        truth = boot['truth']
        true_params=boot['true_params'][fit_key]
        # if output_file is None:
        #     output_file = str(Path(boot['output_dir']) /
        #                       f'{boot["output_prefix"]}{self.band_key}_bootstrap_'
        #                       f'{fit_key[-1]}p_ellipses_vs_reference.png')

        return plot_bootstrap_ellipses(
            fit[point_estimate], reference_params=ref_params,
            true_params=true_params, show_truth=show_truth,
            intrinsic_data=(truth['a_s'], truth['ba_s'], truth['ca_s']),
            kf_key=f'{self.radius}R_eff', output_file=output_file,ref_label = '',
            **kwargs)
    # ------------------------------------------------------------------
    def run_bootstrap_inference(self,output_dir, n_bootstrap=200,
                                output_prefix='all_halos',
                                multivariate=True, mcmc_4=None, mcmc_7=None,
                                init_from='map', contain=True,
                                n_workers=None, seed=12345, n_keep=400,
                                save=True, save_every=10, force_rerun=False,
                                retry_failed=False,
                                reference_dir=None, reference_prefix=None,
                                n_angles_truth=200,
                                plot=True,
                                true_params=None,
                                true_params_cov=None,
                                max_prob_params=None,
                                max_prob_params_cov=None,
                                **plot_kwargs):
        """
        Bootstrap over viewing angles: n_bootstrap realisations, each with ONE
        random viewing angle per galaxy (the galaxy set itself is fixed).

        Parameters
        ----------
        n_bootstrap : int
            Number of realisations. Increasing it later reuses saved iterations
            and only runs the new ones.
        mcmc_4, mcmc_7 : dict
            n_walkers / n_steps / burn_in for the 4- and 7-parameter fits.
        init_from : 'map' or 'median'
            Which 4-param estimate warm-starts the 7-param fit.
        n_workers : int or None
            Parallel processes, one bootstrap iteration each. None/1 = serial.
        seed : int
            Iteration i uses SeedSequence([seed, i]), so results are reproducible
            and resuming gives the same answer as one uninterrupted run.
        n_keep : int
            Thinned posterior samples kept per iteration.
        save, save_every : bool, int
            Save to <output_dir>/<output_prefix>_bootstrap.pkl every save_every
            completed iterations, on Ctrl-C, and at the end.
        force_rerun : bool
            Ignore (and overwrite) the saved file.
        retry_failed : bool
            Rerun iterations that previously raised an error.
        reference_dir, reference_prefix :
            Location of your existing many-angle run (the output_dir /
            output_prefix given to run_inference_all_halos), overplotted
            as the "reference" result.
        n_angles_truth : int
            Angles per halo used to project the intrinsic 3D shapes (q_true).
        **plot_kwargs : forwarded to plot_bootstrap_results.

        Returns
        -------
        dict (also stored as self.bootstrap) with keys 'fit4', 'fit7'
        (per-iteration arrays), 'true_params', 'reference_params', 'truth', ...
        """
        multivariate=False
        mcmc_4 = {**dict(n_walkers=16, n_steps=1000, burn_in=750), **(mcmc_4 or {})}
        mcmc_7 = {**dict(n_walkers=4, n_steps=200, burn_in=100), **(mcmc_7 or {})}

        out_dir_boot = Path(output_dir) / 'bootstrap'
        out_dir_boot.mkdir(parents=True, exist_ok=True)
        path = out_dir_boot / f'{output_prefix}_bootstrap.pkl'

        print("[boot] output_dir        :", output_dir)
        print("[boot] output_prefix     :", output_prefix)
        print("[boot] reference_dir     :", reference_dir)
        print("[boot] reference_prefix  :", reference_prefix)
        print("[boot] band_key          :", repr(self.band_key))
        print("[boot] bootstrap pkl     :", path)
        for d in (Path(output_dir), Path(output_dir).with_name(Path(output_dir).name + '_covariance')):
            print(f"[boot] {d} exists={d.exists()}")
            if d.exists():
                print("        results pkls:", sorted(p.name for p in d.glob('*_results.pkl')))

        halo_keys = list(self.halo_data.keys())
        config = dict(halo_keys=halo_keys, radius=self.radius, reff_index=self.reff_index,
                      band_key=self.band_key, multivariate=multivariate,
                      mcmc_4=mcmc_4, mcmc_7=mcmc_7 if multivariate else None,
                      init_from=init_from, contain=contain, seed=seed, n_keep=n_keep)

        store = None
        if not force_rerun and path.exists():
            store = self._load_bootstrap(path, config)
        if store is None:
            store = {'config': config, 'results': {}}

        done = {i for i, r in store['results'].items()
                if r['error'] is None or not retry_failed}
        todo = [i for i in range(n_bootstrap) if i not in done]

        if todo:
            cfg = dict(contain=contain, mcmc_4=mcmc_4, mcmc_7=mcmc_7,
                       multivariate=multivariate, init_from=init_from, n_keep=n_keep)
            tasks = []
            for i in todo:
                angle_seed, mcmc_seed = np.random.SeedSequence([seed, i]).generate_state(2)
                q, a = self._draw_single_angle_sample(int(angle_seed), halo_keys)
                tasks.append((i, int(mcmc_seed), q, a, cfg))

            print(f'Running {len(tasks)} bootstrap iterations '
                  f'({len(halo_keys)} galaxies, {"4+7" if multivariate else "4"}-param, '
                  f'{n_workers or 1} worker(s))')
            t0 = time.time()
            try:
                if n_workers is not None and n_workers > 1:
                    with Pool(processes=n_workers) as pool:
                        self._consume_bootstrap(
                            pool.imap_unordered(_bootstrap_worker, tasks, chunksize=1),
                            store, path, save, save_every, len(tasks), t0)
                else:
                    self._consume_bootstrap(map(_bootstrap_worker, tasks),
                                            store, path, save, save_every, len(tasks), t0)
            finally:  # also runs on Ctrl-C
                if save:
                    self._save_bootstrap(store, path)

        failed = [r for i, r in store['results'].items() if i < n_bootstrap and r['error']]
        if failed:
            print(f'{len(failed)} iterations raised errors, e.g. #{failed[0]["i"]}: '
                  f'{failed[0]["error"]}')

        truth = self._intrinsic_truth(n_angles_truth, seed=seed)


        self.bootstrap = dict(
            fit4=self._collect_bootstrap(store, n_bootstrap, 'fit4'),
            fit7=self._collect_bootstrap(store, n_bootstrap, 'fit7') ,
            true_params={'fit7':true_params_cov,'fit4':true_params},
            reference_params={'fit7':max_prob_params_cov,'fit4':max_prob_params},
            truth=truth, config=config,
            output_dir=str(out_dir_boot), output_prefix=output_prefix,
            reference_dir=str(reference_dir), reference_prefix=reference_prefix,
        )

        #print(self.bootstrap)
        if plot:
            self.plot_bootstrap_results(out_dir_boot, output_dir, output_prefix)
        return self.bootstrap

    @classmethod
    def _consume_bootstrap(cls, iterator, store, path, save, save_every, total, t0):
        n_done = n_err = 0
        for res in iterator:
            store['results'][res['i']] = res
            n_done += 1
            n_err += res['error'] is not None
            if n_done % save_every == 0 or n_done == total:
                el = time.time() - t0
                print(f'  [{n_done}/{total}] {el / 60:.1f} min elapsed, '
                      f'~{el / n_done * (total - n_done) / 60:.1f} min left, {n_err} failed')
                if save:
                    cls._save_bootstrap(store, path)

    # ------------------------------------------------------------------
    def plot_bootstrap_results(self, output_dir_boot,output_dir,output_prefix, point_estimate='map',
                               label='Bootstrap', color='C0', bin_width=0.04,
                               n_band_curves=100, n_model=5000,
                               ):
        """
        Figures + summary table for each fit in `boot` (default self.bootstrap).
        point_estimate : 'median' (posterior median, more stable) or 'map'.
        """
        boot = self.bootstrap
        #out_dir = Path(boot['output_dir'])
        out_dir = output_dir_boot
        truth = boot['truth']
        intrinsic = (truth['a_s'], truth['ba_s'], truth['ca_s'])
        rng = np.random.default_rng(boot['config']['seed'])

        for fit_key in ('fit4', 'fit7'):
            fit = boot.get(fit_key)
            if fit is None:
                continue
            ndim = FIT_NDIM[fit_key]
            tag = f'{boot["output_prefix"]}{self.band_key}_bootstrap_{ndim}p'

            path = lambda name: str(out_dir / f'{tag}_{name}.png')
            tp = boot['true_params'][fit_key]
            rp = boot['reference_params'][fit_key]
            print(f"[plot] fit_key={fit_key} ndim={ndim} tag={tag}")
            print(f"[plot] rp is None: {rp is None}, tp is None: {tp is None}")
            print(f"[plot] ellipse path -> {path('ellipse')}")
            #print(fit)
            #print(point_estimate)
            pts = fit[point_estimate]
            ttl = ''#f'{label}: {len(pts)} single-angle realisations ({ndim}-param)'

            # # 1. corner of point estimates, and of the pooled posteriors
            # print(f'corner_plot_info{rp},{tp}')
            # fig = plot_corner(pts, max_prob_params=rp, true_params=tp,
            #                   output_file=path('corner_estimates'),
            #                   title=f'{ttl} — {point_estimate}s')
            # plt.close(fig)
            # fig = plot_corner(fit['thinned'].reshape(-1, ndim), max_prob_params=rp,
            #                   true_params=tp, output_file=path('corner_pooled_posterior'),
            #                   title=f'{ttl} — pooled posteriors')
            # plt.close(fig)
            #
            # # 2. marginals: scatter of estimates vs typical posterior width
            # fig = plot_bootstrap_marginals(fit, point_estimate, tp, rp, color,
            #                                output_file=path('marginals'), title=ttl)
            # plt.close(fig)
            #
            # # 3. shape plane
            # fig = plot_bootstrap_shape_plane(fit, point_estimate, tp, rp, intrinsic, color,
            #                                  output_file=path('shape_plane'), title=ttl)
            # plt.close(fig)'{tag}.png'
            fig = self.plot_bootstrap_vs_reference(fit_key=fit_key,)
            plt.close(fig)
            # fig = self.plot_bootstrap_vs_reference(boot, 'fit7', show_truth=False,
            #                                  reference_dir='Marvel.DC.results', reference_prefix='all_galaxies.radius2.contained.False')
            # plt.close(fig)

            # 4. calibration against truth and against the many-angle reference
            calib = {}
            for name, target in [('truth', tp), ('reference', rp)]:
                if target is None:
                    continue
                fig, calib[name] = plot_bootstrap_calibration(
                    fit, target, name, point_estimate,
                    output_file=path(f'calibration_{name}'), title=ttl)
                plt.close(fig)

            # 5. q (and a) distributions: bands over bootstrap models
            sel = rng.choice(len(pts), size=min(n_band_curves, len(pts)), replace=False)
            draws = [sample_model_projections(pts[k], n_model) for k in sel]
            q_bins = np.arange(0, 1 + bin_width, bin_width)
            fig = plot_bootstrap_hist_band(fit['q_obs'], [d[0] for d in draws], q_bins,
                                           reference=truth['q_true'], xlabel='q = b/a',
                                           color=color, output_file=path('q_band'), title=ttl)
            plt.close(fig)
            if ndim == 7:
                lo, hi = np.percentile(np.concatenate([fit['a_obs'].ravel(), truth['a_true']]),
                                       [0.5, 99.5])
                a_bins = np.arange(lo, hi + bin_width, bin_width)
                fig = plot_bootstrap_hist_band(fit['a_obs'], [d[1] for d in draws], a_bins,
                                               reference=truth['a_true'],
                                               xlabel=r'$\log_{10}$ a + 3', color=color,
                                               output_file=path('a_band'), title=ttl)
                plt.close(fig)

            # # 6. MCMC diagnostics
            # n_steps = boot['config'][f'mcmc_{ndim}']['n_steps']
            # fig = plot_bootstrap_diagnostics(fit, n_steps, output_file=path('diagnostics'),
            #                                  title=ttl)
            # plt.close(fig)

            # 7. summary table
            text = bootstrap_summary_text(fit, point_estimate, tp, rp, calib)
            text = f'{ttl}\n{fit["n_failed"]} failed iterations\n' + text
            with open(out_dir / f'{tag}_summary.txt', 'w') as f:
                f.write(text + '\n')
            print(text)


class SyntheticEllipseCollection(GalaxyEllipseCollection):
    """
    Drop-in replacement for GalaxyEllipseCollection where the "halos" are
    ellipsoids drawn from a known parameter vector instead of simulation data.

    Everything downstream (run_inference_all_halos, plot_results,
    assess_parameter_recovery, the covariance branch) is inherited unchanged;
    only the q/a generation and the setup are replaced.
    """

    def __init__(self, params, n_ellipsoids=100, run_covariances=False,
                 sim_name="mock", seed=None):
        print(params)
        # Parent state we still rely on. reff_index is unused here (no
        # interpolators) but kept so the inherited signature stays valid.
        super().__init__(reff_index=0, run_covariances=run_covariances)

        self.params = np.asarray(params, dtype=float)
        self.n_params = self.params.size
        if self.n_params not in (4, 7):
            raise ValueError(
                f"params must have 4 or 7 entries, got {self.n_params}"
            )
        self.n_ellipsoids = n_ellipsoids
        self.sim_name = sim_name

        if seed is not None:
            np.random.seed(seed)

        # One draw of the intrinsic population; each ellipsoid is a fake halo.
        B, C, A, _ = generate_ellipsoid_distribution(self.params, n_ellipsoids)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.A = np.asarray(A, dtype=float)  # all-nan for the 4-param model

        for i in range(n_ellipsoids):
            halo_key = (sim_name, i)
            halo_data = {
                'a_s': self.A[i],
                'ba_s': self.B[i],
                'ca_s': self.C[i],
            }
            self.halos[halo_key] = halo_data
            self.halo_data[halo_key] = halo_data

    # # -- disabled parent machinery -------------------------------------------
    #
    # def add_halo(self, *args, **kwargs):
    #     raise NotImplementedError(
    #         "MockEllipseCollection builds its halos from a parameter vector; "
    #         "use the constructor instead."
    #     )
    #
    # def copy_halo_from(self, *args, **kwargs):
    #     raise NotImplementedError(
    #         "MockEllipseCollection has no interpolators to copy."
    #     )
    #
    # # -- the one method that actually changes --------------------------------

    def generate_q_distribution_all_halos(self, angles_per_halo):
        """
        Project every fake halo from `angles_per_halo` random viewing angles.

        Returns (q_values, a_values) with length n_ellipsoids * angles_per_halo,
        matching the parent's contract. a_values is all-nan for the 4-parameter
        model, which is expected and unused downstream.
        """
        if not self.halos:
            raise ValueError("No halos in collection")

        q_values = []
        a_values = []
        i = 0
        j = 0

        for halo_key in self.halos.keys():
            halo_data = self.halos[halo_key]
            try:
                b = np.full(angles_per_halo, halo_data['ba_s'])
                c = np.full(angles_per_halo, halo_data['ca_s'])
                a = np.full(angles_per_halo, halo_data['a_s'])

                phi, theta = random_viewing_angles(angles_per_halo)
                alpha, beta = projected_semi_axes(phi, theta, b, c)

                q_values.extend(beta / alpha)
                a_values.extend(a * np.sqrt(alpha * beta))
                i += 1
            except Exception as ex:
                print(f"Error generating q values for halo {halo_key}: {ex}")
                j += 1

        print(f'Generated q values from {i} halos, failed for {j} halos.')

        q_values = np.array(q_values)
        a_values = np.array(a_values)

        if np.any(np.isnan(q_values)):
            raise ValueError("q values contain nans")
        # a_values is intentionally all-nan in the 4-parameter case, so it is
        # not checked here.

        return q_values, a_values

    # -- convenience for reading off recovery --------------------------------

    def sample_true_params(self):
        """
        The `true_params` vector run_inference_all_halos constructs internally,
        recomputed here so it can be compared against the input vector without
        touching that method.
        """
        ba_s = np.array([h['ba_s'] for h in self.halos.values()])
        ca_s = np.array([h['ca_s'] for h in self.halos.values()])
        return np.array([np.mean(ba_s), np.mean(ca_s),
                         np.std(ba_s), np.std(ca_s)])

    def report_input_vs_sample(self):
        """Print the input vector against what the finite draw actually realised."""
        sample = self.sample_true_params()
        if self.n_params == 4:
            names = ['mu_ba', 'mu_ca', 'sigma_ba', 'sigma_ca']
            inp = self.params
        else:
            names = ['mu_ba', 'mu_ca', 'sigma_ba', 'sigma_ca']
            inp = self.params[[1, 2, 4, 5]]

        print(f"{'param':>10} {'input':>10} {'sampled':>10} {'diff':>10}")
        for name, v_in, v_s in zip(names, inp, sample):
            print(f"{name:>10} {v_in:10.4f} {v_s:10.4f} {v_s - v_in:10.4f}")
        return sample
