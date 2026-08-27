import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
import pynbody
import warnings
import traceback
import logging

# from galaxy_ellipse_collection import GalaxyEllipseCollection

#
# #load configs
# import sys
# from config import db_connection, sys_path, results_output_directory, pickle_file
# os.environ['TANGOS_DB_CONNECTION'] = db_connection
# os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
# os.environ['TANGOS_SIMULATION_FOLDER'] = '~/data/CDM_z0'
# sys.path.append(sys_path)
# import tangos
#
# from tangos.properties import LivePropertyCalculation, LivePropertyCalculationInheritingMetaProperties
# from tangos.properties.pynbody import PynbodyPropertyCalculation
# from tangos.properties.pynbody.centring import centred_calculation


# setup the function I need


# load test halo
# halo = tangos.get_halo("r431.romulus25.3072g1HsbBH/r431.romulus25.3072g1HsbBH.004096/r431.romulus25.3072g1HsbBH.004096/halo_1")

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pynbody
import pymp

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pynbody
import pymp
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsophoteAnalysis")


def extract_floats(s):
    a = (float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s))[1]
    theta = a[1]
    phi = a[3]
    return theta, phi


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


def create_spherical_grid(N):
    theta_list, phi_list = np.ones(N + 2), np.ones(N + 2)
    N_count = 0
    a = 4 * np.pi / N
    d = np.sqrt(a)
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta

    d_phi = a / d_theta
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            theta_list[N_count], phi_list[N_count] = theta, phi
            N_count += 1

    # manually add points at face-on and exactly opposite
    theta_list[N_count], phi_list[N_count] = 0, 0
    N_count += 1
    theta_list[N_count], phi_list[N_count] = np.pi, 0
    # remove any nans
    nanmask = np.isnan(theta_list)
    theta_list, phi_list = theta_list[~nanmask], phi_list[~nanmask]

    x_angles, y_angles = spherical_to_rotation_angles(theta_list, phi_list)

    return theta_list, phi_list, x_angles, y_angles


class ImageHalo:
    """Base class for generating luminosity/density images at different orientations
    and calculating effective radii for each projection to study galaxy morphology.

    Standalone pynbody + pickle version. Each orientation is checkpointed as its own
    pickle file inside a per-halo folder, so runs can resume where they left off."""

    imaging_qty = None
    imaging_units = None
    particle_type_attr = None
    sb_profile_key = None
    lum_den_key = None
    magnitude_key = None

    def __init__(self, cache_dir='image_cache', dx=10, dy=10, n_procs=1, resolution=1000):
        self.cache_dir = cache_dir
        self.dx = dx
        self.dy = dy
        self.n_procs = n_procs
        self.resolution = resolution  # <-- image resolution (px per side); was hard-coded to 1000
        self.image_times = {}  # <-- per-orientation image-creation time, keyed by str(key)
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Cache helpers  (per-halo folder, per-orientation file)
    # ------------------------------------------------------------------ #
    def halo_dir(self, halo_id):
        """Folder holding all orientation pickles for one halo."""
        path = os.path.join(self.cache_dir, f'{self.__class__.__name__}_halo{halo_id}')
        os.makedirs(path, exist_ok=True)
        return path

    def orientation_path(self, halo_id, key):
        """Pickle path for a single orientation (e.g. key='x000y090')."""
        return os.path.join(self.halo_dir(halo_id), f'{key}.pkl')

    def orientation_completed(self, halo_id, key):
        """An orientation is 'done' if its pickle file exists."""
        return os.path.exists(self.orientation_path(halo_id, key))

    def save_orientation(self, halo_id, key, data):
        """Atomically write one orientation's dict to its own pickle."""
        path = self.orientation_path(halo_id, key)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic — no half-written files count as "done"
        return path

    def load_orientation(self, halo_id, key):
        with open(self.orientation_path(halo_id, key), 'rb') as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------ #
    # Sersic fitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def fit_sersic_profile(prof, sb_key):
        """Fits a Sérsic profile to determine the effective radius (Reff)."""

        def sersic(r, mueff, reff, n):
            return mueff + 2.5 * (0.868 * n - 0.142) * (
                    (r / reff) ** (1. / n) - 1)

        if sb_key is None:
            raise NotImplementedError("Sérsic fitting for mass density not yet implemented")

        profile_data = prof[sb_key]
        smooth = np.nanmean(
            np.pad(profile_data.astype(float), (0, 3 - profile_data.size % 3),
                   mode='constant', constant_values=np.nan).reshape(-1, 3),
            axis=1)

        x = np.arange(len(smooth)) * 0.3 + 0.15
        x[0] = .05

        y = smooth[~np.isnan(smooth)]
        x = x[~np.isnan(smooth)]

        r0 = x[int(len(x) / 2)]
        m0 = np.mean(y[:3])

        par, _ = curve_fit(sersic, x, y, p0=(m0, r0, 1),
                           bounds=([10, 0, 0.5], [40, 100, 16.5]))
        return par[1]

    # ------------------------------------------------------------------ #
    # Imaging
    # ------------------------------------------------------------------ #
    def generate_image(self, particles, width):
        """Generate image with smoothing appropriate for isophote fitting."""
        f = plt.figure(frameon=False)
        f.set_size_inches(10, 10)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)

        try:
            eps = particles['eps']
        except KeyError:
            eps = particles.properties['eps']
        eps = np.median(eps)
        smooth_floor = 3 * eps

        im = pynbody.plot.sph.image(
            particles,
            qty=self.imaging_qty,
            width=width,
            subplot=ax,
            units=self.imaging_units,
            resolution=self.resolution,  # <-- was hard-coded 1000; now configurable
            smooth_floor=smooth_floor,
            denoise=False,
            show_cbar=False,
            ret_im=True
        )
        data = im.get_array()
        plt.close(f)
        return data

    def process_orientation(self, particles, width, Rhalf):
        """Process a single orientation and extract relevant profile data."""
        orientation_data = {'Rhalf': Rhalf.view(np.ndarray)}
        prof = pynbody.analysis.profile.Profile(particles, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                                nbins=int((5 * Rhalf) / 0.1))

        orientation_data.update({
            'rbins': prof['rbins'].copy().view(np.ndarray),
            'binarea': prof._binsize.in_units('pc^2').copy().view(np.ndarray)
        })

        if self.sb_profile_key:
            orientation_data[f'sb_{self.imaging_qty.split("_")[0]}'] = prof[self.sb_profile_key].copy().view(np.ndarray)

        if self.lum_den_key:
            orientation_data[f'{self.imaging_qty}'] = prof[self.lum_den_key].copy().view(np.ndarray)

        if self.magnitude_key:
            orientation_data[f'mags_{self.imaging_qty.split("_")[0]}'] = prof[self.magnitude_key].copy().view(
                np.ndarray)
            orientation_data['lum_den'] = (
                    10.0 ** (-0.4 * prof[self.magnitude_key]) / prof._binsize.in_units('pc^2')).copy().view(np.ndarray)

        if self.imaging_qty == 'rho':
            orientation_data['rho'] = prof['density'].copy().view(np.ndarray)

        if self.sb_profile_key:
            orientation_data['Reff'] = self.fit_sersic_profile(prof, self.sb_profile_key)
        else:
            orientation_data['Reff'] = np.nan

        orientation_data['image'] = self.generate_image(particles, width)
        return orientation_data

    # ------------------------------------------------------------------ #
    # Main driver
    # ------------------------------------------------------------------ #
    def process_halo(self, halo, halo_id, existing_properties=None, overwrite=False,
                     orientation_indices=None, orientations=None):
        """Generate images and measure Reff across viewing angles, checkpointing
        each orientation to its own pickle file. Skips orientations already on disk.

        The one-time setup below (faceon, half_light_r, particle load) runs ONCE
        per call, then every requested orientation is imaged in the loop — so a
        single call over many orientations amortises setup exactly like the full
        production run does.

        orientation_indices : optional iterable of integer indices into the
            spherical grid. If None (default) all grid orientations are processed
            exactly as before. Pass e.g. [39] to process a single hardcoded view.
        orientations : optional list of explicit (theta, phi) pairs in radians.
            If given, the spherical grid is bypassed entirely and exactly these
            viewing angles are imaged, in the order provided. Takes precedence
            over orientation_indices.
        overwrite : if True, recompute even if a pickle already exists (so a
            timing test measures real work on every run)."""
        if existing_properties is None:
            existing_properties = {}

        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)

        particles = getattr(halo, self.particle_type_attr)

        if self.imaging_qty == 'rho':
            Rhalf = existing_properties.get('Rhalf_v', None)
            if Rhalf is None:
                Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        else:
            Rhalf = pynbody.analysis.luminosity.half_light_r(halo)

        width = 9 * Rhalf
        ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)

        # xrotations = np.arange(0, 180, self.dx)
        # yrotations = np.arange(0, 360, self.dy)

        if orientations is not None:
            # Explicit (theta, phi) pairs — bypass the spherical grid.
            theta_l = np.asarray([tp[0] for tp in orientations], dtype=float)
            phi_l = np.asarray([tp[1] for tp in orientations], dtype=float)
            xrotations, yrotations = spherical_to_rotation_angles(theta_l, phi_l)
            iter_indices = range(len(theta_l))
        else:
            theta_l, phi_l, xrotations, yrotations = create_spherical_grid(100)
            # Default: every orientation on the grid (original behaviour).
            if orientation_indices is None:
                orientation_indices = range(len(xrotations))
            iter_indices = orientation_indices

        # with pymp.Parallel(self.n_procs) as p:
        for i in iter_indices:
            xrotation, yrotation = xrotations[i], yrotations[i]
            # key = f'x{xrotation:03d}y{yrotation:03d}'
            key = (theta_l[i], phi_l[i])

            # Checkpoint skip: if this orientation's pickle exists, don't redo it
            # (unless overwrite=True, e.g. for a timing test).
            if not overwrite and self.orientation_completed(halo_id, key):
                continue

            # Time this single orientation's work: rotation + render + profile +
            # sersic + pickle write. Setup above is NOT included here, so summing
            # these and subtracting from the run wall-time isolates the one-time
            # setup cost.
            _t0 = time.time()
            with halo.rotate_x(xrotation).rotate_y(yrotation):
                sb_dict = self.process_orientation(particles[ImageSpace], width, Rhalf)

            # Write immediately so progress survives interruption.
            self.save_orientation(halo_id, key, sb_dict)
            self.image_times[f'{key}'] = time.time() - _t0
            # print status
            print(
                f'{i} out of {len(xrotations)} orientations completed for halo {halo_id} (x={xrotation}, y={yrotation})')

        return self.halo_dir(halo_id)

    def run(self, halo, halo_id, existing_properties=None, overwrite=False,
            orientation_indices=None, orientations=None):
        """Entry point. Computes any missing orientations for this halo and returns
        the folder path. Use load_results(halo_id) to reassemble the full dict."""
        folder = self.process_halo(halo, halo_id, existing_properties, overwrite=overwrite,
                                   orientation_indices=orientation_indices,
                                   orientations=orientations)
        print(f'[{self.__class__.__name__}] halo {halo_id} orientations complete — {folder}')
        return folder

    # ------------------------------------------------------------------ #
    # Reassembly  (load per-orientation pickles back into aggregated lists)
    # ------------------------------------------------------------------ #
    def load_results(self, halo_id):
        """Load all orientation pickles for a halo and aggregate them the way the
        old tangos version returned data (sorted by orientation key)."""
        folder = self.halo_dir(halo_id)
        keys = sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.pkl'))

        per_orientation = {k: self.load_orientation(halo_id, k) for k in keys}

        images, reff_values = [], []
        profile_rbins, profile_binarea = [], []
        profile_sb, profile_lum_den, profile_mags = [], [], []
        profile_lum_den_calc, profile_rho = [], []

        qty0 = self.imaging_qty.split("_")[0]

        for k in keys:
            d = per_orientation[k]
            images.append(d['image'])
            reff_values.append(d['Reff'])
            profile_rbins.append(d['rbins'])
            profile_binarea.append(d['binarea'])
            if f'sb_{qty0}' in d:
                profile_sb.append(d[f'sb_{qty0}'])
            if self.imaging_qty in d:
                profile_lum_den.append(d[self.imaging_qty])
            if f'mags_{qty0}' in d:
                profile_mags.append(d[f'mags_{qty0}'])
            if 'lum_den' in d:
                profile_lum_den_calc.append(d['lum_den'])
            if 'rho' in d:
                profile_rho.append(d['rho'])

        # Rhalf is identical across orientations; grab it from the first.
        Rhalf = per_orientation[keys[0]]['Rhalf'] if keys else None

        return {
            'images': images,
            'reff_values': reff_values,
            'orientations': keys,
            'Rhalf': Rhalf,
            'profile_rbins': profile_rbins,
            'profile_binarea': profile_binarea,
            'profile_sb': profile_sb,
            'profile_lum_den': profile_lum_den,
            'profile_mags': profile_mags,
            'profile_lum_den_calc': profile_lum_den_calc,
            'profile_rho': profile_rho
        }


class VBandStarImages(ImageHalo):
    """V-band images for stellar particles."""

    imaging_qty = 'V_lum_den'
    imaging_units = 'kpc^-2'
    particle_type_attr = 's'
    sb_profile_key = 'sb,V'
    lum_den_key = 'V_lum_den'
    magnitude_key = 'magnitudes,V'


class IsophoteAnalysis:
    """Analyzes isophotes to measure projected galaxy shapes at different radii.

    For each projection, measures ellipticity and position angle at 2-4 Reff
    to track how galaxy shape varies with radius. Initial ellipse geometry is
    seeded from a 2-D elliptical Sérsic fit (via SersicFitter.fit_2d_log).

    Standalone pynbody + pickle version. Reads the per-orientation image data
    produced by VBandStarImages, and checkpoints isophote results after every
    orientation into a single pickle file so runs can resume."""

    def __init__(self, image_type='v_stars', cache_dir='isophote_cache'):
        self.image_type = image_type
        self.cache_dir = cache_dir
        self.visualization_enabled = False
        self.fit_times = {}  # <-- per-orientation isophote-fit time, keyed by orientation
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def cache_path(self, halo_id):
        return os.path.join(self.cache_dir,
                            f'{self.__class__.__name__}_halo{halo_id}.pkl')

    def load_cache(self, halo_id):
        """Load the checkpoint dict {orientation_key: params}, or empty if none."""
        path = self.cache_path(halo_id)
        if not os.path.exists(path):
            return {}
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def save_cache(self, halo_id, params):
        """Atomically rewrite the checkpoint dict."""
        path = self.cache_path(halo_id)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(params, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------------ #
    # Initial-parameter estimation via SersicFitter
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sersic_initial_params(image, reff_kpc, kpc_per_pixel):
        """Estimate ellipse geometry from a 2-D Sérsic fit.

        Returns (center_x, center_y, eps, pa, radius_factor) with center in
        pixels, eps in [0.01, 0.85], pa in radians, radius_factor in [2.5, 4.5].
        """
        result = SersicFitter.fit_2d_log(image, kpc_per_pixel)

        nan_keys = [k for k, v in result.items() if np.isnan(v)]
        if nan_keys:
            raise RuntimeError(
                f"Sérsic fit failed for image (reff={reff_kpc:.3f} kpc): "
                f"NaN values in {nan_keys}."
            )

        center_x = result['x0'] / kpc_per_pixel
        center_y = result['y0'] / kpc_per_pixel
        eps = float(np.clip(1.0 - result['q'], 0.01, 0.85))
        pa = float(result['theta'])
        radius_factor = float(np.clip(result['reff'] / reff_kpc, 2.5, 4.5))

        return center_x, center_y, eps, pa, radius_factor

    @staticmethod
    def fit_single_image(image_data, radius, step_size_factors, center,
                         eps, pa, sma_factor, kpc_per_pixel,
                         plot=True, apply_smoothing=True, smoothing_kpc=0.1):
        """Fit elliptical isophotes to a single image.

        Returns (result, all_targets_met) where result is a list of
        [sma_px, eps, pa, grad_err, x0, y0, intens, rms] for each target
        radius (2, 3, 4 × Reff).
        """
        from scipy.interpolate import interp1d
        from photutils.isophote import EllipseGeometry, Ellipse

        radius_pixels = radius / kpc_per_pixel
        target_multipliers = [2.0, 3.0, 4.0]
        target_radii_kpc = {m: m * radius for m in target_multipliers}

        geometry = EllipseGeometry(
            x0=center[0], y0=center[1],
            sma=sma_factor * radius_pixels,
            eps=eps, pa=pa
        )

        try:
            ellipse = Ellipse(image_data, geometry)
            isolist = ellipse.fit_image(
                minsma=1.0 * radius_pixels,
                maxsma=5.0 * radius_pixels,
                sma0=geometry.sma,
                linear=False,
                step=0.1,
                maxit=50,
                minit=10,
                fix_center=False,
                fix_eps=False,
                fix_pa=False,
                sclip=3.0,
                nclip=2
            )
            logger.info(f"Fitted {len(isolist.sma)} isophotes.")
        except Exception as e:
            logger.error(f"Ellipse fitting failed: {e}")
            return [], False

        if len(isolist.sma) == 0:
            logger.warning("No isophotes found.")
            return [], False

        smas_kpc = isolist.sma * kpc_per_pixel
        good_mask = isolist.grad_r_error < 0.2
        if not np.any(good_mask):
            good_mask = np.ones(len(isolist.sma), dtype=bool)

        sort_idx = np.argsort(smas_kpc[good_mask])
        good_smas = smas_kpc[good_mask][sort_idx]
        good_eps = isolist.eps[good_mask][sort_idx]
        good_pa = isolist.pa[good_mask][sort_idx]
        good_x0 = isolist.x0[good_mask][sort_idx]
        good_y0 = isolist.y0[good_mask][sort_idx]
        good_intens = isolist.intens[good_mask][sort_idx]
        good_gerr = isolist.grad_r_error[good_mask][sort_idx]

        result = []
        targets_met = {m: False for m in target_multipliers}

        if len(good_smas) >= 2:
            try:
                f_eps = interp1d(good_smas, good_eps, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_pa = interp1d(good_smas, good_pa, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
                f_x0 = interp1d(good_smas, good_x0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
                f_y0 = interp1d(good_smas, good_y0, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

                for m in target_multipliers:
                    t_kpc = target_radii_kpc[m]
                    if good_smas[0] <= t_kpc <= good_smas[-1]:
                        ci = np.argmin(np.abs(good_smas - t_kpc))
                        if abs(good_smas[ci] - t_kpc) < 0.05 * t_kpc:
                            result.append([
                                good_smas[ci] / kpc_per_pixel,
                                good_eps[ci], good_pa[ci],
                                good_gerr[ci],
                                good_x0[ci], good_y0[ci],
                                good_intens[ci], 0.0
                            ])
                        else:
                            result.append([
                                t_kpc / kpc_per_pixel,
                                float(f_eps(t_kpc)), float(f_pa(t_kpc)),
                                0.1,
                                float(f_x0(t_kpc)), float(f_y0(t_kpc)),
                                0.0, 0.0
                            ])
                        targets_met[m] = True
                    else:
                        logger.warning(
                            f"Target {m}×Reff ({t_kpc:.2f} kpc) outside fitted "
                            f"range [{good_smas[0]:.2f}, {good_smas[-1]:.2f}]."
                        )
            except Exception as e:
                logger.error(f"Interpolation failed: {e}")

        # Last-resort fallback: nearest available isophote
        if not result and len(good_smas) > 0:
            logger.warning("Interpolation skipped; returning nearest isophotes.")
            for m in target_multipliers:
                ci = np.argmin(np.abs(good_smas - target_radii_kpc[m]))
                result.append([
                    good_smas[ci] / kpc_per_pixel,
                    good_eps[ci], good_pa[ci],
                    good_gerr[ci],
                    good_x0[ci], good_y0[ci],
                    good_intens[ci], 0.0
                ])

        return result, all(targets_met.values())

    # ------------------------------------------------------------------ #
    # Main orchestration
    # ------------------------------------------------------------------ #
    def get_isophote(self, halo_id, images, reff_values, orientations, Rhalf,
                     overwrite=False):
        """Fit isophotes to every projection, checkpointing after each one.

        Args:
            halo_id:      Identifier for cache naming.
            images:       List of 2-D image arrays, one per orientation.
            reff_values:  List of effective radii (kpc), one per orientation.
            orientations: List of orientation keys (e.g. 'x000y090').
            Rhalf:        Half-light radius (kpc).
            overwrite:    If True, ignore existing checkpoint and recompute all.

        Returns:
            List of isophote parameter lists, one per orientation, sorted by key.
        """
        print(f'\tGenerating isophotes for {self.image_type}...\n')

        params = {} if overwrite else self.load_cache(halo_id)

        step_size_factors = [1.0, 0.5, 0.25, 0.125]  # kept for compatibility
        kpc_per_pixel = (9 * Rhalf) / np.size(images[0], axis=0)

        for k, (image_data, radius, orientation) in enumerate(
                zip(images, reff_values, orientations)):

            # Checkpoint skip: already computed this orientation.
            if orientation in params:
                continue

            # Time this orientation's fit: sersic init + isophote fit + cache
            # write. The image read (load_results) happens once before this loop,
            # so it lands in the run wall-time but NOT in these per-orientation
            # numbers — subtract the sum of fit_times from the run time to see it.
            _t0 = time.time()

            # --- Sérsic-based initial geometry ---
            center_x, center_y, initial_eps, initial_pa, radius_factor = \
                self._sersic_initial_params(image_data, radius, kpc_per_pixel)

            # --- Isophote fitting ---
            try:
                param_i, all_targets_met = self.fit_single_image(
                    image_data, radius, step_size_factors,
                    (center_x, center_y), initial_eps, initial_pa,
                    3, kpc_per_pixel,
                    plot=self.visualization_enabled and k == 39
                )
            except Exception as e:
                logger.error(f'Error in orientation {orientation}: {e}')
                traceback.print_exc()
                raise

            params[orientation] = param_i

            # Save after each orientation so progress survives interruption.
            self.save_cache(halo_id, params)
            self.fit_times[orientation] = time.time() - _t0

            pct = round((k + 1) / len(images) * 100, 2)
            print(f'\tGenerating isophotes for {self.image_type}: {pct}%')

        # Return results sorted by orientation key
        return [params[key] for key in sorted(params)]

    def run(self, halo_id, imager, overwrite=False):
        """Convenience entry point that pulls image data from a VBandStarImages
        (or compatible) instance via load_results(), then fits isophotes.

        Args:
            halo_id:  Identifier, must match the one used during imaging.
            imager:   An ImageHalo-subclass instance with load_results(halo_id).
            overwrite: Recompute from scratch if True.
        """
        data = imager.load_results(halo_id)
        return self.get_isophote(
            halo_id,
            images=data['images'],
            reff_values=data['reff_values'],
            orientations=data['orientations'],
            Rhalf=data['Rhalf'],
            overwrite=overwrite
        )


class SersicFitter:
    """2-D elliptical Sérsic fitter, extracted so it can be used by any
    LivePropertyCalculation without inheriting from the heavy ImageHalo class."""

    @staticmethod
    def _sersic_1d(r, mueff, reff, n):
        bn = 0.868 * n - 0.142
        return mueff + 2.5 * bn * ((r / reff) ** (1.0 / n) - 1.0)

    @staticmethod
    def _elliptical_radius(x, y, x0, y0, theta, q):
        dx, dy = x - x0, y - y0
        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
        return np.sqrt(x_rot ** 2 + (y_rot / q) ** 2)

    @classmethod
    def _sersic_2d(cls, xy, mueff, reff, n, x0, y0, theta, q):
        x, y = xy
        r = cls._elliptical_radius(x, y, x0, y0, theta, q)
        bn = 0.868 * n - 0.142
        return mueff + 2.5 * bn * ((r / reff) ** (1.0 / n) - 1.0)

    @classmethod
    def fit_1d(cls, sb_profile, pixel_scale_kpc_per_bin):
        """Fit a 1-D circular Sérsic to an azimuthally-averaged profile array.

        Returns (mueff, reff, n) or (nan, nan, nan) on failure.
        """
        profile_data = np.asarray(sb_profile, dtype=float)
        # 3-point smoothing (same as original code)
        pad = (3 - profile_data.size % 3) % 3
        smooth = np.nanmean(
            np.pad(profile_data, (0, pad), mode='constant',
                   constant_values=np.nan).reshape(-1, 3),
            axis=1)

        x = np.arange(len(smooth)) * 0.3 + 0.15
        x[0] = 0.05
        valid = ~np.isnan(smooth)
        y, x = smooth[valid], x[valid]
        if len(x) < 4:
            return np.nan, np.nan, np.nan

        r0 = x[len(x) // 2]
        m0 = np.mean(y[:3])
        try:
            par, _ = curve_fit(
                cls._sersic_1d, x, y, p0=(m0, r0, 1),
                bounds=([10, 0, 0.5], [40, 100, 16.5]))
            return tuple(par)  # (mueff, reff, n)
        except (RuntimeError, ValueError):
            return np.nan, np.nan, np.nan

    @staticmethod
    def _sersic_2d_mag(xy, mueff, reff, n, x0, y0, theta, q):
        """Sérsic in magnitude/log space — fits directly to log10(image)."""
        x, y = xy
        dx = x - x0
        dy = y - y0
        x_rot = dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
        r = np.sqrt(x_rot ** 2 + (y_rot / q) ** 2)
        bn = 0.868 * n - 0.142
        # Surface brightness form, but in log10(intensity) units:
        # log10(I) = log10(Ieff) - (bn/ln10) * ((r/reff)^(1/n) - 1)
        return mueff - (bn / np.log(10)) * ((r / reff) ** (1.0 / n) - 1.0)

    @classmethod
    def fit_2d_log(cls, image, pixel_scale, p0_1d=None, rmax=None, rmin=None):
        """
        Fit 2-D elliptical Sérsic in log space.

        p0_1d : (mueff, reff, n) from the 1D profile_sb_v fit — already in
                log/mag units so plugs in directly
        """
        nan_result = {k: np.nan for k in
                      ('mueff', 'reff', 'n', 'x0', 'y0', 'theta', 'q')}

        ny, nx = (np.size(image, axis=0), np.size(image, axis=1))
        yy, xx = (np.mgrid[0:ny, 0:nx] + 0.5) * pixel_scale

        x0_g = nx * pixel_scale / 2.0
        y0_g = ny * pixel_scale / 2.0

        # Radial mask
        if rmax is not None or rmin is not None:
            r_from_centre = np.sqrt((xx - x0_g) ** 2 + (yy - y0_g) ** 2)
            radial_mask = np.ones((ny, nx), dtype=bool)
            if rmax is not None:
                radial_mask &= r_from_centre <= rmax
            if rmin is not None:
                radial_mask &= r_from_centre >= rmin  # <-- exclude inner pixels
        else:
            radial_mask = np.ones((ny, nx), dtype=bool)

        # Convert to log10 — this is the key change
        log_image = np.log10(image)
        mask = np.isfinite(log_image) & radial_mask
        if mask.sum() < 10:
            return nan_result

        xy_data = np.vstack([xx[mask], yy[mask]])
        z_flat = log_image[mask]

        # p0_1d from profile_sb_v fit is already (mueff, reff, n) in mag units,
        # but we want log10(I) units: log10(I) = -mueff/2.5 + const
        # Simplest: just use reff and n as guesses, estimate mueff from image centre
        if p0_1d is not None and not any(np.isnan(p0_1d)):
            _, reff_g, n_g = p0_1d
            mueff_g = float(np.nanmedian(z_flat))  # log10(Ieff) guess from data
        else:
            mueff_g = float(np.nanmedian(z_flat))
            reff_g = min(nx, ny) * pixel_scale / 4.0
            n_g = 1.0

        centre_tol = pixel_scale * 1.0
        p0 = [mueff_g, reff_g, n_g, x0_g, y0_g, 0.0, 0.9]
        bounds_lo = [mueff_g - 5, 0.01, 0.5, x0_g - centre_tol, y0_g - centre_tol, -np.pi, 0.1]
        bounds_hi = [mueff_g + 5, 100, 16.5, x0_g + centre_tol, y0_g + centre_tol, np.pi, 1.0]

        try:
            popt, _ = curve_fit(
                cls._sersic_2d_mag, xy_data, z_flat,
                p0=p0, bounds=(bounds_lo, bounds_hi),
                maxfev=100000)
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"2-D Sérsic fit failed: {e}")
            return nan_result

        return dict(zip(('mueff', 'reff', 'n', 'x0', 'y0', 'theta', 'q'), popt))


class VBandIsophoteAnalysis(IsophoteAnalysis):
    """V-band isophote analysis (default)."""

    def __init__(self, cache_dir='isophote_cache'):
        super().__init__(image_type='v_stars', cache_dir=cache_dir)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator


def plot_isophotes(images, isophote_params, orientations, reffs, rhalf, filename, plot=True):
    from matplotlib.patches import Ellipse
    # 72 images, 72 isophote_params, 72 orientations
    extent = 9 * rhalf  # in kpc
    kpc_per_pixel = extent / images[0].shape[0]
    images_3d = np.array(images)
    # fig, axs = plt.subplots(18, 36, figsize=(90, 45))
    print(len(images))
    if plot:
        fig, axs = plt.subplots(6, 10, figsize=(40, 20))
        fig.patch.set_facecolor('black')  # Set the figure background color
        axs = axs.flatten()
    ellipse_dict = {}
    center = (images[0].shape[0] // 2, images[0].shape[1] // 2)
    for i in range(len(images)):

        # print(orientations[i])

        vmin = np.min(np.log10(images[i]))
        if vmin < -1:
            vmin = -1
        if plot:
            axs[i].imshow(np.log10(images[i]), cmap='magma', origin='lower', vmin=vmin)
        reff = reffs[i]
        # convert reff to pixels
        reff = reff / kpc_per_pixel

        # plot isophotes
        iso_params = isophote_params[i]

        smas, epss, pas, grad_errs, x0s, y0s, intenss, rmss = [], [], [], [], [], [], [], []

        for j in range(len(iso_params)):
            sma, eps, pa, grad_err, x0, y0, intens, rms = iso_params[j]
            # print(f'sma: {sma}, eps: {eps}, pa: {pa}, grad_err: {grad_err}, x0: {x0}, y0: {y0}')
            # print(sma,eps,pa,grad_err,x0,y0)
            if grad_err < 0.15:
                smas.append(sma)
                epss.append(eps)
                pas.append(pa)
                grad_errs.append(grad_err)
                x0s.append(x0)  # remove center addition later
                y0s.append(y0)
                intenss.append(intens)
                rmss.append(rms)

        ellipses = np.ones(3) * np.nan
        for k in [2, 3, 4]:
            # boolean filter for grad_err <0.1

            # find index of sma closest to j*reff
            try:
                idx = (np.abs(np.array(smas) - k * reff)).argmin()
            except:
                # print(f"Available smas: {smas}")
                continue

            sma = smas[idx]

            # if sma is far, print
            # if np.abs(sma - k*reff) > 0.3*reff:
            #     print(f'smas: {sma:.2f}, reff: {k*reff:.2f}')
            eps = epss[idx]
            pa = pas[idx]
            grad_err = grad_errs[idx]
            x0 = x0s[idx]
            y0 = y0s[idx]
            intens = intenss[idx]
            rms = rmss[idx]

            # print(idx,sma,eps,pa,grad_err,x0,y0)
            center_offset = np.sqrt((images[i].shape[0] // 2 - x0) ** 2 + (images[i].shape[1] // 2 - y0) ** 2)
            # plot ellipse

            # get ellipse parameters
            # color by gradient error
            vmin = 0
            vmax = 0.15
            # create colormap
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.winter
            # if center_offset < 100:
            # print(sma,eps,pa,grad_err)
            # distance in pixels from 0.1kpc
            d = 0.5 / kpc_per_pixel * reff
            # set linestyle based on intensity
            if rms == 0:
                rms = 1e-3
            if intens / rms > 0.5 and intens > 1:
                linestyle = '-'
            else:
                linestyle = '--'

            if (center_offset < d) and (grad_err < 0.3):
                if plot:
                    ellipse = Ellipse((x0, y0), 2 * sma, 2 * sma * (1 - eps), angle=np.degrees(pa),
                                      edgecolor=cmap(norm(grad_err)), facecolor='none',
                                      linestyle=linestyle, linewidth=1.5)
                    axs[i].add_patch(ellipse)
                ellipses[k - 2] = eps
            else:
                ellipses[k - 2] = np.nan

        # save ellipses to dict
        ellipse_dict[orientations[i]] = ellipses
        if plot:
            axs[i].axis('off')
            axs[i].set_aspect('equal')
            theta, phi = extract_floats(orientations[i])
            axs[i].set_title(f'{theta},{phi}', color='white', y=0.85)
    # reduce white space
    if plot:
        plt.subplots_adjust(wspace=0, hspace=0)
        # make sure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=150)
        plt.close(fig)
    return ellipse_dict


if __name__ == '__main__':
    # ====================================================================== #
    # RESOLUTION TIMING / ACCURACY TEST  (3 orientations, first discarded)
    #   - one halo, four resolutions: 0.5, 0.75, 1.0, 1.5 x the base resolution
    #   - THREE orientations imaged in ONE imager.run() call per resolution, so
    #     the one-time setup (faceon, half_light_r, particle load) is paid once,
    #     exactly like the 100-orientation production run:
    #         orientation 0 : warm-up throwaway  (absorbs first-touch / lazy I/O)
    #         orientation 1 : near face-on   (theta = 0,    phi = 0)
    #         orientation 2 : side-on/edge-on (theta = pi/2, phi = 0)
    #   - per-orientation image time + isophote-fit time are captured separately
    #     (imager.image_times / isophote.fit_times); the warm-up is dropped.
    #   - one-time setup is recovered as (run wall-time - sum of per-orientation
    #     times) for both the imaging and the fitting stages, so you can see how
    #     much is fixed overhead vs marginal-per-orientation work.
    #   - writes timings + shapes to a text file.
    # Each resolution gets its own cache_dir so all pickles are still written.
    # ====================================================================== #

    sim = pynbody.load(
        '/home/bk639/data/CDM_z0/r431.romulus25.3072g1HsbBH/r431.romulus25.3072g1HsbBH.004096/r431.romulus25.3072g1HsbBH.004096')
    sim.physical_units()
    halos = sim.halos(halo_numbers='v1')

    halo = halos[1]
    halo_id = 1

    # ----------------------------- config -------------------------------- #
    BASE_RESOLUTION = 1000  # your current resolution (1.0x)
    RESOLUTION_FACTORS = [0.5, 0.75, 1.0, 1.5]
    OUTPUT_FILE = 'resolution_test_results.txt'
    N_PROD = 100  # orientations in the real production run (for the estimate)
    # OVERWRITE=True forces real recomputation every run so the timings are
    # meaningful even if the per-resolution caches already exist.
    OVERWRITE = True

    # Viewing angles (theta, phi) in radians. The FIRST is the warm-up throwaway
    # (can be anything); the other two are the ones we actually report.
    WARMUP = (np.pi / 4.0, 0.0)  # anything inclined
    FACE_ON = (0.0, 0.0)  # near face-on
    SIDE_ON = (np.pi / 2.0, 0.0)  # side-on / edge-on (phi is free)
    ORIENTATIONS = [WARMUP, FACE_ON, SIDE_ON]


    # --------------------------------------------------------------------- #

    def key_str(theta, phi):
        """Rebuild the exact orientation-key string that process_halo writes and
        load_results reads back (keeps us independent of the buggy extract_floats
        and of any numpy repr quirks)."""
        return f'{(np.float64(theta), np.float64(phi))}'


    ORIENT_NAME = {
        key_str(*WARMUP): 'warmup',
        key_str(*FACE_ON): 'face_on',
        key_str(*SIDE_ON): 'side_on',
    }

    records = []  # one dict per resolution

    for factor in RESOLUTION_FACTORS:
        resolution = int(round(BASE_RESOLUTION * factor))
        tag = f'res_{factor:g}'
        print(f'\n=== Resolution factor {factor} ({resolution} px) ===')

        # ---- image creation: all 3 orientations in ONE run (setup paid once) ----
        imager = VBandStarImages(cache_dir=f'vband_cache_{tag}',
                                 resolution=resolution, n_procs=1)
        t0 = time.time()
        imager.run(halo, halo_id, orientations=ORIENTATIONS, overwrite=OVERWRITE)
        t_run_image = time.time() - t0

        # ---- isophote fitting: all 3 orientations ----
        isophote = VBandIsophoteAnalysis(cache_dir=f'isophote_cache_{tag}')
        t0 = time.time()
        isophote_params = isophote.run(halo_id, imager, overwrite=OVERWRITE)
        t_run_iso = time.time() - t0

        # ---- gather per-orientation timings + shapes ----
        image_dict = imager.load_results(halo_id)
        orientations = image_dict['orientations']

        ellipse_dict = plot_isophotes(
            image_dict['images'], isophote_params,
            orientations, image_dict['reff_values'], image_dict['Rhalf'],
            filename='unused_when_plot_false.png', plot=False)

        per = {}
        for orient in orientations:
            name = ORIENT_NAME.get(orient, 'warmup')
            per[name] = {
                'orient': orient,
                't_image': float(imager.image_times.get(orient, np.nan)),
                't_iso': float(isophote.fit_times.get(orient, np.nan)),
                'eps': np.asarray(ellipse_dict[orient], dtype=float),
            }

        # One-time setup = run wall-time minus the sum of per-orientation work.
        # (image setup ~ faceon + half_light_r + particle load;
        #  iso setup   ~ load_results reading the image pickles back off disk.)
        img_setup = t_run_image - float(np.nansum(list(imager.image_times.values())))
        iso_setup = t_run_iso - float(np.nansum(list(isophote.fit_times.values())))

        records.append({
            'factor': factor, 'resolution': resolution,
            't_run_image': t_run_image, 't_run_iso': t_run_iso,
            'img_setup': img_setup, 'iso_setup': iso_setup,
            'per': per,
        })

        for name in ('face_on', 'side_on'):
            p = per.get(name, {})
            print(f'  {name:8s}: image {p.get("t_image", np.nan):.2f}s | '
                  f'isophote {p.get("t_iso", np.nan):.2f}s')
        print(f'  one-time setup — image {img_setup:.2f}s | isophote(load) {iso_setup:.2f}s')

    # ------------------------ write results file ------------------------- #
    ref_idx = RESOLUTION_FACTORS.index(1.0) if 1.0 in RESOLUTION_FACTORS else 0

    with open(OUTPUT_FILE, 'w') as fh:
        fh.write('Resolution timing / accuracy test (3 orientations, warm-up discarded)\n')
        fh.write('orientation 0 = warm-up (dropped); reported: face-on (theta=0), '
                 'side-on (theta=pi/2)\n')
        fh.write(f'halo_id={halo_id}  base_resolution={BASE_RESOLUTION}px  '
                 f'overwrite={OVERWRITE}\n\n')

        fh.write('--- One-time setup per resolution (paid ONCE for all orientations) ---\n')
        fh.write(f'{"factor":>8} {"res(px)":>8} {"img_setup":>10} {"iso_setup":>10}\n')
        for r in records:
            fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} '
                     f'{r["img_setup"]:>10.3f} {r["iso_setup"]:>10.3f}\n')

        fh.write('\n--- Per-orientation IMAGE creation time (s) ---\n')
        fh.write(f'{"factor":>8} {"res(px)":>8} {"face_on":>10} {"side_on":>10}\n')
        for r in records:
            fo = r['per'].get('face_on', {}).get('t_image', np.nan)
            so = r['per'].get('side_on', {}).get('t_image', np.nan)
            fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} {fo:>10.3f} {so:>10.3f}\n')

        fh.write('\n--- Per-orientation ISOPHOTE fit time (s) ---\n')
        fh.write(f'{"factor":>8} {"res(px)":>8} {"face_on":>10} {"side_on":>10}\n')
        for r in records:
            fo = r['per'].get('face_on', {}).get('t_iso', np.nan)
            so = r['per'].get('side_on', {}).get('t_iso', np.nan)
            fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} {fo:>10.3f} {so:>10.3f}\n')

        fh.write('\n--- Per-orientation MARGINAL total, image+fit (s) ---\n')
        fh.write(f'{"factor":>8} {"res(px)":>8} {"face_on":>10} {"side_on":>10}\n')
        for r in records:
            fo = r['per'].get('face_on', {})
            so = r['per'].get('side_on', {})
            fo_t = fo.get('t_image', np.nan) + fo.get('t_iso', np.nan)
            so_t = so.get('t_image', np.nan) + so.get('t_iso', np.nan)
            fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} {fo_t:>10.3f} {so_t:>10.3f}\n')

        for name, label in (('face_on', 'face-on'), ('side_on', 'side-on')):
            fh.write(f'\n--- {label} ellipticity (eps at 2, 3, 4 x Reff) ---\n')
            fh.write(f'{"factor":>8} {"res(px)":>8} {"eps@2Reff":>10} {"eps@3Reff":>10} {"eps@4Reff":>10}\n')
            for r in records:
                e = r['per'].get(name, {}).get('eps', np.array([np.nan] * 3))
                fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} '
                         f'{e[0]:>10.4f} {e[1]:>10.4f} {e[2]:>10.4f}\n')

            ref_e = records[ref_idx]['per'].get(name, {}).get('eps', np.array([np.nan] * 3))
            fh.write(f'--- {label} eps difference vs 1x reference ---\n')
            fh.write(f'{"factor":>8} {"d_eps@2":>10} {"d_eps@3":>10} {"d_eps@4":>10}\n')
            for r in records:
                e = r['per'].get(name, {}).get('eps', np.array([np.nan] * 3))
                d = np.asarray(e, dtype=float) - np.asarray(ref_e, dtype=float)
                fh.write(f'{r["factor"]:>8g} {d[0]:>10.4f} {d[1]:>10.4f} {d[2]:>10.4f}\n')

        # Rough production estimate. NOTE: uses the mean of the face-on & side-on
        # marginal costs; it does NOT capture the growth of the one-time image-load
        # I/O (iso_setup), which in production reads ALL N orientation pickles at
        # once and so scales with N and with resolution. Treat as a lower bound on
        # the fixed part.
        fh.write(f'\n--- Rough estimate for {N_PROD} orientations '
                 f'(img_setup + {N_PROD} x mean marginal) ---\n')
        fh.write(f'{"factor":>8} {"res(px)":>8} {"est_total(s)":>14} {"est_total(min)":>16}\n')
        for r in records:
            fo = r['per'].get('face_on', {})
            so = r['per'].get('side_on', {})
            marg = np.nanmean([
                fo.get('t_image', np.nan) + fo.get('t_iso', np.nan),
                so.get('t_image', np.nan) + so.get('t_iso', np.nan),
            ])
            est = r['img_setup'] + N_PROD * marg
            fh.write(f'{r["factor"]:>8g} {r["resolution"]:>8d} '
                     f'{est:>14.1f} {est / 60.0:>16.2f}\n')

    print(f'\nWrote timing + shape comparison to {OUTPUT_FILE}')

# ====================================================================== #
# ORIGINAL PRODUCTION RUN (preserved, commented out)
# ====================================================================== #
# if __name__ == '__main__':
#     sim = pynbody.load('/home/bk639/data/CDM_z0/r431.romulus25.3072g1HsbBH/r431.romulus25.3072g1HsbBH.004096/r431.romulus25.3072g1HsbBH.004096')
#     sim.physical_units()
#     halos = sim.halos(halo_numbers='v1')
#
#     plot = False #no need to generate, fetch, of fit images, so do not load them
#
#     halo = halos[1]
#
#     imager = VBandStarImages(cache_dir='vband_cache_n_100', dx=10, dy=10, n_procs=1)
#     folder = imager.run(halo,1)
#     image_dict = imager.load_results(1)
#
#     isophote = VBandIsophoteAnalysis(cache_dir='isophote_cache_n_100')
#     isophote_params = isophote.run(1, imager)
#
#     images = image_dict['images']
#     orientations = image_dict['orientations']
#
#     ellipse_dict = plot_isophotes(image_dict['images'],
#                                   isophote_params,image_dict['orientations'],image_dict['reff_values'],
#                                   image_dict['Rhalf'],
#                                   filename='/home/bk639/MCMC-deprojection/angle_test_tangos/r431_angle_test.png',
#                                   plot=plot)
#     #print(ellipse_dict)
#
#     #save ellipse dict as pickle_file
#     dict = {'r431':{'/r431.romulus25.3072g1HsbBH/%/halo_1':ellipse_dict}}
#     with open('angle_test_ellipses_data_n_100', 'wb') as f:
#         pickle.dump(dict,f)