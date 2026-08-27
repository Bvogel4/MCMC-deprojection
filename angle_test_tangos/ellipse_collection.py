from pythonjsonlogger import exception
from scipy import stats
import pynbody
import warnings
import traceback
import logging


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable

from scipy.optimize import curve_fit
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator
import pynbody
import pymp
import pynbody
import pymp
import re
import os
import sys
import gc
import pickle
import traceback
import numpy as np
from config import (db_connection, sys_path, pickle_file, ba_s_key, ca_s_key,
                    results_output_directory)

os.environ['TANGOS_DB_CONNECTION'] = db_connection
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsophoteAnalysis")


_IMAGING_ARRAYS = ['pos', 'mass', 'eps', 'metals', 'tform', 'massform', 'age']

N_GRID = 100
N_PROCS = 30
MIN_NSTAR = 5000
MAX_HALOS = 50
PLOT = True
if PLOT:
    N_PROCS = 1
    print('Plotting requires a single thread, run without to process data first in parallel.')

CACHE_ROOT = '/home/bk639/MCMC-deprojection/caches'          # <-- adjust
FIGURE_DIR = os.path.join(results_output_directory, 'figures')
OUTPUT_PICKLE = 'ellipse_data_n_{}.pickle'.format(N_GRID)

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------




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
    N_count=0
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
    theta_list[N_count],phi_list[N_count] = 0,0
    N_count += 1
    theta_list[N_count],phi_list[N_count] = np.pi ,0
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
    # Arrays needed for V-band imaging + Sersic profiles. 'pos'/'mass'/'eps' are
    # required; the rest feed pynbody's luminosity derivations. Missing ones are
    # skipped silently -- if a profile key comes back empty, add the missing name.


    def __init__(self, cache_dir='image_cache',n_grid=100, n_procs=1):
        self.cache_dir = cache_dir
        self.n_grid = n_grid
        self.n_procs = n_procs
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

    @staticmethod
    def extract_particle_data(particles):
        """Copy the needed arrays out of (possibly shared) pynbody storage into
        plain numpy arrays. Runs in the parent, before the fork."""
        data = {}
        for k in _IMAGING_ARRAYS:
            try:
                arr = particles[k]
            except Exception:
                print(f'Could not load {k}')
                continue
            # np.array(..., copy=True) drops any SharedArray backing.
            data[k] = (np.array(arr, copy=True), str(arr.units))
        if 'pos' not in data:
            raise RuntimeError("could not extract 'pos' from particles")
        return data
    @staticmethod
    def build_private_snapshot(data, properties):
        """Build a standalone snapshot owned solely by the calling process.
        Must be called AFTER the fork so the allocation is process-private."""
        n = len(data['pos'][0])
        snap = pynbody.new(star=n)
        for k, (arr, units) in data.items():
            snap[k] = arr
            try:
                snap[k].units = units
            except Exception:
                pass
        snap.properties.update(properties)
        return snap

    @staticmethod
    def rotation_matrix(xdeg, ydeg):
        """Match pynbody's rotate_x then rotate_y convention."""
        cx, sx = np.cos(np.radians(xdeg)), np.sin(np.radians(xdeg))
        cy, sy = np.cos(np.radians(ydeg)), np.sin(np.radians(ydeg))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Ry @ Rx

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
            resolution=1000,
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
            orientation_data[f'mags_{self.imaging_qty.split("_")[0]}'] = prof[self.magnitude_key].copy().view(np.ndarray)
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



    def process_halo(self, halo, halo_id, existing_properties=None, overwrite=False):
        """Generate images across viewing angles, checkpointing each orientation."""
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

        # A sphere centred on the origin is rotation-invariant, so this cut can be
        # made once here rather than inside each rotation -- same particle set,
        # and it shrinks what every worker has to copy.
        ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)
        imaging_particles = particles[ImageSpace]

        # Extract in the parent; these plain arrays are inherited copy-on-write and
        # are only ever read, never mutated, so they stay shared and cheap.
        data = self.extract_particle_data(imaging_particles)
        properties = dict(imaging_particles.properties)
        pos0 = data['pos'][0]
        pos_units = data['pos'][1]
        n_part = len(pos0)

        theta_l, phi_l, xrotations, yrotations = create_spherical_grid(self.n_grid)
        keys = [f'{i:04d}_t{float(theta_l[i]):+.6f}_p{float(phi_l[i]):+.6f}'
                for i in range(len(xrotations))]

        print(f'halo {halo_id}: {n_part} particles, {len(keys)} orientations, '
              f'{self.n_procs} procs')

        with pymp.Parallel(self.n_procs) as p:
            # Built once per worker, after the fork -> private memory.
            snap = self.build_private_snapshot(data, properties)
            snap.physical_units()

            for i in p.xrange(len(xrotations)):
                key = keys[i]
                if self.orientation_completed(halo_id, key):
                    continue

                # Always from the pristine positions, so no drift accumulates and
                # no other worker can influence this one.
                R = self.rotation_matrix(xrotations[i], yrotations[i])
                snap['pos'] = pynbody.array.SimArray(pos0 @ R.T, pos_units)

                sb_dict = self.process_orientation(snap, width, Rhalf)
                self.save_orientation(halo_id, key, sb_dict)

                #print(f'{i + 1}/{len(xrotations)} done for halo {halo_id} '
                      #f'(x={xrotations[i]}, y={yrotations[i]})')

            del snap

        missing = [k for k in keys if not self.orientation_completed(halo_id, k)]
        if missing:
            raise RuntimeError(
                f'halo {halo_id}: {len(missing)}/{len(keys)} orientations failed '
                f'(check worker tracebacks above)')

        return self.halo_dir(halo_id)

    def run(self, halo, halo_id, existing_properties=None, overwrite=False):
        """Entry point. Computes any missing orientations for this halo and returns
        the folder path. Use load_results(halo_id) to reassemble the full dict."""
        folder = self.process_halo(halo, halo_id, existing_properties, overwrite=overwrite)
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



class iBandStarImages(ImageHalo):
    """i-band images for stellar particles."""

    imaging_qty = 'i_lum_den'
    imaging_units = 'kpc^-2'
    particle_type_attr = 's'
    sb_profile_key = 'sb,i'
    lum_den_key = 'i_lum_den'
    magnitude_key = 'magnitudes,i'


class IsophoteAnalysis:
    """Analyzes isophotes to measure projected galaxy shapes at different radii.

    For each projection, measures ellipticity and position angle at 2-4 Reff
    to track how galaxy shape varies with radius. Initial ellipse geometry is
    seeded from a 2-D elliptical Sérsic fit (via SersicFitter.fit_2d_log).

    Standalone pynbody + pickle version. Reads the per-orientation image data
    produced by VBandStarImages, and checkpoints isophote results after every
    orientation into a single pickle file so runs can resume."""

    def __init__(self, image_type='v_stars', cache_dir='isophote_cache',n_procs = 1):
        self.image_type = image_type
        self.cache_dir = cache_dir
        self.visualization_enabled = False
        os.makedirs(self.cache_dir, exist_ok=True)
        self.n_procs = n_procs

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

        center_x      = result['x0'] / kpc_per_pixel
        center_y      = result['y0'] / kpc_per_pixel
        eps           = float(np.clip(1.0 - result['q'], 0.01, 0.85))
        pa            = float(result['theta'])
        radius_factor = float(np.clip(result['reff'] / reff_kpc, 2.5, 4.5))

        return center_x, center_y, eps, pa, radius_factor

    @staticmethod
    def fit_single_image(image_data, radius, step_size_factors, center,
                         eps, pa, sma_factor, kpc_per_pixel,
                         plot=True, apply_smoothing=True, smoothing_kpc=0.1):
        """Fit elliptical isophotes to a single image.

        Returns (result, ok) where result is a list of
        [sma_px, eps, pa, grad_err, x0, y0, intens, rms] for *every*
        isophote returned by Ellipse.fit_image, sorted by increasing sma.
        Filtering (grad_r_error cuts, target-radius selection) is left to
        downstream code.
        """
        from photutils.isophote import EllipseGeometry, Ellipse

        radius_pixels = radius / kpc_per_pixel

        geometry = EllipseGeometry(
            x0=center[0], y0=center[1],
            sma=sma_factor * radius_pixels,
            eps=eps, pa=pa
        )

        try:
            ellipse = Ellipse(image_data,geometry=geometry)
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
                n_clip=2
            )
        except Exception as e:
            logger.info(f"Ellipse fitting failed: {e}")
            return [], False

        if len(isolist.sma) == 0:
            logger.info("No isophotes found.")
            return [], False

        def _col(name):
            # photutils can emit None for undefined errors -> object array
            arr = np.asarray(getattr(isolist, name), dtype=object)
            return np.array([np.nan if v is None else float(v) for v in arr])

        sma = np.asarray(isolist.sma, dtype=float)  # pixels
        epsv = _col('eps')
        pav = _col('pa')
        gerr = _col('grad_r_error')
        x0 = _col('x0')
        y0 = _col('y0')
        intens = _col('intens')
        rms = _col('rms')

        order = np.argsort(sma)
        result = [
            [sma[i], epsv[i], pav[i], gerr[i], x0[i], y0[i], intens[i], rms[i]]
            for i in order
        ]

        return result, True

    # ------------------------------------------------------------------ #
    # Main orchestration
    # ------------------------------------------------------------------ #
    def get_isophote(self, halo_id, images, reff_values, orientations,
                     Rhalf, params=None):
        """Measure isophotes for every orientation, checkpointing as we go."""
        kpc_per_pixel = (9 * Rhalf) / np.size(images[0], axis=0)
        step_size_factors = [1.0, 0.5, 0.25, 0.125]
        if params is None:
            params = self.load_cache(halo_id)
        shared_params = pymp.shared.dict()
        for k, v in params.items():
            shared_params[k] = v
        failed = pymp.shared.list()
        n = len(images)

        started = pymp.shared.array((1,), dtype='int64')
        completed = pymp.shared.array((1,), dtype='int64')

        sys.stdout.write('\n\n')  # reserve 2 lines for the bars
        sys.stdout.flush()

        with pymp.Parallel(self.n_procs) as p:
            for k in p.xrange(n):
                image_data = images[k]
                radius = reff_values[k]
                orientation = orientations[k]

                with p.lock:
                    started[0] += 1
                    print_progress(started[0], completed[0], n, len(failed))

                # Checkpoint skip: already computed this orientation.
                if orientation in shared_params:
                    with p.lock:
                        completed[0] += 1
                        print_progress(started[0], completed[0], n, len(failed))
                    continue

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
                    with p.lock:
                        failed.append(orientation)
                        completed[0] += 1  # done being processed, just unsuccessfully
                        print_progress(started[0], completed[0], n, len(failed))
                    continue

                with p.lock:
                    shared_params[orientation] = param_i
                    completed[0] += 1
                    print_progress(started[0], completed[0], n, len(failed))

        print()  # move past the bars
        self.save_cache(halo_id, dict(shared_params))

        if failed:
            raise RuntimeError(
                f'halo {halo_id}: {len(failed)}/{n} orientations failed '
                f'(check worker tracebacks above): {sorted(failed)}')
        return [shared_params[key] for key in sorted(shared_params.keys())]

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
        x_rot =  dx * np.cos(theta) + dy * np.sin(theta)
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
            return tuple(par)           # (mueff, reff, n)
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
    def fit_2d_log(cls, image, pixel_scale, p0_1d=None, rmax = None, rmin = None):
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

    def __init__(self, cache_dir='isophote_cache',n_procs=1):
        super().__init__(image_type='v_stars',n_procs=n_procs, cache_dir=cache_dir)

class iBandIsophoteAnalysis(IsophoteAnalysis):
    """V-band isophote analysis (default)."""

    def __init__(self, cache_dir='isophote_cache',n_procs=1):
        super().__init__(image_type='i_stars',n_procs=n_procs, cache_dir=cache_dir)


# def plot_isophotes(images, isophote_params, orientations, reffs, rhalf, filename,
#                    plot=True, grad_err_max=0.3, grad_err_cut=0.15,
#                    cmap_img='magma', cmap_iso='winter', dpi=150):
#     """Plot images with isophote ellipses at 2/3/4 Reff, labelled by orientation.
#
#     Returns {orientation: array([eps_2reff, eps_3reff, eps_4reff])}.
#     """
#     n = len(images)
#     extent_kpc = 9 * rhalf
#     kpc_per_pixel = extent_kpc / images[0].shape[0]
#
#     # --- figure layout: square-ish grid sized to the number of images ---
#     if plot:
#         ncols = int(np.ceil(np.sqrt(n)))
#         nrows = int(np.ceil(n / ncols))
#         panel = 3.2  # inches per panel
#         fig, axs = plt.subplots(nrows, ncols,
#                                 figsize=(panel * ncols, panel * nrows),
#                                 facecolor='black')
#         axs = np.atleast_1d(axs).ravel()
#         for ax in axs:
#             ax.set_facecolor('black')
#             ax.axis('off')
#
#         # shared colour normalisation for the ellipses
#         norm = plt.Normalize(vmin=0.0, vmax=grad_err_cut)
#         cmap = plt.get_cmap(cmap_iso)
#
#         # shared intensity scale so panels are comparable
#         finite = np.concatenate([np.log10(im[im > 0]).ravel() for im in images])
#         vmin_g, vmax_g = np.percentile(finite, [5, 99.5])
#         vmin_g = max(vmin_g, -1)
#
#         # physical scale bar length (kpc) -> pixels
#         bar_kpc = round(extent_kpc / 5, 1)
#         bar_pix = bar_kpc / kpc_per_pixel
#
#     ellipse_dict = {}
#
#     for i in range(n-1):#ignore last one, which is at theta, phi = 1,1 on accident
#         img = images[i]
#         reff = reffs[i] / kpc_per_pixel  # pixels
#
#         if plot:
#             ax = axs[i]
#             ax.imshow(np.log10(np.where(img > 0, img, np.nan)),
#                       cmap=cmap_img, origin='lower',
#                       vmin=vmin_g, vmax=vmax_g, interpolation='nearest')
#
#             # --- label: theta / phi in degrees ---
#             th, ph = extract_floats(orientations[i])
#             if th:
#                 label = rf'$\theta={np.degrees(th[0]):.0f}^\circ$   ' \
#                         rf'$\phi={np.degrees(ph[0]):.0f}^\circ$'
#             else:
#                 label = str(orientations[i])
#             ax.text(0.5, 0.965, label, transform=ax.transAxes,
#                     color='white', fontsize=11, ha='center', va='top',
#                     bbox=dict(facecolor='black', alpha=0.45,
#                               edgecolor='none', pad=2.5))
#
#         # --- filter isophote table ---
#         cols = np.array([p for p in isophote_params[i]], dtype=float)
#         if cols.size == 0:
#             ellipse_dict[orientations[i]] = np.full(3, np.nan)
#             continue
#         good = cols[:, 3] < grad_err_cut
#         cols = cols[good]
#
#         ellipses = np.full(3, np.nan)
#         if cols.shape[0]:
#             smas = cols[:, 0]
#             for k in (2, 3, 4):
#                 idx = np.abs(smas - k * reff).argmin()
#                 sma, eps, pa, grad_err, x0, y0, intens, rms = cols[idx]
#
#                 center_offset = np.hypot(img.shape[0] // 2 - x0,
#                                          img.shape[1] // 2 - y0)
#                 d = 0.5 / kpc_per_pixel * reff
#                 if rms == 0:
#                     rms = 1e-3
#                 linestyle = '-' if (intens / rms > 0.5 and intens > 1) else '--'
#
#                 if (center_offset < d) and (grad_err < grad_err_max):
#                     ellipses[k - 2] = eps
#                     if plot:
#                         axs[i].add_patch(Ellipse(
#                             (x0, y0), 2 * sma, 2 * sma * (1 - eps),
#                             angle=np.degrees(pa),
#                             edgecolor=cmap(norm(grad_err)), facecolor='none',
#                             linestyle=linestyle, linewidth=1.6, alpha=0.95))
#
#         ellipse_dict[orientations[i]] = ellipses
#
#         if plot:
#             axs[i].set_aspect('equal')
#             axs[i].set_xlim(0, img.shape[1])
#             axs[i].set_ylim(0, img.shape[0])
#
#     if plot:
#         # scale bar on the first panel only
#         y0b = 0.06 * images[0].shape[0]
#         x0b = 0.06 * images[0].shape[1]
#         axs[0].plot([x0b, x0b + bar_pix], [y0b, y0b], color='white', lw=2.5)
#         axs[0].text(x0b + bar_pix / 2, y0b * 1.35, f'{bar_kpc:g} kpc',
#                     color='white', fontsize=10, ha='center', va='bottom')
#
#         fig.subplots_adjust(wspace=0.01, hspace=0.01,
#                             left=0.01, right=0.93, top=0.99, bottom=0.01)
#
#         # colourbar for gradient error
#         sm = ScalarMappable(norm=norm, cmap=cmap)
#         cax = fig.add_axes([0.945, 0.25, 0.012, 0.5])
#         cb = fig.colorbar(sm, cax=cax)
#         cb.set_label('gradient error', color='white', fontsize=13)
#         cb.ax.tick_params(colors='white')
#         cb.outline.set_edgecolor('white')
#
#         d = os.path.dirname(filename)
#         if d:
#             os.makedirs(d, exist_ok=True)
#         fig.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=dpi,
#                     facecolor=fig.get_facecolor())
#         plt.close(fig)
#
#     return ellipse_dict

def _parse_tp(key):
    """Return (theta, phi) in radians from an orientation key, or None."""
    m = re.search(r'np\.float64\(([^)]+)\)\s*,\s*np\.float64\(([^)]+)\)', key)
    if not m:
        m = re.search(r't([+-]?[\d.]+)_p([+-]?[\d.]+)', key)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _isophote_eps(iso_params, reff_pix, img_shape, kpc_per_pixel,
                  grad_err_cut=0.15, grad_err_max=0.3):
    """Return (ellipses_for_return, drawlist) where drawlist holds the ellipse
    geometry to render: (x0, y0, sma, eps, pa, grad_err, linestyle)."""
    ellipses = np.full(3, np.nan)
    draw = []
    cols = np.asarray(iso_params, dtype=float)
    if cols.size == 0:
        return ellipses, draw
    cols = cols[cols[:, 3] < grad_err_cut]
    if cols.shape[0] == 0:
        return ellipses, draw

    smas = cols[:, 0]
    d = 0.5 / kpc_per_pixel * reff_pix
    for k in (2, 3, 4):
        idx = np.abs(smas - k * reff_pix).argmin()
        sma, eps, pa, grad_err, x0, y0, intens, rms = cols[idx]
        offset = np.hypot(img_shape[0] // 2 - x0, img_shape[1] // 2 - y0)
        if rms == 0:
            rms = 1e-3
        if (offset < d) and (grad_err < grad_err_max):
            ellipses[k - 2] = eps
            ls = '-' if (intens / rms > 0.5 and intens > 1) else '--'
            draw.append((x0, y0, sma, eps, pa, grad_err, ls))
    return ellipses, draw


def _parse_tp(key):
    """Return (theta, phi) in radians from an orientation key, or None."""
    m = re.search(r'np\.float64\(([^)]+)\)\s*,\s*np\.float64\(([^)]+)\)', key)
    if not m:
        m = re.search(r't([+-]?[\d.]+)_p([+-]?[\d.]+)', key)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _wrap180(phi_rad):
    """Wrap an angle in radians into (-pi, pi]."""
    return (phi_rad + np.pi) % (2 * np.pi) - np.pi


def _isophote_eps(iso_params, reff_pix, img_shape, kpc_per_pixel,
                  grad_err_cut=0.15, grad_err_max=0.3):
    """Return (eps at 2/3/4 Reff, list of ellipses to draw)."""
    ellipses = np.full(3, np.nan)
    draw = []
    cols = np.asarray(iso_params, dtype=float)
    if cols.size == 0:
        return ellipses, draw
    cols = cols[cols[:, 3] < grad_err_cut]
    if cols.shape[0] == 0:
        return ellipses, draw

    smas = cols[:, 0]
    d = 0.5 / kpc_per_pixel * reff_pix
    reffs = [2,2.5,3]
    for i in range(len(reffs)):
        target_reff = reffs[i]
        idx = np.abs(smas - target_reff * reff_pix).argmin()
        sma, eps, pa, grad_err, x0, y0, intens, rms = cols[idx]
        offset = np.hypot(img_shape[0] // 2 - x0, img_shape[1] // 2 - y0)
        if rms == 0:
            rms = 1e-3
        if (offset < d) and (grad_err < grad_err_max):
            ellipses[i] = eps
            ls = '-' if (intens / rms > 0.5 and intens > 1) else '--'
            draw.append((x0, y0, sma, eps, pa, grad_err, ls))
    return ellipses, draw


def plot_isophotes(images, isophote_params, orientations, reffs, rhalf,
                            filename, plot=True, panel=2.6,
                            grad_err_cut=0.15, grad_err_max=0.3,
                            cmap_img='magma', cmap_iso='winter',
                            theta_decimals=0, dpi=150,
                            row_gap=0.004, col_gap=0.004):
    """One row per theta (0 at top, 180 at bottom), phi increasing left to
    right with phi = 0 in the same central column for every row.
    Returns {orientation: array([eps_2reff, eps_3reff, eps_4reff])}.
    """
    kpc_per_pixel = (9 * rhalf) / images[0].shape[0]
    orientations = orientations[0:-1]

    # ---- group by theta; sort by phi wrapped into (-180, 180] ----
    rows = {}
    for i, key in enumerate(orientations):
        tp = _parse_tp(str(key))
        if tp is None:
            continue
        t_deg = round(np.degrees(tp[0]), theta_decimals)
        rows.setdefault(t_deg, []).append((_wrap180(tp[1]), i))
    for t in rows:
        rows[t].sort(key=lambda x: x[0])
    theta_vals = sorted(rows)
    nrows = len(theta_vals)

    # ---- column index of phi=0 in each row, and the global grid width ----
    zero_idx = {}
    for t, entries in rows.items():
        phis = np.array([e[0] for e in entries])
        zero_idx[t] = int(np.abs(phis).argmin())  # entry nearest phi = 0
    max_left = max(zero_idx[t] for t in theta_vals)
    max_right = max(len(rows[t]) - 1 - zero_idx[t] for t in theta_vals)
    ncols = max_left + max_right + 1
    center_col = max_left  # phi = 0 lives here

    ellipse_dict = {}

    if plot:
        fig = plt.figure(figsize=(panel * ncols, panel * nrows),
                         facecolor='black')
        norm = plt.Normalize(vmin=0.0, vmax=grad_err_cut)
        cmap = plt.get_cmap(cmap_iso)
        finite = np.concatenate([np.log10(im[im > 0]).ravel() for im in images])
        vmin_g, vmax_g = np.percentile(finite, [5, 99.5])
        vmin_g = max(vmin_g, -1)

        left, right = 0.045, 0.945
        w = (right - left) / ncols
        h = 1.0 / nrows

    for r, t_deg in enumerate(theta_vals):
        entries = rows[t_deg]
        i0 = zero_idx[t_deg]
        if plot:
            y_bot = 1.0 - (r + 1) * h
            fig.text(left * 0.6, y_bot + h / 2,
                     rf'$\theta={t_deg:.0f}^\circ$', color='white',
                     fontsize=13, ha='center', va='center', rotation=90)

        for j, (phi_w, i) in enumerate(entries):
            img = images[i]
            reff_pix = reffs[i] / kpc_per_pixel
            eps3, draw = _isophote_eps(isophote_params[i], reff_pix, img.shape,
                                       kpc_per_pixel, grad_err_cut, grad_err_max)
            ellipse_dict[orientations[i]] = eps3
            if not plot:
                continue

            c = center_col + (j - i0)  # aligns phi=0 across rows
            ax = fig.add_axes([left + c * w + col_gap / 2,
                               y_bot + row_gap / 2,
                               w - col_gap, h - row_gap])
            ax.set_facecolor('black')
            ax.axis('off')
            ax.imshow(np.log10(np.where(img > 0, img, np.nan)),
                      cmap=cmap_img, origin='lower',
                      vmin=vmin_g, vmax=vmax_g, interpolation='nearest')
            for (x0, y0, sma, eps, pa, ge, ls) in draw:
                ax.add_patch(Ellipse((x0, y0), 2 * sma, 2 * sma * (1 - eps),
                                     angle=np.degrees(pa),
                                     edgecolor=cmap(norm(ge)), facecolor='none',
                                     linestyle=ls, linewidth=1.4, alpha=0.95))
            ax.set_aspect('equal')
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(0, img.shape[0])
            ax.text(0.5, 0.97, rf'$\phi={np.degrees(phi_w):+.0f}^\circ$',
                    transform=ax.transAxes, color='white', fontsize=9,
                    ha='center', va='top',
                    bbox=dict(facecolor='black', alpha=0.4,
                              edgecolor='none', pad=1.5))

    if plot:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([0.958, 0.35, 0.008, 0.3])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('gradient error', color='white', fontsize=12)
        cb.ax.tick_params(colors='white')
        cb.outline.set_edgecolor('white')

        d = os.path.dirname(filename)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(filename, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)



    return ellipse_dict


def render_bar(current, total, width=30):
    total_safe = max(total, 1)
    frac = min(current / total_safe, 1.0)
    filled = int(width * frac)
    bar = '#' * filled + '-' * (width - filled)
    return f'[{bar}] {current:>4}/{total} ({frac*100:5.1f}%)'

def print_progress(started, completed, total, failed=0):
    line1 = f'Started  : {render_bar(started, total)}'
    line2 = f'Completed: {render_bar(completed, total)}' + (f'  ({failed} failed)' if failed else '')
    sys.stdout.write('\033[2A\r\033[K' + line1 + '\n\033[K' + line2 + '\n')
    sys.stdout.flush()
    
def expected_orientation_keys(n_grid):
    """The orientation keys process_halo() will write, without touching a snapshot."""
    theta_l, phi_l,_, _ = create_spherical_grid(n_grid)
    keys = []
    for i in range(n_grid):
        key = f'{i:04d}_t{float(theta_l[i]):+.6f}_p{float(phi_l[i]):+.6f}'
        keys.append(key)
    return keys

def missing_orientations(imager, hid, keys):
    return [k for k in keys if not imager.orientation_completed(hid, k)]


def resume_dict(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        print('Resuming from {} ({} sims already present)'.format(path, len(d)))
        return d
    return {}


def select_halos(sim, sim_name):
    """Halo selection, done entirely in tangos — no snapshot I/O."""
    if not sim.timesteps:
        return []
    timestep = sim.timesteps[-1]
    halos = timestep.halos[:MAX_HALOS]

    # Massive MERIANs: keep only the most star-rich halo
    if sim_name.startswith('r') and not sim_name.startswith('rogue'):
        max_halo, max_stars = None, 0
        for i, halo in enumerate(halos):
            n_stars = halo['n_star'][0]
            if n_stars > max_stars:
                max_stars, max_halo = n_stars, i
        if max_halo is None:
            return []
        halos = [halos[max_halo]]
        print('  keeping only halo {}'.format(halos[0].basename.split('_')[1]))
    elif sim_name.startswith('h'):
        print('  dropping halo 0')
        halos = halos[1:]

    return halos


# ---------------------------------------------------------------------------
def process_halo_ref(halo_ref, hid, imager, isophote, sim_name, orientation_keys):
    """Returns the ellipse dict for one halo. Loads particle data only if the
    image cache is incomplete."""
    missing = missing_orientations(imager, hid, orientation_keys)

    if missing:
        print('  {} / {} orientations missing — loading particle data'.format(
            len(missing), len(orientation_keys)))
        pyn_halo = tangos.get_halo(halo_ref).load()
        #print(pyn_halo.s.loadable_keys())
        try:
            imager.run(pyn_halo, hid)
        finally:
            del pyn_halo
            gc.collect()
    else:
        print('  image cache complete — no particle load needed')

    # Everything below reads only from the pickle caches. The images are the
    # only large objects in play, so they are loaded exactly once here and
    # dropped as soon as the fits and the figure are done with them.
    image_dict = imager.load_results(hid)
    try:
        # Call get_isophote directly rather than isophote.run(hid, imager):
        # run() would call load_results() again and hold a second copy of
        # every image for the duration of the fit.
        isophote_params = isophote.get_isophote(
            hid,
            images=image_dict['images'],
            reff_values=image_dict['reff_values'],
            orientations=image_dict['orientations'],
            Rhalf=image_dict['Rhalf'],
        )

        filename = os.path.join(
            FIGURE_DIR, '{}.{}.isophotes.png'.format(sim_name, hid))
        print(filename)

        halo_dict = plot_isophotes(
            image_dict['images'],
            isophote_params,
            image_dict['orientations'],
            np.array(image_dict['reff_values']),
            image_dict['Rhalf'],
            filename=filename,
            plot=PLOT,
        )
    except Exception as e:
        print(f'failed to plot isophotes for {sim_name},{hid}')
        #trace
        traceback.print_exc()
        sys.exit()
    finally:
        # Drop the image arrays before returning, even on failure, so a bad
        # halo can't leave ~1 GB pinned for the rest of the run.
        image_dict.pop('images', None)
        del image_dict
        gc.collect()

    return halo_dict


def main():
    print(f'N_procs = {N_PROCS}')
    ellipse_dict = resume_dict(OUTPUT_PICKLE)
    orientation_keys = expected_orientation_keys(N_GRID)

    for sim in tangos.all_simulations():

        sim_name = str(sim.basename)

        print('\nSimulation {}'.format(sim_name))


        ellipse_dict.setdefault(sim_name, {})

        # one cache dir + one imager/isophote pair per simulation
        imager = iBandStarImages(
            cache_dir=os.path.join(CACHE_ROOT, 'iband_cache_{}_n{}'.format(sim_name, N_GRID)),
            n_grid=N_GRID, n_procs=N_PROCS)
        isophote = iBandIsophoteAnalysis(
            cache_dir=os.path.join(CACHE_ROOT, 'i.isophote_cache_{}_n{}'.format(sim_name, N_GRID)),
            n_procs=N_PROCS)

        for halo in select_halos(sim, sim_name):
            halo_name = halo.basename
            halo_ref = '{}/%/{}'.format(sim_name, halo_name)
            hid = int(halo_name.split('_')[1])

            # if halo_ref in ellipse_dict[sim_name]:
            #     if PLOT == False:
            #         print('  skipping {} - already processed'.format(halo_name))
            #         continue #only skip if not plotting

            try:
                nstar = halo.calculate('NStar()')
            except Exception:
                print('  no NStar for {}, skipping'.format(halo_name))
                continue
            if nstar < MIN_NSTAR:
                continue

            print('Processing {} halo {} ({} stars)'.format(sim_name, hid, nstar))

            try:
                halo_dict = process_halo_ref(halo_ref, hid, imager, isophote,
                                             sim_name, orientation_keys)
            except Exception:
                print('  FAILED on {} halo {}:'.format(sim_name, hid))
                traceback.print_exc()
                continue

            ellipse_dict[sim_name][halo_ref] = halo_dict

            with open(OUTPUT_PICKLE, 'wb') as f:
                pickle.dump(ellipse_dict, f)
            print('  saved {} after halo {}'.format(OUTPUT_PICKLE, hid))

            gc.collect()

        del imager, isophote

    # ---- summary ----
    n_total = 0
    for sim_name in ellipse_dict:
        hids = [ref.split('_')[-1] for ref in ellipse_dict[sim_name]]
        n_total += len(hids)
        print('halos processed in sim {}:\n\t{}'.format(sim_name, ','.join(hids)))
    print('total number of halos {}'.format(n_total))

    with open(OUTPUT_PICKLE, 'wb') as f:
        pickle.dump(ellipse_dict, f)
    print(f'saved all halos, results in {OUTPUT_PICKLE}, figures in {FIGURE_DIR}')


if __name__ == '__main__':
    main()





