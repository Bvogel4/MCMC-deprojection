# Standard library imports
import gc
import logging
import multiprocessing as mp
import sys
import traceback
import warnings

# Third-party imports
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pynbody
from pynbody import array, units
from pynbody.plot.sph import image
import pymp
import scipy
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from skimage.measure import moments, moments_central
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model

# Local application imports
from tangos.properties import LivePropertyCalculation, LivePropertyCalculationInheritingMetaProperties
from tangos.properties.pynbody import PynbodyPropertyCalculation
from tangos.properties.pynbody.centring import centred_calculation

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IsophoteAnalysis")

def extract_single_value(value):
    """Extract scalar value from potentially nested data structures"""
    if isinstance(value, list):
        return value[0]
    return float(value)


def myprint(text, clear=False):
    """Custom print function for progress display"""
    import sys
    if clear:
        sys.stdout.write("\r" + " " * 100 + "\r")  # Clear line
    sys.stdout.write(f"\r{text}")
    sys.stdout.flush()


# logger = logging.getLogger('pynbody.analysis.halo')
logger = logging.getLogger('tangos')

#modifed from pynbody.analysis.halo to add parallelization within halos
def shape(sim, nbins=100, rmin=None, rmax=None, bins='equal',
          ndim=3, max_iterations=100, tol=1e-3, justify=False, weighting = 'None', mass_array = 'mass'):
    """Calculates the shape of the provided particles in homeoidal shells, over a range of nbins radii.

    Homeoidal shells maintain a fixed area (ndim=2) or volume (ndim=3). Note that all provided particles are used in
    calculating the shape, so e.g. to measure dark matter halo shape from a halo with baryons, you should pass
    only the dark matter particles.

    The simulation must be pre-centred, e.g. using :func:`center`.

    The algorithm is sensitive to substructure, which should ideally be removed.

    Caution is advised when assigning large number of bins and radial ranges with many particles, as the
    algorithm becomes very slow.

    Parameters
    ----------

      nbins : int
          The number of homeoidal shells to consider. Shells with few particles will take longer to fit.

      rmin : float
          The minimum radial bin in units of sim['pos']. By default this is taken as rout/1000.
          Note that this applies to axis a, so particles within this radius may still be included within
          homeoidal shells.

      rmax : float
          The maximum radial bin in units of sim['pos']. By default this is taken as the largest radial value
          in the halo particle distribution.

      bins : str
          The spacing scheme for the homeoidal shell bins. 'equal' initialises radial bins with equal numbers
          of particles, with the exception of the final bin which will accomodate remainders. This
          number is not necessarily maintained during fitting. 'log' and 'lin' initialise bins
          with logarithmic and linear radial spacing.

      ndim : int
          The number of dimensions to consider; either 2 or 3 (default). If ndim=2, the shape is calculated
          in the x-y plane. If using ndim=2, you may wish to make a cut in the z direction before
          passing the particles to this routine (e.g. using :class:`pynbody.filt.BandPass`).

      max_iterations : int
          The maximum number of shape calculations (default 10). Fewer iterations will result in a speed-up,
          but with a bias towards spheroidal results.

      tol : float
          Convergence criterion for the shape calculation. Convergence is achieved when the axial ratios have
          a fractional change <=tol between iterations.

      justify : bool
          Align the rotation matrix directions such that they point in a single consistent direction
          aligned with the overall halo shape. This can be useful if working with slerps.

    Returns
    -------

      rbin : SimArray
          The radial bins used for the fitting

      axis_lengths : SimArray
          A nbins x ndim array containing the axis lengths of the ellipsoids in each shell

      num_particles : np.ndarray
          The number of particles within each bin

      rotation_matrices : np.ndarray
          The rotation matrices for each shell

    """

    # Sanitise inputs:
    if (rmax == None): rmax = sim['r'].max()
    if (rmin == None): rmin = rmax / 1E3
    assert ndim in [2, 3]
    assert max_iterations > 0
    assert tol > 0
    assert rmin >= 0
    assert rmax > rmin
    assert nbins > 0
    if ndim == 2:
        assert np.sum((sim['rxy'] >= rmin) & (sim['rxy'] < rmax)) > nbins * 2
    elif ndim == 3:
        assert np.sum((sim['r'] >= rmin) & (sim['r'] < rmax)) > nbins * 2
    if bins not in ['equal', 'log', 'lin']: bins = 'equal'

    # Handy 90 degree rotation matrices:
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # -----------------------------FUNCTIONS-----------------------------
    sn = lambda r, N: np.append([r[i * int(len(r) / N):(1 + i) * int(len(r) / N)][0] \
                                 for i in range(N)], r[-1])

    # General equation for an ellipse/ellipsoid:
    def Ellipsoid(pos, a, R):
        x = np.dot(R.T, pos.T)
        return np.sum(np.divide(x.T, a) ** 2, axis=1)

    # Define moment of inertia tensor:
    #mass weighted tensor
    def MoI_R_ell(r, m, a, ndim=3):
        ba = a[1] / a[0]
        ca = a[2] / a[0]
        r_ell = (r[:, 0]**2 + (r[:, 1]/ba)**2 + (r[:, 2]/ca)**2)
        S = np.zeros([ndim, ndim])
        for i in range(ndim):
            for j in range(ndim):
                S[i, j] = np.sum(m * r[:, i] * r[:, j]/r_ell)
        return S

    def MoI(r, m, a, ndim=3):
        return np.array([[np.sum(m * r[:, i] * r[:, j]) for j in range(ndim)] for i in range(ndim)])


    # Calculate the shape in a single shell:
    def shell_shape(r, pos, mass, a, R, r_range, ndim=3, func = MoI):

        # Find contents of homoeoidal shell:
        mult = r_range / np.mean(a)
        in_shell = (r > min(a) * mult[0]) & (r < max(a) * mult[1])
        pos, mass = pos[in_shell], mass[in_shell]
        inner = Ellipsoid(pos, a * mult[0], R)
        outer = Ellipsoid(pos, a * mult[1], R)
        in_ellipse = (inner > 1) & (outer < 1)
        ellipse_pos, ellipse_mass = pos[in_ellipse], mass[in_ellipse]

        # End if there is no data in range:
        if not len(ellipse_mass):
            return a, R, np.sum(in_ellipse)

        # Calculate shape tensor & diagonalise:
        D = list(np.linalg.eigh(func(ellipse_pos, ellipse_mass,a ,ndim) / np.sum(ellipse_mass)))

        # Rescale axis ratios to maintain constant ellipsoidal volume:
        R2 = np.array(D[1])
        a2 = np.sqrt(abs(D[0]) * ndim)
        div = (np.prod(a) / np.prod(a2)) ** (1 / float(ndim))
        a2 *= div

        return a2, R2, np.sum(in_ellipse)

    # Re-align rotation matrix:
    def realign(R, a, ndim):
        if ndim == 3:
            if a[0] > a[1] > a[2] < a[0]:
                pass  # abc
            elif a[0] > a[1] < a[2] < a[0]:
                R = np.dot(R, Rx)  # acb
            elif a[0] < a[1] > a[2] < a[0]:
                R = np.dot(R, Rz)  # bac
            elif a[0] < a[1] > a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Ry)  # bca
            elif a[0] > a[1] < a[2] > a[0]:
                R = np.dot(np.dot(R, Rx), Rz)  # cab
            elif a[0] < a[1] < a[2] > a[0]:
                R = np.dot(R, Ry)  # cba
        elif ndim == 2:
            if a[0] > a[1]:
                pass  # ab
            elif a[0] < a[1]:
                R = np.dot(R, Rz[:2, :2])  # ba
        return R

    # Calculate the angle between two vectors:
    def angle(a, b):
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Flip x,y,z axes of R2 if they provide a better alignment with R1.
    def flip_axes(R1, R2):
        for i in range(len(R1)):
            if angle(R1[:, i], -R2[:, i]) < angle(R1[:, i], R2[:, i]):
                R2[:, i] *= -1
        return R2

    def process_bin(i, r, pos, mass, rbins, bin_edges, ndim, func):
        a = np.ones(ndim) * rbins[i]
        R = np.identity(ndim)

        for j in range(max_iterations):
            a2 = a.copy()
            a, R, N = shell_shape(r, pos, mass, a, R, bin_edges[[i, i + 1]], ndim, func)

            convergence_criterion = np.all(np.isclose(np.sort(a), np.sort(a2), rtol=tol))
            if convergence_criterion:
                R = realign(R, a, ndim)
                if np.sign(np.linalg.det(R)) == -1:
                    R[:, 1] *= -1
                a = np.flip(np.sort(a))
                return i, a, R, N
        return i, np.ones(ndim) * np.nan, np.identity(ndim) * np.nan, 0


    # -----------------------------FUNCTIONS-----------------------------

    # Set up binning:
    r = np.array(sim['r']) if ndim == 3 else np.array(sim['rxy'])
    pos = np.array(sim['pos'])[:, :ndim]

    #change this to change quantity we are analysing, such as luminosity!
    mass = np.array(sim[mass_array])

    if (bins == 'equal'):  # Bins contain equal number of particles
        full_bins = sn(np.sort(r[(r >= rmin) & (r <= rmax)]), nbins * 2)
        bin_edges = full_bins[0:nbins * 2 + 1:2]
        rbins = full_bins[1:nbins * 2 + 1:2]
    elif (bins == 'log'):  # Bins are logarithmically spaced
        bin_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        rbins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    elif (bins == 'lin'):  # Bins are linearly spaced
        bin_edges = np.linspace(rmin, rmax, nbins + 1)
        rbins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Initialise the shape arrays:
    rbins = array.SimArray(rbins, sim['pos'].units)
    axis_lengths = array.SimArray(np.zeros([nbins, ndim]), sim['pos'].units) * np.nan
    N_in_bin = np.zeros(nbins).astype('int')
    V_in_bin = np.zeros(nbins)
    #create an array with n R = np.identity(ndim)
    rotations = np.array([np.identity(ndim)] * nbins) * np.nan


    # Calculate the shape in each bin:
    threads = 40
    #create shared objects axis_lengths, N_in_bin, rotations
    shared_results = pymp.shared.dict()

    #choose function based on weighting
    if weighting == 'none':
        func = MoI
    if weighting == 'elliptical':
        func = MoI_R_ell


    with pymp.Parallel(threads) as p:
        for i in p.range(nbins):
            #p.print(f'Processing bin {i}')
            i, a, R, N = process_bin(i, r, pos, mass, rbins, bin_edges, ndim, func)
            #store in results
            with p.lock:
                shared_results[i] = (a, R, N)
    #unpack results
    results = dict(shared_results)
    for i in range(nbins):
        a, R, N = results[i]
        axis_lengths[i] = a
        N_in_bin[i] = N
        rotations[i] = R

    # for each bin, calculate the shape sensitivity
    # eq 7 from https://www.aanda.org/articles/aa/abs/2023/02/aa45031-22/aa45031-22.html
    # sensitivity = V_i / sqrt(N_i) * abs(N_(i-1)/V_(i-1) - N_(i+1)/V_(i+1))
    # where V is the volume of the shell and N is the number of particles in the shell

    # Ensure the axis vectors point in a consistent direction:
    if justify:
        _, _, _, R_global = shape(sim, nbins=1, rmin=rmin, rmax=rmax, ndim=ndim)
        rotations = np.array([flip_axes(R_global, i) for i in rotations])
    #print(rotations[0])
    axis_lengths = np.squeeze(axis_lengths.T).T
    rotations = np.squeeze(rotations)

    return rbins, axis_lengths, N_in_bin, rotations

def get_bins(n):
    if n < 1e4:
        return int(20 * n/1e4 /2 + 10)
    elif n >= 1e4:
        return int( (np.log10(n) - 3) ** 2 * 20)


# class ImageHalo(PynbodyPropertyCalculation):
#     """Generates V-band luminosity density images at different orientations and calculates
#     effective radii for each projection to study galaxy morphology"""
#
#     # Store the key properties we need for shape analysis
#     names = ['halo_images', 'image_reffs', 'image_orientations', 'Rhalf',
#              'profile_sb_v', 'profile_v_lum_den', 'profile_rbins',
#              'profile_lum_den', 'profile_mags_v', 'profile_binarea']
#
#     @staticmethod
#     def fit_sersic_profile(prof):
#         """Fits a Sérsic profile to determine the effective radius (Reff) for each projection.
#
#         The Sérsic profile describes how galaxy brightness varies with radius:
#         μ(r) = μeff + 2.5(0.868n - 0.142)((r/reff)^(1/n) - 1)
#         where μeff is surface brightness at effective radius, n is Sérsic index
#         """
#
#         def sersic(r, mueff, reff, n):
#             return mueff + 2.5 * (0.868 * n - 0.142) * (
#                     (r / reff) ** (1. / n) - 1)
#
#         # Smooth the V-band surface brightness profile to reduce noise
#         vband = prof['sb,V']
#         smooth = np.nanmean(
#             np.pad(vband.astype(float), (0, 3 - vband.size % 3),
#                    mode='constant', constant_values=np.nan).reshape(-1, 3),
#             axis=1)
#
#         # Set up radial coordinates for fitting
#         x = np.arange(
#             len(smooth)) * 0.3 + 0.15  # Convert to physical units (kpc)
#         x[0] = .05  # Avoid r=0 singularity
#
#         # Remove any NaN values before fitting
#         y = smooth[~np.isnan(smooth)]
#         x = x[~np.isnan(smooth)]
#
#         # Initial guesses for fit parameters
#         r0 = x[int(len(x) / 2)]  # Initial Reff guess is middle of radial range
#         m0 = np.mean(
#             y[:3])  # Initial surface brightness guess from central region
#
#         # Fit Sérsic profile with reasonable bounds for galaxy parameters
#         par, _ = curve_fit(sersic, x, y, p0=(m0, r0, 1),
#                            bounds=([10, 0, 0.5], [40, 100, 16.5]))
#         return par[1]  # Return fitted Reff
#
#     @staticmethod
#     def generate_image(stars, width):
#         f = plt.figure(frameon=False)
#         f.set_size_inches(10, 10)
#         ax = plt.Axes(f, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         f.add_axes(ax)
#         im = pynbody.plot.sph.image(stars, qty='V_lum_den', width=width, subplot=ax, units='kpc^-2', resolution=1000,
#                                     show_cbar=False, ret_im=True)
#         data = im.get_array()  # Get the numpy array
#         plt.close(f)
#         return data
#
#
#     def process_orientation(self,halo, width, Rhalf):
#         orientation_data = {'Rhalf': Rhalf.view(np.ndarray)}
#         prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25, max=5 * Rhalf, ndim=2,
#                                                 nbins=int((5 * Rhalf) / 0.1))
#
#         orientation_data.update({
#             'sb,v': prof['sb,V'].copy().view(np.ndarray),
#             'v_lum_den': prof['V_lum_den'].copy().view(np.ndarray),
#             'rbins': prof['rbins'].copy().view(np.ndarray),
#             'lum_den': (10.0 ** (-0.4 * prof['magnitudes,V']) / prof._binsize.in_units('pc^2')).copy().view(np.ndarray),
#             'mags,v': prof['magnitudes,V'].copy().view(np.ndarray),
#             'binarea': prof._binsize.in_units('pc^2').copy().view(np.ndarray)
#         })
#         orientation_data['Reff'] = self.fit_sersic_profile(prof)
#
#         orientation_data['image'] = self.generate_image(halo.s, width)
#
#
#         return orientation_data
#
#     def process_halo(self, halo, existing_properties):
#         """Generate images and measure Reff across different viewing angles.
#
#         We sample viewing angles by rotating the galaxy through θ and φ to create
#         a set of 2D projections that mimic observational studies.
#         """
#         dx, dy = 30, 30  # Rotation increments (default: 30,30 for finer sampling)
#         halo.physical_units()  # Ensure physical units are used
#
#         # Orient galaxy face-on initially using gas angular momentum
#         pynbody.analysis.angmom.faceon(halo)
#         # make sure halo has stars
#         Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
#         width = 9 * Rhalf  # Image width captures extended structure
#
#         # Select stars within a sphere that contains full projection at any angle
#         ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)
#
#         xrotations = np.arange(0, 180, dx)
#         yrotations = np.arange(0, 360, dy)
#
#         # Create a list of all orientation combinations
#         all_orientations = [(x, y) for x in xrotations for y in yrotations]
#
#         # Shared dictionary to store results
#         shared_dict = pymp.shared.dict()
#
#         # Process all orientations in parallel
#         with pymp.Parallel(1) as p:  # Adjust number of processes as needed
#             for i in p.range(len(all_orientations)):
#                 xrotation, yrotation = all_orientations[i]
#                 # Apply rotations
#                 with halo.rotate_x(xrotation).rotate_y(yrotation):
#                     key = f'x{xrotation:03d}y{yrotation:03d}'
#                     sb_dict = self.process_orientation(halo.s[ImageSpace], width, Rhalf)
#
#                 # Store result with lock to avoid conflicts
#                 with p.lock:
#                     shared_dict[key] = sb_dict
#
#         shared_dict = dict(shared_dict)
#         # sort the dictionary by key
#         shared_dict = dict(sorted(shared_dict.items()))
#
#         # Unpack results from shared dictionary
#         orientations = list(shared_dict.keys())
#         #initialize lists
#         images = []
#         reff_values = []
#         profile_sb_v = []
#         profile_v_lum_den = []
#         profile_rbins = []
#         profile_lum_den = []
#         profile_mags_v = []
#         profile_binarea = []
#         for key in orientations:
#             images.append(shared_dict[key]['image'])
#             reff_values.append(shared_dict[key]['Reff'])
#             profile_sb_v.append(shared_dict[key]['sb,v'])
#             profile_v_lum_den.append(shared_dict[key]['v_lum_den'])
#             profile_rbins.append(shared_dict[key]['rbins'])
#             profile_lum_den.append(shared_dict[key]['lum_den'])
#             profile_mags_v.append(shared_dict[key]['mags,v'])
#             profile_binarea.append(shared_dict[key]['binarea'])
#
#
#         return images, reff_values, orientations, Rhalf, profile_sb_v, profile_v_lum_den, profile_rbins, profile_lum_den, profile_mags_v, profile_binarea
#     def calculate(self,halo,existing_properties):
#         return self.process_halo(halo, existing_properties)
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

class ImageHalo(PynbodyPropertyCalculation):
    """Base class for generating luminosity/density images at different orientations
    and calculating effective radii for each projection to study galaxy morphology"""

    # These should be overridden in subclasses
    imaging_qty = None  # e.g., 'V_lum_den', 'U_lum_den', 'rho'
    imaging_units = None  # e.g., 'kpc^-2', 'Msol kpc^-3'
    particle_type_attr = None  # e.g., 's', 'dm', 'g'
    sb_profile_key = None  # e.g., 'sb,V', 'sb,U', None for mass density
    lum_den_key = None  # e.g., 'V_lum_den', 'U_lum_den', None for mass density
    magnitude_key = None  # e.g., 'magnitudes,V', 'magnitudes,U', None for mass density

    @staticmethod
    def fit_sersic_profile(prof, sb_key):
        """Fits a Sérsic profile to determine the effective radius (Reff) for each projection.

        The Sérsic profile describes how galaxy brightness varies with radius:
        μ(r) = μeff + 2.5(0.868n - 0.142)((r/reff)^(1/n) - 1)
        where μeff is surface brightness at effective radius, n is Sérsic index
        """

        def sersic(r, mueff, reff, n):
            return mueff + 2.5 * (0.868 * n - 0.142) * (
                    (r / reff) ** (1. / n) - 1)

        # Smooth the surface brightness profile to reduce noise
        if sb_key is None:
            # For mass density, we might need to calculate surface brightness differently
            # or use a different approach - this would need to be implemented based on your needs
            raise NotImplementedError("Sérsic fitting for mass density not yet implemented")

        profile_data = prof[sb_key]
        smooth = np.nanmean(
            np.pad(profile_data.astype(float), (0, 3 - profile_data.size % 3),
                   mode='constant', constant_values=np.nan).reshape(-1, 3),
            axis=1)

        # Set up radial coordinates for fitting
        x = np.arange(len(smooth)) * 0.3 + 0.15  # Convert to physical units (kpc)
        x[0] = .05  # Avoid r=0 singularity

        # Remove any NaN values before fitting
        y = smooth[~np.isnan(smooth)]
        x = x[~np.isnan(smooth)]

        # Initial guesses for fit parameters
        r0 = x[int(len(x) / 2)]  # Initial Reff guess is middle of radial range
        m0 = np.mean(y[:3])  # Initial surface brightness guess from central region

        # Fit Sérsic profile with reasonable bounds for galaxy parameters
        par, _ = curve_fit(sersic, x, y, p0=(m0, r0, 1),
                           bounds=([10, 0, 0.5], [40, 100, 16.5]))
        return par[1]  # Return fitted Reff

    def generate_image(self, particles, width):
        """Generate image with smoothing appropriate for isophote fitting"""
        f = plt.figure(frameon=False)
        f.set_size_inches(10, 10)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)

        #set smooth floor approximately to gravitational softening length
        try:
            eps = particles['eps']
        except KeyError:
            eps = particles.properties['eps']

        eps = np.median(eps)

        smooth_floor = 3*eps

        im = pynbody.plot.sph.image(
            particles,
            qty=self.imaging_qty,
            width=width,
            subplot=ax,
            units=self.imaging_units,
            resolution=1000,
            smooth_floor=smooth_floor,  # <-- Key addition
            denoise=False,  # <-- Optional
            show_cbar=False,
            ret_im=True
        )
        data = im.get_array()
        plt.close(f)
        return data

    def process_orientation(self, particles, width, Rhalf):
        """Process a single orientation and extract relevant profile data"""
        orientation_data = {'Rhalf': Rhalf.view(np.ndarray)}
        prof = pynbody.analysis.profile.Profile(particles, type='lin', min=.25, max=5 * Rhalf, ndim=2,
                                                nbins=int((5 * Rhalf) / 0.1))

        # Always store the shared properties
        orientation_data.update({
            'rbins': prof['rbins'].copy().view(np.ndarray),
            'binarea': prof._binsize.in_units('pc^2').copy().view(np.ndarray)
        })

        # Store imaging-specific properties
        if self.sb_profile_key:
            orientation_data[f'sb_{self.imaging_qty.split("_")[0]}'] = prof[self.sb_profile_key].copy().view(np.ndarray)

        if self.lum_den_key:
            orientation_data[f'{self.imaging_qty}'] = prof[self.lum_den_key].copy().view(np.ndarray)

        if self.magnitude_key:
            orientation_data[f'mags_{self.imaging_qty.split("_")[0]}'] = prof[self.magnitude_key].copy().view(
                np.ndarray)
            orientation_data['lum_den'] = (
                        10.0 ** (-0.4 * prof[self.magnitude_key]) / prof._binsize.in_units('pc^2')).copy().view(
                np.ndarray)

        # For mass density, handle differently
        if self.imaging_qty == 'rho':
            orientation_data['rho'] = prof['density'].copy().view(np.ndarray)

        # Fit Sérsic profile if applicable
        if self.sb_profile_key:
            orientation_data['Reff'] = self.fit_sersic_profile(prof, self.sb_profile_key)
        else:
            # For mass density, you might want to define a different effective radius measure
            orientation_data['Reff'] = np.nan  # or implement alternative method

        orientation_data['image'] = self.generate_image(particles, width)

        return orientation_data

    def process_halo(self, halo, existing_properties):
        """Generate images and measure Reff across different viewing angles."""
        dx, dy = 10, 10  # Rotation increments (default: 30,30 for finer sampling)
        halo.physical_units()  # Ensure physical units are used

        # Orient galaxy face-on initially using gas angular momentum
        pynbody.analysis.angmom.faceon(halo)

        # Get the appropriate particle type
        particles = getattr(halo, self.particle_type_attr)

        # Calculate half-light radius (or equivalent for mass density)
        if self.imaging_qty == 'rho':
            #get V-band rhalf from existing properties if available
            Rhalf = existing_properties.get('Rhalf_v', None)
            if Rhalf is None:
                Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        else:
            Rhalf = pynbody.analysis.luminosity.half_light_r(halo)

        width = 9 * Rhalf  # Image width captures extended structure

        # Select particles within a sphere that contains full projection at any angle
        ImageSpace = pynbody.filt.Sphere(width * np.sqrt(2) * 1.01)

        theta_l, phi_l, xrotations, yrotations = create_spherical_grid(100)
        shared_dict = {}

        # with pymp.Parallel(self.n_procs) as p:
        for i in range(len(xrotations)):
            xrotation, yrotation = xrotations[i], yrotations[i]
            # key = f'x{xrotation:03d}y{yrotation:03d}'
            key = str(theta_l[i])+','+str(phi_l[i])
            # Apply rotations
            with halo.rotate_x(xrotation).rotate_y(yrotation):
                sb_dict = self.process_orientation(particles[ImageSpace], width, Rhalf)

            shared_dict[key] = sb_dict

        #shared_dict = dict(shared_dict)
        # sort the dictionary by key
        shared_dict = dict(sorted(shared_dict.items()))

        # Unpack results from shared dictionary
        orientations = list(shared_dict.keys())

        # Initialize lists for all possible outputs
        images = []
        reff_values = []

        # Shared properties (same across all imaging types for this particle type)
        profile_rbins = []
        profile_binarea = []

        # Imaging-specific properties
        profile_sb = []
        profile_lum_den = []
        profile_mags = []
        profile_lum_den_calc = []
        profile_rho = []

        for key in orientations:
            images.append(shared_dict[key]['image'])
            reff_values.append(shared_dict[key]['Reff'])
            profile_rbins.append(shared_dict[key]['rbins'])
            profile_binarea.append(shared_dict[key]['binarea'])

            # Add imaging-specific data if it exists
            if f'sb_{self.imaging_qty.split("_")[0]}' in shared_dict[key]:
                profile_sb.append(shared_dict[key][f'sb_{self.imaging_qty.split("_")[0]}'])
            if self.imaging_qty in shared_dict[key]:
                profile_lum_den.append(shared_dict[key][self.imaging_qty])
            if f'mags_{self.imaging_qty.split("_")[0]}' in shared_dict[key]:
                profile_mags.append(shared_dict[key][f'mags_{self.imaging_qty.split("_")[0]}'])
            if 'lum_den' in shared_dict[key]:
                profile_lum_den_calc.append(shared_dict[key]['lum_den'])
            if 'rho' in shared_dict[key]:
                profile_rho.append(shared_dict[key]['rho'])

        # Return all the data - subclasses will select what they need
        return {
            'images': images,
            'reff_values': reff_values,
            'orientations': orientations,
            'Rhalf': Rhalf,
            'profile_rbins': profile_rbins,
            'profile_binarea': profile_binarea,
            'profile_sb': profile_sb,
            'profile_lum_den': profile_lum_den,
            'profile_mags': profile_mags,
            'profile_lum_den_calc': profile_lum_den_calc,
            'profile_rho': profile_rho
        }

    def calculate(self, halo, existing_properties):
        return self.process_halo(halo, existing_properties)


# Specific subclasses for different bands and particle types

class VBandStarImages(ImageHalo):
    """V-band images for stellar particles"""
    names = ['halo_images_v', 'image_reffs_v', 'image_orientations_v', 'Rhalf_v',
             'profile_sb_v', 'profile_v_lum_den', 'profile_rbins_v',
             'profile_lum_den_v', 'profile_mags_v', 'profile_binarea_v']

    imaging_qty = 'V_lum_den'
    imaging_units = 'kpc^-2'
    particle_type_attr = 's'
    sb_profile_key = 'sb,V'
    lum_den_key = 'V_lum_den'
    magnitude_key = 'magnitudes,V'

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_sb'], data['profile_lum_den'], data['profile_rbins'],
                data['profile_lum_den_calc'], data['profile_mags'], data['profile_binarea'])


class UBandStarImages(ImageHalo):
    """U-band images for stellar particles"""
    names = ['halo_images_u', 'image_reffs_u', 'image_orientations_u', 'Rhalf_u',
             'profile_sb_u', 'profile_u_lum_den', 'profile_rbins_u',
             'profile_lum_den_u', 'profile_mags_u', 'profile_binarea']

    imaging_qty = 'U_lum_den'
    imaging_units = 'kpc^-2'
    particle_type_attr = 's'
    sb_profile_key = 'sb,U'
    lum_den_key = 'U_lum_den'
    magnitude_key = 'magnitudes,U'

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_sb'], data['profile_lum_den'], data['profile_rbins'],
                data['profile_lum_den_calc'], data['profile_mags'], data['profile_binarea'])


class IBandStarImages(ImageHalo):
    """I-band images for stellar particles"""
    names = ['halo_images_i', 'image_reffs_i', 'image_orientations_i', 'Rhalf_i',
             'profile_sb_i', 'profile_i_lum_den', 'profile_rbins_i',
             'profile_lum_den_i', 'profile_mags_i', 'profile_binarea_i']

    imaging_qty = 'I_lum_den'
    imaging_units = 'kpc^-2'
    particle_type_attr = 's'
    sb_profile_key = 'sb,I'
    lum_den_key = 'I_lum_den'
    magnitude_key = 'magnitudes,I'

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_sb'], data['profile_lum_den'], data['profile_rbins'],
                data['profile_lum_den_calc'], data['profile_mags'], data['profile_binarea'])


class MassDensityStarImages(ImageHalo):
    """Mass density images for stellar particles"""
    names = ['halo_images_rho_stars', 'image_reffs_rho_stars', 'image_orientations_rho_stars', 'Rhalf_rho_stars',
             'profile_rho_stars', 'profile_rbins_stars', 'profile_binarea_stars']

    imaging_qty = 'rho'
    imaging_units = 'Msol kpc^-3'
    particle_type_attr = 's'
    sb_profile_key = None
    lum_den_key = None
    magnitude_key = None

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_rho'], data['profile_rbins'], data['profile_binarea'])


class MassDensityDMImages(ImageHalo):
    """Mass density images for dark matter particles"""
    names = ['halo_images_rho_dm', 'image_reffs_rho_dm', 'image_orientations_dm', 'Rhalf_rho_dm',
             'profile_rho_dm', 'profile_rbins_dm', 'profile_binarea_dm']

    imaging_qty = 'rho'
    imaging_units = 'Msol kpc^-3'
    particle_type_attr = 'dm'
    sb_profile_key = None
    lum_den_key = None
    magnitude_key = None

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_rho'], data['profile_rbins'], data['profile_binarea'])


class MassDensityGasImages(ImageHalo):
    """Mass density images for gas particles"""
    names = ['halo_images_rho_gas', 'image_reffs_rho_gas', 'image_orientations_gas', 'Rhalf_rho_gas',
             'profile_rho_gas', 'profile_rbins_gas', 'profile_binarea_gas']

    imaging_qty = 'rho'
    imaging_units = 'Msol kpc^-3'
    particle_type_attr = 'g'
    sb_profile_key = None
    lum_den_key = None
    magnitude_key = None

    def calculate(self, halo, existing_properties):
        data = self.process_halo(halo, existing_properties)
        return (data['images'], data['reff_values'], data['orientations'], data['Rhalf'],
                data['profile_rho'], data['profile_rbins'], data['profile_binarea'])


class IsophoteAnalysis(LivePropertyCalculation):
    """Analyzes isophotes to measure projected galaxy shapes at different radii.

    For each projection, we measure ellipticity and position angle at 2-4 Reff
    to track how galaxy shape varies with radius.

    Initial ellipse geometry is seeded from a 2-D elliptical Sérsic fit
    (via SersicFitter.fit_2d_log) rather than image moments, which gives a
    more physically motivated starting point for the isophote tracer.
    """

    def __init__(self, simulation, image_type='v_stars'):
        # print('\ninitializing isophote analysis\n', image_type)
        # print('\n names', self.names)

        super().__init__(simulation)
        self.image_type = image_type
        self.visualization_enabled = False

        self.property_mappings = {
            'v_stars': {
                'images': 'halo_images_v',
                'reffs': 'image_reffs_v',
                'orientations': 'image_orientations_v',
                'rhalf': 'Rhalf_v'
            },
            'u_stars': {
                'images': 'halo_images_u',
                'reffs': 'image_reffs_u',
                'orientations': 'image_orientations_u',
                'rhalf': 'Rhalf_u'
            },
            'i_stars': {
                'images': 'halo_images_i',
                'reffs': 'image_reffs_i',
                'orientations': 'image_orientations_i',
                'rhalf': 'Rhalf_i'
            },
            'rho_stars': {
                'images': 'halo_images_rho_stars',
                'reffs': 'image_reffs_rho_stars',
                'orientations': 'image_orientations_rho_stars',
                'rhalf': 'Rhalf_rho_stars'
            },
            'rho_dm': {
                'images': 'halo_images_rho_dm',
                'reffs': 'image_reffs_rho_dm',
                'orientations': 'image_orientations_dm',
                'rhalf': 'Rhalf_rho_dm'
            },
            'rho_gas': {
                'images': 'halo_images_rho_gas',
                'reffs': 'image_reffs_rho_gas',
                'orientations': 'image_orientations_gas',
                'rhalf': 'Rhalf_rho_gas'
            }
        }

        if self.image_type not in self.property_mappings:
            available_types = list(self.property_mappings.keys())
            raise ValueError(
                f"Unknown image type: {self.image_type}. "
                f"Available types: {available_types}"
            )

    # ------------------------------------------------------------------
    # Abstract / interface methods
    # ------------------------------------------------------------------

    def requires_property(self):
        raise NotImplementedError(
            "Subclasses must implement requires_property() to specify required properties."
        )

    def check_properties_exist(self, existing_properties):
        """Raise ValueError if any property required for this image type is absent."""
        missing = [p for p in self.requires_property()
                   if p not in existing_properties]
        if missing:
            raise ValueError(
                f"Missing required properties for image type "
                f"'{self.image_type}': {missing}"
            )

    # ------------------------------------------------------------------
    # Initial-parameter estimation via SersicFitter
    # ------------------------------------------------------------------

    @staticmethod
    def _sersic_initial_params(image, reff_kpc, kpc_per_pixel):
        """Estimate ellipse geometry from a 2-D Sérsic fit.

        Uses SersicFitter.fit_2d_log to obtain center (x0, y0 in pixels),
        position angle (theta), ellipticity (1 - q), and a starting
        semi-major-axis factor relative to reff.

        Falls back to sensible defaults if the fit fails.

        Args:
            image:         2-D flux / surface-brightness array.
            reff_kpc:      Effective radius in kpc for this projection.
            kpc_per_pixel: Pixel scale in kpc/pixel.

        Returns:
            (center_x, center_y, eps, pa, radius_factor)
            with center in pixels, eps in [0.01, 0.85],
            pa in radians, and radius_factor in [2.5, 4.5].
        """
        result = SersicFitter.fit_2d_log(image, kpc_per_pixel)

        # fit_2d_log works in kpc coordinates (pixels × pixel_scale),
        # so convert x0/y0 back to pixel coordinates.
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

        Args:
            image_data:        2-D flux array.
            radius:            Effective radius in kpc.
            step_size_factors: Unused legacy argument (kept for API compatibility).
            center:            Initial (x, y) centre in pixels.
            eps:               Initial ellipticity.
            pa:                Initial position angle in radians.
            sma_factor:        Starting SMA as a multiple of radius_pixels.
            kpc_per_pixel:     Pixel scale in kpc/pixel.
            plot:              Passed through; currently unused.
            apply_smoothing:   Passed through; currently unused.
            smoothing_kpc:     Passed through; currently unused.

        Returns:
            (result, all_targets_met)
            result is a list of [sma_px, eps, pa, grad_err, x0, y0, intens, rms]
            for each of the three target radii (2, 3, 4 × Reff).
        """
        from scipy.interpolate import interp1d

        radius_pixels   = radius / kpc_per_pixel
        target_multipliers = [2.0, 3.0, 4.0]
        target_radii_kpc   = {m: m * radius for m in target_multipliers}

        geometry = EllipseGeometry(
            x0=center[0], y0=center[1],
            sma=sma_factor * radius_pixels,
            eps=eps, pa=pa
        )

        try:
            ellipse  = Ellipse(image_data, geometry)
            isolist  = ellipse.fit_image(
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

        smas_kpc  = isolist.sma * kpc_per_pixel
        good_mask = isolist.grad_r_error < 0.2
        if not np.any(good_mask):
            good_mask = np.ones(len(isolist.sma), dtype=bool)

        sort_idx     = np.argsort(smas_kpc[good_mask])
        good_smas    = smas_kpc[good_mask][sort_idx]
        good_eps     = isolist.eps[good_mask][sort_idx]
        good_pa      = isolist.pa[good_mask][sort_idx]
        good_x0      = isolist.x0[good_mask][sort_idx]
        good_y0      = isolist.y0[good_mask][sort_idx]
        good_intens  = isolist.intens[good_mask][sort_idx]
        good_gerr    = isolist.grad_r_error[good_mask][sort_idx]

        result       = []
        targets_met  = {m: False for m in target_multipliers}

        if len(good_smas) >= 2:
            try:
                f_eps = interp1d(good_smas, good_eps,  kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_pa  = interp1d(good_smas, good_pa,   kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_x0  = interp1d(good_smas, good_x0,   kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_y0  = interp1d(good_smas, good_y0,   kind='linear',
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

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def get_isophote(self, existing_properties):
        """Fit isophotes to every projection, seeding geometry from SersicFitter.

        Args:
            existing_properties: Dict of already-computed halo properties.

        Returns:
            List of isophote parameter lists, one entry per orientation,
            sorted by orientation key.
        """
        print(f'\tGenerating isophotes for {self.image_type}...\n')
        self.check_properties_exist(existing_properties)

        mapping      = self.property_mappings[self.image_type]
        images       = existing_properties[mapping['images']]
        reff_values  = existing_properties[mapping['reffs']]
        orientations = existing_properties[mapping['orientations']]
        Rhalf        = existing_properties[mapping['rhalf']]

        step_size_factors = [1.0, 0.5, 0.25, 0.125]  # kept for compatibility
        kpc_per_pixel     = (9 * Rhalf) / np.size(images[0],axis=0)

        params = {}

        for k, (image_data, radius, orientation) in enumerate(
                zip(images, reff_values, orientations)):

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

            pct = round((k + 1) / len(images) * 100, 2)
            myprint(
                f'\tGenerating isophotes for {self.image_type}: {pct}%',
                clear=True
            )

        # Return results sorted by orientation key
        return [params[key] for key in sorted(params)]

    def live_calculate(self, existing_properties):
        """Entry point called by tangos."""
        print(f'Calculating isophotes for {self.image_type}...')
        logger.info(f'Calculating isophotes for {self.image_type}...')
        return self.get_isophote(existing_properties)

# Convenience subclasses for common use cases

class VBandIsophoteAnalysis(IsophoteAnalysis):
    """V-band isophote analysis (default)"""
    names = 'isophote_parameters_v_stars'

    def requires_property(self):
        return  ['halo_images_v','image_reffs_v', 'image_orientations_v', 'Rhalf_v']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='v_stars')


class UBandIsophoteAnalysis(IsophoteAnalysis):
    """U-band isophote analysis"""
    names = 'isophote_parameters_u_stars'

    def requires_property(self):
        return  ['halo_images_u','image_reffs_u', 'image_orientations_u', 'Rhalf_u']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='u_stars')


class IBandIsophoteAnalysis(IsophoteAnalysis):
    """I-band isophote analysis"""
    names = 'isophote_parameters_i_stars'

    def requires_property(self):
        return  ['halo_images_i','image_reffs_i', 'image_orientations_i', 'Rhalf_i']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='i_stars')


class StellarMassIsophoteAnalysis(IsophoteAnalysis):
    """Stellar mass density isophote analysis"""
    names = 'isophote_parameters_rho_stars'

    def requires_property(self):
        return ['halo_images_rho_stars', 'image_reffs_rho_stars', 'image_orientations_rho_stars', 'Rhalf_rho_stars']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='rho_stars')


class DarkMatterIsophoteAnalysis(IsophoteAnalysis):
    """Dark matter density isophote analysis"""
    names = 'isophote_parameters_rho_dm'
    def requires_property(self):
        return ['halo_images_rho_dm', 'image_reffs_rho_dm', 'image_orientations_rho_dm', 'Rhalf_rho_dm']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='rho_dm')


class GasIsophoteAnalysis(IsophoteAnalysis):
    """Gas density isophote analysis"""
    names = 'isophote_parameters_rho_gas'
    def requires_property(self):
        return ['halo_images_rho_gas', 'image_reffs_rho_gas', 'image_orientations_rho_gas', 'Rhalf_rho_gas']

    def __init__(self, simulation):
        super().__init__(simulation, image_type='rho_gas')


class r_80(PynbodyPropertyCalculation):
    # get the radius that contains 80% of the mass of the stars
    names = ['r_80']
    def __init__(self, simulation):
        super().__init__(simulation)

    def calculate(self, halo, existing_properties):
        N_star = len(halo.s)
        # get radius that contains 80% of star particles
        rsort = halo.s['r'][np.argsort(halo.s['r'])]
        r_8 = rsort[int(0.8 * N_star)]
        return r_8



class BaseShapeCalculator:
    """
    Base class containing the core shape calculation logic.
    """

    @staticmethod
    def process_shape(particles, rin, rout, bins,component_name = None):

        if component_name == 'luminous':
            bins = 1
            weighting = 'elliptical'
            mass_array = 'V_lum'
            particles.s['V_lum'] = 10 ** (0.4 * (4.83 - particles.s['V_mag']))
            tol = 1e-2
            max_iterations = 100
        else:
            weighting = 'none'
            mass_array = 'mass'
            tol = 5e-3
            max_iterations = 175

        """Process shape calculation for given particles"""
        rbins, axis_lengths, num_particles, rotations = shape(particles,
                                                              nbins=bins,
                                                              ndim=3, rmin=rin,
                                                              rmax=rout,
                                                              max_iterations=max_iterations,
                                                              tol=tol,
                                                              justify=False,
                                                              weighting = weighting,
                                                              mass_array = mass_array)

        if len(rbins) > 1:
            ba = axis_lengths[:, 1] / axis_lengths[:, 0]
            ca = axis_lengths[:, 2] / axis_lengths[:, 0]
            shape_dict = {'ba': ba, 'ca': ca, 'rbins': rbins}
        elif len(rbins)==1:
            ba = axis_lengths[1] / axis_lengths[0]
            ca = axis_lengths[2] / axis_lengths[0]
            shape_dict = {'ba': ba, 'ca': ca, 'rbins': rbins}

        return shape_dict

    def calculate_component_shape(self, halo, rin=0.1, rout=None):
        """
        Calculate shape for this component with error handling.
        Returns nan_dict if calculation fails or no particles exist.
        """
        nan_array = np.array([np.nan] * 100)
        nan_dict = {'ba': nan_array, 'ca': nan_array, 'rbins': nan_array}

        # Prepare halo
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)

        try:
            particles = self.get_particles(halo)
            n_particles = len(particles)
            if n_particles == 0:
                return nan_dict

            bins = int(get_bins(n_particles))



            return self.process_shape(particles, rin, rout, bins,self.component_name)

        except Exception as e:
            print(f'Error calculating {self.component_name} shape: {e}')
            traceback.print_exc()
            raise e

    def get_particles(self, halo):
        """Override in subclasses to specify which particles to use"""
        raise NotImplementedError("Subclasses must implement get_particles")

    def plot_xlog(self):
        return False

    def plot_ylog(self):
        return False

    def plot_xlabel(self):
        return r'$r_{bins}$ (kpc)'

    def plot_ylabel(self):
        return r'$b/a$ or $c/a$'


class StellarShape(BaseShapeCalculator, PynbodyPropertyCalculation):
    """Calculate stellar component shape profile"""

    names = ['ba_s', 'ca_s', 'rbins_s']
    component_name = 'stellar'

    def __init__(self, simulation):
        PynbodyPropertyCalculation.__init__(self, simulation)

    def get_particles(self, halo):
        """Get stellar particles from halo"""
        return halo.s

    def calculate(self, halo, existing_properties):
        """Calculate stellar shape"""
        shape_dict = self.calculate_component_shape(halo)
        return shape_dict['ba'], shape_dict['ca'], shape_dict['rbins']

    def plot_xvalues(self, for_data):
        """Return stellar radial bins for x-axis"""
        return for_data['rbins_s']

    def plot_ylabel(self):
        return r'Stellar $b/a$ or $c/a$'


class DarkMatterShape(BaseShapeCalculator, PynbodyPropertyCalculation):
    """Calculate dark matter component shape profile"""

    names = ['ba_d', 'ca_d', 'rbins_d']
    component_name = 'dark_matter'

    def __init__(self, simulation):
        PynbodyPropertyCalculation.__init__(self, simulation)

    def get_particles(self, halo):
        """Get dark matter particles from halo"""
        return halo.dm

    def calculate(self, halo, existing_properties):
        """Calculate dark matter shape"""
        shape_dict = self.calculate_component_shape(halo)
        return shape_dict['ba'], shape_dict['ca'], shape_dict['rbins']

    def plot_xvalues(self, for_data):
        """Return dark matter radial bins for x-axis"""
        return for_data['rbins_d']

    def plot_ylabel(self):
        return r'Dark Matter $b/a$ or $c/a$'


class GasShape(BaseShapeCalculator, PynbodyPropertyCalculation):
    """Calculate gas component shape profile"""

    names = ['ba_g', 'ca_g', 'rbins_g']
    component_name = 'gas'

    def __init__(self, simulation):
        PynbodyPropertyCalculation.__init__(self, simulation)

    def get_particles(self, halo):
        """Get gas particles from halo"""
        return halo.g

    def calculate(self, halo, existing_properties):
        """Calculate gas shape"""
        shape_dict = self.calculate_component_shape(halo)
        return shape_dict['ba'], shape_dict['ca'], shape_dict['rbins']

    def plot_xvalues(self, for_data):
        """Return gas radial bins for x-axis"""
        return for_data['rbins_g']

    def plot_ylabel(self):
        return r'Gas $b/a$ or $c/a$'


class StellarShapeLuminous(BaseShapeCalculator, PynbodyPropertyCalculation):
    """Calculate stellar component shape profile"""

    names = ['ba_s_v', 'ca_s_v', 'rbins_s_v']
    component_name = 'luminous'

    def requires_property(self):
        return ['Rvir']

    def __init__(self, simulation):
        PynbodyPropertyCalculation.__init__(self, simulation)

    def get_particles(self, halo):
        """Get stellar particles from halo"""
        return halo.s

    def calculate(self, halo, existing_properties):
        """Calculate stellar shape"""
        Rvir  = extract_single_value(existing_properties['Rvir'])
        #let's make sure this is just one value and not an array
        rout = Rvir * 0.1
        shape_dict = self.calculate_component_shape(halo,rout=rout)
        return shape_dict['ba'], shape_dict['ca'], shape_dict['rbins']

    def plot_xvalues(self, for_data):
        """Return stellar radial bins for x-axis"""
        return for_data['rbins_s']

    def plot_ylabel(self):
        return r'Stellar $b/a$ or $c/a$'


class SmoothAxisRatio(LivePropertyCalculation):
    names = ['r_s_f', 'ba_s_smoothed', 'ca_s_smoothed', 'r_d_f', 'ba_d_smoothed', 'ca_d_smoothed', 'r_g_f',
             'ba_g_smoothed', 'ca_g_smoothed']

    def requires_property(self):
        return ['ba_s', 'ba_d', 'ba_g', 'ca_s', 'ca_d', 'ca_g', 'rbins_s', 'rbins_d', 'rbins_g']

    @staticmethod
    def nan_func(x):
        return np.nan

    @staticmethod
    def smooth_shape(rbins, ba, ca, k=3):
        s_factor = 1
        """
        Smooth and filter data, handling a few NaN values gracefully.

        Parameters:
        rbins, ba, ca: array-like, input data
        k: int, degree of the smoothing spline (default 3, recommended cubic)
        s_factor: float, smoothing factor as a fraction of len(rbins) (default 1)

        Returns:
        rbins, ba, ca: filtered arrays
        ba_s, ca_s: smoothed spline functions (or nan_func if insufficient data)
        """
        import numpy as np
        from scipy.interpolate import splrep, splev
        import scipy

        # Remove rows where either ba or ca is NaN
        mask = ~np.isnan(ba) & ~np.isnan(ca)
        rbins_filtered = rbins[mask]
        ba_filtered = ba[mask]
        ca_filtered = ca[mask]

        # Check if we have enough points for any meaningful spline
        min_points = max(k + 1, 3)  # Need at least k+1 points for degree k spline
        if len(rbins_filtered) < min_points:
            return rbins_filtered, ba_filtered, ca_filtered, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

        # Calculate smoothing parameter
        s = s_factor * len(rbins_filtered)

        # Create initial splines with bounded domain
        xb, xe = rbins_filtered[0], rbins_filtered[-1]
        ba_s_tck = scipy.interpolate.splrep(rbins_filtered, ba_filtered, k=k, s=s, xb=xb, xe=xe)
        ca_s_tck = scipy.interpolate.splrep(rbins_filtered, ca_filtered, k=k, s=s, xb=xb, xe=xe)

        # Calculate residuals and remove outliers
        ba_residuals = ba_filtered - splev(rbins_filtered, ba_s_tck)
        ca_residuals = ca_filtered - splev(rbins_filtered, ca_s_tck)

        # Calculate the standard deviation of the residuals
        ba_std = np.std(ba_residuals)
        ca_std = np.std(ca_residuals)

        # Remove outliers
        d = 5
        mask = np.abs(ba_residuals) < d * ba_std
        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]

        # Check again after outlier removal
        if len(rbins_filtered) < min_points:
            return rbins_filtered, ba_filtered, ca_filtered, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

        mask = np.abs(ca_residuals[mask]) < d * ca_std
        rbins_filtered = rbins_filtered[mask]
        ba_filtered = ba_filtered[mask]
        ca_filtered = ca_filtered[mask]

        # Check again after second outlier removal
        if len(rbins_filtered) < min_points:
            return rbins_filtered, ba_filtered, ca_filtered, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

        # Recreate splines with bounded domain
        xb, xe = rbins_filtered[0], rbins_filtered[-1]
        ba_s_tck = scipy.interpolate.splrep(rbins_filtered, ba_filtered, k=k, s=s, xb=xb, xe=xe)
        ca_s_tck = scipy.interpolate.splrep(rbins_filtered, ca_filtered, k=k, s=s, xb=xb, xe=xe)

        # Remove large gaps
        diff = np.diff(rbins_filtered, prepend=0)
        mask = diff > 1
        rbins_filtered = rbins_filtered[~mask]
        ba_filtered = ba_filtered[~mask]
        ca_filtered = ca_filtered[~mask]

        # Final check after gap removal
        if len(rbins_filtered) < min_points:
            return rbins_filtered, ba_filtered, ca_filtered, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

        # Final spline creation with bounded domain
        xb, xe = rbins_filtered[0], rbins_filtered[-1]
        ba_s_tck = scipy.interpolate.splrep(rbins_filtered, ba_filtered, k=k, s=s, xb=xb, xe=xe)
        ca_s_tck = scipy.interpolate.splrep(rbins_filtered, ca_filtered, k=k, s=s, xb=xb, xe=xe)

        # Create callable functions with bounds checking
        def ba_s_func(x):
            x = np.asarray(x)
            # Clip to bounds to avoid extrapolation issues
            x_clipped = np.clip(x, xb, xe)
            return splev(x_clipped, ba_s_tck)

        def ca_s_func(x):
            x = np.asarray(x)
            # Clip to bounds to avoid extrapolation issues
            x_clipped = np.clip(x, xb, xe)
            return splev(x_clipped, ca_s_tck)

        return rbins_filtered, ba_filtered, ca_filtered, ba_s_func, ca_s_func

    def calculate(self, halo, existing_properties):
        rbins_s = existing_properties['rbins_s']
        rbins_d = existing_properties['rbins_d']
        rbins_g = existing_properties['rbins_g']

        # Process stellar component first to get the radial range
        rbins_s, ba_s, ca_s, ba_s_spline, ca_s_spline = self.smooth_shape(rbins_s, existing_properties['ba_s'],
                                                                          existing_properties['ca_s'])

        # Get the maximum radial bin from the stellar component
        if len(rbins_s) > 0:
            max_rbin_s = np.max(rbins_s) + 1
        else:
            # If no stellar data, use a very small value to filter everything
            max_rbin_s = -1

        # Filter dark matter component to stellar range
        mask_d = rbins_d <= max_rbin_s
        rbins_d_filtered = rbins_d[mask_d]
        ba_d_filtered = existing_properties['ba_d'][mask_d]
        ca_d_filtered = existing_properties['ca_d'][mask_d]

        # Filter gas component to stellar range
        mask_g = rbins_g <= max_rbin_s
        rbins_g_filtered = rbins_g[mask_g]
        ba_g_filtered = existing_properties['ba_g'][mask_g]
        ca_g_filtered = existing_properties['ca_g'][mask_g]

        # Process dark matter component with filtered data
        rbins_d, ba_d, ca_d, ba_d_spline, ca_d_spline = self.smooth_shape(rbins_d_filtered, ba_d_filtered,
                                                                          ca_d_filtered)

        # Process gas component with filtered data
        rbins_g, ba_g, ca_g, ba_g_spline, ca_g_spline = self.smooth_shape(rbins_g_filtered, ba_g_filtered,
                                                                          ca_g_filtered)

        return rbins_s, ba_s_spline, ca_s_spline, rbins_d, ba_d_spline, ca_d_spline, rbins_g, ba_g_spline, ca_g_spline
    

class DynamicalMass(PynbodyPropertyCalculation):
    """
    Calculate the dynamical mass of a halo using the circular velocity profile at the half-light radius.
    :returns: Mdyn in Msol
    """
    names = 'Mdyn'

    # def requires_property(self):
    #     return ['Rhalf']

    def calculate(self, halo,existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        #Rhalf = existing_properties['Rhalf']
        Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        prof = pynbody.analysis.profile.Profile(halo,type='lin',min=.25,max=5*Rhalf,ndim=2,nbins=int((5*Rhalf)/0.1))
        indeff = np.argmin(np.abs(prof['rbins']-Rhalf))
        veff = prof['v_circ'][indeff]
        Mdyn=  ( (Rhalf*1e3)*veff**2)/(4.3009172706e-3)

        return Mdyn

class SersicFit(PynbodyPropertyCalculation):
    names = ['reff', 'rhalf']

    @staticmethod
    def sersic(r, mueff, reff, n):
        return mueff + 2.5 * (0.868 * n - 0.142) * ((r / reff) ** (1. / n) - 1)

    def calculate(self, halo, existing_properties):
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        # Get the surface density profile
        try:
            Rhalf = pynbody.analysis.luminosity.half_light_r(halo)
        except:
            Rhalf = np.nan
        try:
            prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=.25,
                                                    max=5 * Rhalf, ndim=2,
                                                    nbins=int(
                                                        (5 * Rhalf) / 0.1))
            vband = prof['sb,V']
            smooth = np.nanmean(
                np.pad(vband.astype(float), (0, 3 - vband.size % 3),
                       mode='constant', constant_values=np.nan).reshape(
                    -1, 3), axis=1)
            x = np.arange(len(smooth)) * 0.3 + 0.15
            x[0] = .05
            if True in np.isnan(smooth):
                x = np.delete(x, np.where(np.isnan(smooth) == True))
                y = np.delete(smooth, np.where(np.isnan(smooth) == True))
            else:
                y = smooth
            r0 = x[int(len(x) / 2)]
            m0 = np.mean(y[:3])
            par, ign = curve_fit(self.sersic, x, y, p0=(m0, r0, 1),
                                 bounds=([10, 0, 0.5], [40, 100, 16.5]))
            reff = pynbody.array.SimArray(par[1], 'kpc')
        except:
            print("Sersic fit failed")
            print(traceback.format_exc())
            # set reff to value of later halo
            try:
                reff = halo.calculate('later(1).reff')
            except:
                reff = np.nan
        return reff, Rhalf
        # except:
        #     print("Sersic fit failed")
        #     print(traceback.format_exc())
        #     return np.nan


class dynamical_time(PynbodyPropertyCalculation):
    names = ['tdyn']

    def requires_property(self):
        return ['rbins']

    def calculate(self, halo, existing_properties):
        pynbody.analysis.angmom.faceon(halo)
        rbins = existing_properties['rbins']
        prof = pynbody.analysis.profile.Profile(halo, bins=rbins, ndim=2)
        mass_enc = prof['mass_enc']
        dyntime = (rbins ** 3 / (2 * pynbody.units.G * mass_enc)) ** (1 / 2)
        return dyntime


class BaryonicFractionReff(PynbodyPropertyCalculation):
    names = ['Mvir_within_reff', 'Mstar_within_reff', 'Mgas_within_reff',
             'Mb_mvir_within_reff']

    def requires_property(self):
        return ['reff', 'max_radius']

    @staticmethod
    def mass_properties_within_r(halo, r):
        # halo should be in physcial units, but just in case
        halo.physical_units()

        sphere_filter = pynbody.filt.Sphere(r)
        sphere = halo[sphere_filter]

        m_tot = (sphere['mass'].sum().in_units('Msol'))
        m_gas = (sphere.gas['mass'].sum().in_units('Msol'))
        m_star = (sphere.star['mass'].sum().in_units('Msol'))
        m_dm = (sphere.dm['mass'].sum().in_units('Msol'))
        m_vir_within_r = m_gas + m_star + m_dm
        # assert that all of these values are positive, and not close to 0 they are stored as pynbody SimArrays in units of solar masses
        # assert that m_tot is the sum of the other masses within floating point error
        assert np.isclose(m_tot, m_vir_within_r,
                          rtol=1e-10), f"Total mass is {m_tot}, sum of components is {m_gas + m_star + m_dm}"
        Mb_within_r = m_gas + m_star
        mb_mvir_within_r = Mb_within_r / m_vir_within_r

        return m_vir_within_r, m_star, m_gas, mb_mvir_within_r

    def calculate(self, halo, existing_properties):
        reff = existing_properties['reff']
        Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff = self.mass_properties_within_r(
            halo, reff)
        return Mvir_within_reff, Mstar_within_reff, Mgas_within_reff, mb_mvir_within_reff


class BaryonicFractionVirial(PynbodyPropertyCalculation):
    names = ['Mvir', 'Mstar', 'Mgas', 'Mb_mvir']

    def calculate(self, halo, existing_properties):
        m_gas = halo.gas['mass'].sum().in_units('Msol').view(np.ndarray)
        m_star = halo.star['mass'].sum().in_units('Msol').view(np.ndarray)
        m_dm = halo.dm['mass'].sum().in_units('Msol').view(np.ndarray)
        m_vir = halo['mass'].sum().in_units('Msol').view(np.ndarray)
        try:
            Mb = m_gas + m_star
            mb_mvir = Mb / m_vir
        except ZeroDivisionError:
            mb_mvir = np.nan

        return m_vir, m_star, m_gas, mb_mvir


class StarFormationProfile(PynbodyPropertyCalculation):
    """
    Calculate star formation rate profile and edge radius for a halo.

    This class computes two key properties related to star formation in a halo:

    1. R_edge: The radius at which star formation effectively ceases, defined as the
       smallest radius where the normalized star formation rate (s_sfr) drops to
       zero or below. This represents the "edge" of active star formation.

    2. s_sfr_profile: The normalized star formation rate profile, calculated as the
       ratio of newly formed stellar mass to total stellar mass in radial bins.
       This profile shows how star formation efficiency varies with radius.

    The calculation:
    - Orients the halo face-on for consistent radial measurements
    - Identifies newly formed stars within a specified lookback time (default 100 Myr)
    - Creates radial profiles for both newly formed and total stellar mass
    - Computes the normalized star formation rate (s_sfr) as their ratio
    - Determines R_edge as the first radius where star formation ceases

    Attributes:
        lookback_time (float): Time period in Myr to define "newly formed" stars.
                              Default is 100 Myr.

    Example:
        # Use default 100 Myr lookback time
        calculator = StarFormationProfile()

        # Use custom 50 Myr lookback time
        calculator = StarFormationProfile(lookback_time=50.0)

        # Use custom 200 Myr lookback time
        calculator = StarFormationProfile(lookback_time=200.0)
    """
    names = ['r_edge','r1', 's_sfr_profile']
    lookback_time = 100.0  # Default lookback time in Myr

    def calculate(self, particle_data, existing_properties):
        """
        Calculate the star formation profile and edge radius.

        Returns:
            tuple: (R_edge, s_sfr_profile) where:
                - R_edge: Radius where star formation ceases (float)
                - s_sfr_profile: Normalized star formation rate vs radius (array)
        """
        halo = particle_data

        # Set physical units and orient face-on for consistent radial measurements
        halo.physical_units()
        pynbody.analysis.angmom.faceon(halo)
        Rhalf = pynbody.analysis.luminosity.half_light_r(halo)

        # Find newly formed stars in the specified lookback time
        newly_formed_stars = halo.s[halo.s['tform'] > (halo.s['tform'].max() - self.lookback_time * pynbody.units.Myr)]

        # Create profiles
        prof_sfr = pynbody.analysis.profile.Profile(newly_formed_stars, type='lin', min=Rhalf/3,
                                                    max=halo.s['r'].max(), ndim=2,
                                                    nbins=int((5 * halo.s['r'].max()) / 0.1))

        prof = pynbody.analysis.profile.Profile(halo.s, type='lin', min=Rhalf/3,
                                                max=halo.s['r'].max(), ndim=2,
                                                nbins=int((5 * halo.s['r'].max()) / 0.1))
        bin_area = prof._binsize.in_units('pc^2')

        density = prof['mass'].in_units('Msol') / bin_area
        
        R1 = np.min(prof['rbins'][density < 1])

        # Calculate s_sfr (star formation rate profile)
        s_sfr = prof_sfr['mass'] / prof['mass']

        # Calculate r_edge as the smallest radius where s_sfr <= 0
        # Handle case where no bins meet the criteria
        valid_bins = s_sfr <= 0
        if np.any(valid_bins):
            r_edge = np.min(prof_sfr['rbins'][valid_bins])
        else:
            r_edge = np.nan  # or some default value

        # Return values in the same order as names
        return r_edge,R1, s_sfr







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
    


# ── LivePropertyCalculation ──────────────────────────────────────────────────

class VBandEllipticalSersic(LivePropertyCalculation):
    """Fit a 2-D elliptical Sérsic profile to every stored V-band image.

    Reads from properties already written by VBandStarImages, so no
    simulation I/O is needed.

    Stored outputs (one value per orientation):
        sersic2d_reff_v   – effective (semi-major-axis) radius  [kpc]
        sersic2d_n_v      – Sérsic index
        sersic2d_q_v      – axis ratio  b/a
        sersic2d_theta_v  – position angle  [radians, CCW from x-axis]
        sersic2d_mueff_v  – surface brightness at Reff  [mag arcsec^-2]
    """

    names = [
        'sersic2d_reff_v',
        'sersic2d_n_v',
        'sersic2d_q_v',
        'sersic2d_theta_v',
        'sersic2d_Ieff_v',
    ]

    # ── image geometry ───────────────────────────────────────────────────────
    # Must match what VBandStarImages used:
    #   width      = 9 * Rhalf   (kpc)
    #   resolution = 1000 pixels
    _IMAGE_WIDTH_IN_RHALFS = 9
    _IMAGE_RESOLUTION_PX   = 1000

    def requires_property(self):
        return ['halo_images_v', 'Rhalf_v', 'profile_rbins_v', 'profile_sb_v', 'Rvir' ]

    def live_calculate(self, existing_properties):
        images = existing_properties['halo_images_v']
        Rhalf  = existing_properties['Rhalf_v']
        Rvir = existing_properties['Rvir']
        rmax = Rvir * 0.1
        rmin = 0.3
        sb_profiles = existing_properties['profile_sb_v']
        rbins = existing_properties['profile_rbins_v']


        images = np.asarray(images)          # shape: (N_orient, px, px)
        if images.ndim == 2:                 # single image edge-case
            images = images[np.newaxis]

        pixel_scale = (self._IMAGE_WIDTH_IN_RHALFS * float(Rhalf)
                       / self._IMAGE_RESOLUTION_PX)   # kpc / pixel

        fitter = SersicFitter()
        reff_list, n_list, q_list, theta_list, mueff_list = [], [], [], [], []

        for i, img in enumerate(images):
            p0_1d = None
            if sb_profiles is not None and rbins is not None:
                sb_arr = np.asarray(sb_profiles)
                if sb_arr.ndim == 2 and i < len(sb_arr):
                    # fit_1d already expects mag-space data and returns (mueff, reff, n)
                    p0_1d = fitter.fit_1d(sb_arr[i], pixel_scale)

            result = fitter.fit_2d_log(img, pixel_scale, p0_1d=p0_1d, rmax = rmax, rmin = rmin)
            if i in [0,3]:
                print(p0_1d)
                print(result)
            #(('Ieff', 'reff', 'n', 'x0', 'y0', 'theta', 'q'),
            reff_list.append(result['reff'])
            n_list.append(result['n'])
            q_list.append(result['q'])
            theta_list.append(result['theta'])
            mueff_list.append(result['mueff'])


        return (np.array(reff_list),
                np.array(n_list),
                np.array(q_list),
                np.array(theta_list),
                np.array(mueff_list))


class Rvir(LivePropertyCalculation):
    names = 'Rvir'
    # if Rvir is named something else, let us make a new entry
    def requires_property(self):
        return ['Rhalo']

    def live_calculate(self, existing_properties):
        return extract_single_value(existing_properties['Rhalo'])



class StellarProfileDiagnosis(LivePropertyCalculation):
    '''
    calculate sersic profile fit properties based on existing surface brightness profiles
    '''
    def __init__(self, simulation, band, sats=0, sblimit=28, type='hlr', smooth=0):
        '''
        :param band: which band of calculated sb profile to use (band_surface_brightness)
        :param sats: if 1, search for nearby halos and limit the maximum radius to fit over accordingly
        :param type: hlr = limit max radius to 5 times half light radius. sb = limit based on sblimit
        :param sblimit: lowest surface brightness to consider (will cut off once sb reaches below this limit, only if type='sb')
        :param smooth: smooth over this number of bins when fitting. Useful when resolution limit is courser than spatial bin size
        '''
        super(StellarProfileDiagnosis, self).__init__(simulation)
        self.band = band
        self.type=type
        self.sblimit=sblimit
        self.sats=sats
        self.smooth=smooth
        self.requires_particle_data = False
        if smooth==0:
            self.smooth=1

    names=["half_light","sersic_m0", "sersic_n", "sersic_r0"]

    def requires_property(self):
        return list(StellarProfileFaceOn.names) + ['max_radius', 'shrink_center']

    def sersic_surface_brightness(self,r, mueff, reff, n):
        # I(R) = m0 + 2.5*b_n/ln(10) ( (r/reff)^(1/n)-1 )
        #b_n taken based on solution to gamma function (Capaccioli 1989) for n < 10.
        return mueff + 2.5*(0.868*n-0.142)*((r/reff)**(1./n) - 1)

    def fit_sersic(self, r, surface_brightness, return_cov=False):
        s0_guess = np.mean(surface_brightness[:3])
        s0_range = [10,40]
        n_range = [0.5,16.5]
        r0_range=[0,100]

        r0_guess = min(r[int(len(r)/2)], 50)
        n_guess = 1.0

        sigma = 10**(0.6*(surface_brightness-20))/r
        sigma = None

        popt, pcov = scipy.optimize.curve_fit(self.sersic_surface_brightness,r,surface_brightness,
                                          bounds=np.array((s0_range, r0_range, n_range)).T,
                                          sigma=sigma,
                                          p0=(s0_guess, r0_guess, n_guess))

        if return_cov:
            return popt, pcov
        else:
            return popt

    def live_calculate(self, halo, *args):
        delta_r = self.get_simulation_property("approx_resolution_kpc", 0.1)
        r0 = delta_r/2
        sb_property = self.band+"_surface_brightness"
        if sb_property in halo.keys():
            surface_brightness = halo[sb_property]
        else:
            try:
                surface_brightness = halo.calculate(sb_property+"()")
            except NoResultsError:
                return null_result(self)
        if len(surface_brightness)<self.smooth*2:
            return null_result(self)
        if surface_brightness.max()==0:
            return null_result(self)
        r = np.arange(len(surface_brightness))*delta_r + delta_r/2.
        nbins = len(surface_brightness)
        if self.smooth > 1:
            surface_brightness_new = np.nanmean(np.pad(surface_brightness[self.smooth:].astype(float),
                                                (0,3-surface_brightness[self.smooth:].size%self.smooth),mode='constant',
                                                constant_values=np.NaN).reshape(-1,self.smooth),axis=1)
            surface_brightness = np.insert(surface_brightness_new,0,np.mean(surface_brightness[:self.smooth]))
            r = np.arange(len(surface_brightness))*(delta_r*self.smooth) + delta_r*self.smooth/2.
            r[0] = r0


        flux_density = 10**(surface_brightness/-2.5)
        flux_density[flux_density!=flux_density]=0
        cumu_flux_density = (r * flux_density).cumsum()
        cumu_flux_density/=cumu_flux_density[-1]

        try:
            half_light_i = np.where(cumu_flux_density>0.5)[0][0]
            half_light = r0+delta_r * half_light_i * self.smooth
        except:
            half_light = None
        if self.type=='hlr':
            if half_light is None:
                return null_result(self)
            maxrad = half_light*5

        if self.type=='sb':
            maxradi = np.where(surface_brightness>self.sblimit)[0]
            if len(maxradi) == 0:
                maxrad = halo['max_radius']
            else:
                maxrad = r[maxradi[0]]

        if self.sats:
            cen, mvir = halo.timestep.gather_property('shrink_center', 'finder_mass')
            darr = halo['shrink_center'] - cen
            D = np.sqrt(np.sum(darr**2,axis=1))
            dmin = D[(D>0)].min()
            if dmin < maxrad:
                maxrad = dmin

        usefit = np.where(r<=maxrad)[0]
        r_fit = r[usefit]
        sb_fit = surface_brightness[usefit]

        mask_not_nan = sb_fit==sb_fit
        r_fit = r_fit[mask_not_nan]
        sb_fit = sb_fit[mask_not_nan]
        if len(r_fit)<2:
            return null_result(self)
        else:
            try:
                m0, r0, n = self.fit_sersic(r_fit, sb_fit)
            except:
                m0 = None
                n = None
                r0 = None
            return half_light, m0, n, r0