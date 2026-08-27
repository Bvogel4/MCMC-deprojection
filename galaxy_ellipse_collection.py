#this contains a bunch of helper functions for MCMC_sims to run and plot MCMC codes from shape_inference and shape_plotting.

import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import pickle
import re
from glob import glob

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import is_color_like
from joblib import Memory

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
    random_viewing_angles
    
)


from shape_plotting import (
    n_params, unpack_params, sample_model_projections,
    plot_corner, plot_chain_evolution, plot_ellipsoid_shapes,
    plot_projected_distributions, plot_projected_distributions_with_model,
    plot_shape_distribution, plot_q_comparison,
)



def get_keys(simname, hid, orientation_coords, keys, cachedir='caches'):

    """
    Pull one or more keys out of a cached face-on/oriented image dict.

    simname            : e.g. 'r431.romulus25.3072g1HsbBH'
    hid                : halo id
    orientation_coords : (theta, phi)
    keys               : str or list of str
    """
    t, p = orientation_coords
    pattern = (f'{cachedir}/vband_cache_{simname}_n100/'
               f'VBandStarImages_halo{hid}/*_t{t:+f}_p{p:+f}.pkl')

    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f'No cache file matching {pattern}')

    with open(matches[0], 'rb') as f:
        d = pickle.load(f)

    if isinstance(keys, str):
        return d[keys]
    #print(d.keys())
    return [d[k] for k in keys]



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

        values = np.array([np.atleast_1d(v) for v in halo_data.values()], dtype=float)

        # Drop samples containing NaNs (e.g. failed ellipticity fits at some
        # viewing angles) so they don't poison the interpolant.
        good = ~np.isnan(values).any(axis=1)
        self.n_dropped = int((~good).sum())
        if self.n_dropped:
            thetas, phis, values = thetas[good], phis[good], values[good]
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
    if attainable(params):
        return params
    else:
        raise ValueError(f"cov_fit(): cov_fit failed for {params}")



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

class GalaxyEllipseCollection:
    """
    A collection of galaxy ellipse data with methods to generate observed axis ratio distributions
    and run shape inference using MCMC.
    """

    def __init__(self,reff_index=1,run_covariances=False):
        """Initialize an empty collection of galaxy ellipses."""
        self.halos = {}  # Dictionary to store halo data {(sim_name, halo_id): {data}}
        self.interpolators = {}  # Dictionary to store interpolation functions
        self.reff_interpolators = {}
        self.n_steps = 30000
        self.halo_data = {}  # Store raw halo data for reference
        self.reff_index = reff_index
        self.run_covariances = run_covariances




    def add_halo(self, sim_name, halo_id, halo_data,
                 interpolation_method='linear', coordinate_system='angles'):
        halo_key = (sim_name, halo_id)
        self.halo_data[halo_key] = halo_data  # Store raw data for reference
        self.halos[halo_key] = halo_data
        data = halo_data.copy()
        if 'a_s' in data.keys():
            del data['ba_s']
            del data['ca_s']
            del data['a_s']

        interpolator = SphereInterpolator(data)
        self.interpolators[halo_key] = interpolator
        reff_dict = self.create_reff_dict(data,sim_name,halo_id)
        self.reff_interpolators[halo_key] = SphereInterpolator(reff_dict)

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
        self.halos[halo_key] = other_collection.halos[halo_key]
        self.interpolators[halo_key] = other_collection.interpolators[halo_key]
        self.halo_data[halo_key] = other_collection.halos[halo_key]
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
        if halo_key not in self.halos:
            raise KeyError(f"Halo {halo_id} from simulation {sim_name} not found in collection")

        interpolator = self.interpolators[halo_key]
        reff_interpolator = self.reff_interpolators[halo_key]
        phi,theta = random_viewing_angles(n_angles)
        e = interpolator(theta,phi)[:,self.reff_index]
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
        if not self.halos:
            raise ValueError("No halos in collection")

        q_values = []
        reff_values = []
        i = 0
        j = 0
        
        for halo_key in self.halos.keys():
            interpolator = self.interpolators[halo_key]
            reff_interpolator = self.reff_interpolators[halo_key]
            # Generate random viewing angles for this halo
            phi, theta = random_viewing_angles(angles_per_halo)
            # Get q values for each angle
            try:
                e = interpolator(theta,phi)[:,self.reff_index]
                q = 1 - e
                q_values.extend(q)
                reff_values.extend(reff_interpolator(theta,phi))
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

        return np.array(q_values), np.log(np.array(reff_values)) + 2

    def generate_q_distribution_all_halos_sideon(self):
        if not self.halos:
            raise ValueError("No halos in collection")

        q_values = []
        i = 0
        j = 0
        sideon_orientations = [f'x090y{i:03d}' for i in np.arange(0, 330, 30)]
        for halo_key in self.halos.keys():
            halo_data = self.halo_data[halo_key]
            e = np.nanmax([halo_data[orientation][self.reff_index] for orientation in sideon_orientations])
            q = 1 - e
            i = i + 1
            q_values.append(q)



        print(f'Generated q values from {i} halos, failed for {j} halos.')

        return np.array(q_values)
    
    def run_inference_all_halos(self, n_angles_per_halo=100, weighted=False,
                                n_walkers=32, n_steps=3000, burn_in=500, n_cores=None,
                                output_prefix="all_halos", output_dir="results",
                                force_rerun=False,
                                label="All Halos Combined", color="blue"):
        """
        Run shape inference on a distribution from all halos.

        (docstring unchanged)
        """
        full_output_dir = Path(output_dir)
        full_output_dir.mkdir(parents=True, exist_ok=True)

        if self.run_covariances:
            cov_output_dir = full_output_dir.with_name(full_output_dir.name + "_covariance")
            cov_output_dir.mkdir(parents=True, exist_ok=True)
            cov_prefix = f"{output_prefix}"

        def _save_results(stage_dir, stage_prefix, samples, max_prob_params, chain,
                          q_obs, reff_values, sampler):
            results = {
                'samples': samples,
                'max_prob_params': max_prob_params,
                'full_chain': chain,
                'q_obs': q_obs,
                'a_obs': reff_values,
                'sampler': sampler,
            }
            with open(stage_dir / f"{stage_prefix}_results.pkl", 'wb') as f:
                pickle.dump(results, f)

        # Get true_params by averaging over all halos
        ba_s = []
        ca_s = []
        a_s = []
        for halo_key in self.halos.keys():
            halo_data = self.halos[halo_key]
            ba_s.append(halo_data['ba_s'])
            ca_s.append(halo_data['ca_s'])
            a_s.append(halo_data['a_s'])
        ba_s = np.array(ba_s)
        ca_s = np.array(ca_s)
        a_s = np.log(np.array(a_s)) + 2
        #print(ba_s, ca_s, a_s)
        if self.run_covariances:
            theta_data = cov_fit(a_s, ba_s, ca_s)  # intrinsic shapes from the sim
            from shape_inference import attainable
            print(theta_data, attainable(theta_data))

        existing_results = None
        if not force_rerun:
            existing_results = check_existing_results(str(full_output_dir), output_prefix)
            if existing_results:
                samples, max_prob_params, chain, q_obs, reff_values, sampler = existing_results
                print("Using existing inference results for all halos combined")

        if force_rerun or not existing_results:
            start_time = time.time()
            q_obs, reff_values = self.generate_q_distribution_all_halos(n_angles_per_halo)
            np.save(full_output_dir / f"{output_prefix}_q_obs.npy", q_obs)

            samples, max_prob_params, sampler = infer_intrinsic_shape(
                q_obs,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                n_cores=n_cores,
                output_prefix=output_prefix,
                output_dir=str(full_output_dir),
            )
            end_time = time.time()
            print(f"Inference completed in {end_time - start_time:.2f} seconds")

            chain = sampler.get_chain()
            _save_results(full_output_dir, output_prefix, samples, max_prob_params,
                          chain, q_obs, reff_values, sampler)

            try:
                tau = sampler.get_autocorr_time(quiet=True)
                print(f"Autocorrelation times: {tau}")
                print(f"Number of effective samples: {n_steps / np.max(tau)}")
            except Exception as e:
                print(f"Could not compute autocorrelation time (might need more samples): {e}")

        if self.run_covariances:
            existing_results_cov = None
            if not force_rerun:
                existing_results_cov = check_existing_results(str(cov_output_dir), cov_prefix)
                if existing_results_cov:
                    samples_cov, max_prob_params_cov, chain_cov, q_obs, reff_values, sampler = existing_results_cov
                    print("Using existing inference results for all halos combined (covariance)")
                    print(f'{cov_output_dir}\n{cov_prefix}')

            if force_rerun or not existing_results_cov:
                samples_cov, max_prob_params_cov, sampler = infer_intrinsic_shape_multivariate(
                    q_obs, max_prob_params, reff_values,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    burn_in=burn_in,
                    n_cores=n_cores,
                    output_prefix=cov_prefix,
                    output_dir=str(cov_output_dir),
                )
                chain_cov = sampler.get_chain()
                print(max_prob_params_cov)
                _save_results(cov_output_dir, cov_prefix, samples_cov, max_prob_params_cov,
                              chain_cov, q_obs, reff_values, sampler)

        ba_s_sigma = np.std(ba_s)
        ca_s_sigma = np.std(ca_s)

        mu_ba_s = np.mean(ba_s)
        mu_ca_s = np.mean(ca_s)

        true_params = np.array([mu_ba_s, mu_ca_s, ba_s_sigma, ca_s_sigma])

        #q_true, _ = sample_model_projections(true_params, len(q_obs))
        q_true = []
        a_true = []
        for i in range(len(ba_s)):
            phi,theta=random_viewing_angles(n_angles_per_halo)
            alpha,beta = projected_semi_axes(phi,theta,ba_s[i],ca_s[i])
            q = beta/alpha
            a = np.log(a_s[i])*np.sqrt(alpha*beta) + 2
            # q = project_ellipticity_xu(theta,phi)
            q_true.extend(q)
            a_true.extend(a)
        q_model, _ = sample_model_projections(max_prob_params, len(q_obs))

        recovery_results = assess_parameter_recovery(
            max_prob_params, true_params, q_obs=q_obs, q_model=q_model, q_true=q_true
        )
        with open(full_output_dir / f"{output_prefix}_recovery.txt", 'w') as f:
            f.write(recovery_results['output_text'])

        if not is_color_like(color):
            color = None

        self.plot_results(
            samples=samples,
            max_prob_params=max_prob_params,
            true_params=true_params,
            q_obs=q_obs,
            q_model=q_model,
            q_true=q_true,
            chain=chain,
            burn_in=burn_in,
            label=label,
            color=color,
            output_prefix=output_prefix,
            output_dir=str(full_output_dir),
            intrinsic_data=(a_s, ba_s, ca_s),
        )

        if self.run_covariances:
            mu_A, mu_b, mu_c, sigma_A, sigma_b, sigma_c, sigma_ac = cov_fit(a_s, ba_s, ca_s)
            true_params_cov = np.array([mu_A, mu_b, mu_c, sigma_A, sigma_b, sigma_c, sigma_ac])

            #q_true_cov, a_true_cov = sample_model_projections(true_params_cov, 10000)
            q_true_cov =  q_model
            a_true_cov = a_true

            print(max_prob_params_cov)
            q_model_cov, a_model_cov = sample_model_projections(max_prob_params_cov, 10000)

            self.plot_results(
                samples=samples_cov,
                max_prob_params=max_prob_params_cov,
                true_params=true_params_cov,
                q_obs=q_obs,
                q_model=q_model_cov,
                q_true=q_true_cov,
                chain=chain_cov,  # the covariance chain, not the 4-param one
                burn_in=burn_in,
                label=label,
                color=color,
                output_prefix=output_prefix + '_covariance',
                output_dir=str(full_output_dir) + '_covariance',
                # 7-parameter extras (all optional)
                a_obs=reff_values,
                intrinsic_data=(a_s, ba_s, ca_s),
            )

        
    def plot_results(self, samples, max_prob_params, q_obs, q_model=None, q_true=None,
                     true_params=None, chain=None, burn_in=500, label="Model",
                     color="blue", output_prefix=None, output_dir="results",
                     a_obs=None, intrinsic_data=None, bin_width=0.04):
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
        """
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
            #title=f"Projected Axis Ratios: Observed vs Model for {label}",
            bin_width=bin_width,
        )
        plt.close(fig)

        fig = plot_projected_distributions(
            [q_obs], [label], [color],
            output_file=path("projections_hist"),
            #title=f"Projected Axis Ratios for {label}",
            bin_width=bin_width, kde=False,
        )
        plt.close(fig)

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
        # Pass colour as a LIST. A bare string used to be zipped character-wise.
        fig = plot_ellipsoid_shapes(
            [samples], [max_prob_params], [true_params], [label], [color],
            output_file=path("ellipsoid_shapes"),
            #title=f"Intrinsic Shapes: {label}",
            focus_on_max_prob=True,
            intrinsic_data=intrinsic_data,
        )
        plt.close(fig)

        fig = plot_ellipsoid_shapes(
            [samples], [max_prob_params], [true_params], [label], ["blue"],
            output_file=path("ellipsoid_shapes_all"),
            #title=f"Intrinsic Shapes: {label}",
            focus_on_max_prob=False, show_samples=True, show_ellipses=False,
        )
        plt.close(fig)

        # --- 7-parameter extras ----------------------------------------------
        if ndim == 7:
            a_data = b_data = c_data = None
            if intrinsic_data is not None:
                a_data, b_data, c_data = intrinsic_data

            # Intrinsic A / B/A / C/A marginals, the shape plane, and the
            # A - C/A covariance panel where sigma_ac shows up.
            fig = plot_shape_distribution(
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

            p = unpack_params(max_prob_params)
            rho = p['sigma_ac'] / (p['sigma_A'] * p['sigma_C'])
            print(f"  sigma_ac = {p['sigma_ac']:.4f}  ->  rho_ac = {rho:+.3f} "
                  f"(bound |sigma_ac| < {p['sigma_A'] * p['sigma_C']:.4f})")
            if abs(rho) > 0.9:
                print("  WARNING: correlation is close to degenerate; the rejection "
                      "sampler will have a low acceptance rate.")

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
        halo_data = self.halos[(sim_name, halo_id)]
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

        return samples, max_prob_params, sampler, q_obs

    def get_all_halo_keys(self):
        """Return a list of all (sim_name, halo_id) keys in the collection."""
        return list(self.halos.keys())

    def get_halo_count(self):
        """Return the number of halos in the collection."""
        return len(self.halos)


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
