#this contains a bunch of helper functions for MCMC_sims to run and plot MCMC codes from shape_inference and shape_plotting.

import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import pickle
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf
from matplotlib.colors import is_color_like
from joblib import Memory

memory = Memory(location='.cache/interpolators', verbose=0)

from shape_inference import (
    generate_projections,
    generate_ellipsoid_distribution,
    generate_projections_from_distribution,
    generate_model_projections,
    infer_intrinsic_shape,
    load_results
)

from shape_plotting import (
    plot_corner,
    plot_ellipsoid_shapes,
    plot_projected_distributions,
    plot_projected_distributions_with_model,
    plot_chain_evolution,
    plot_comparison_grid,
    plot_statistics
)


@memory.cache
def _build_interpolators_cached(sim_name, halo_id, halo_data, reff_multipliers,
                                 interpolation_method, coordinate_system):
    def parse_orientation(orientation):
        try:
            x_angle = int(orientation[1:4])
            y_angle = int(orientation[5:8])
            return x_angle, y_angle
        except (ValueError, IndexError):
            print(f"Invalid orientation format: {orientation}")
            return None, None

    original_points = []
    original_values = {i: [] for i in range(len(reff_multipliers))}

    for orientation, ellipticities in halo_data.items():  # ← was halo_ellipses
        if not (orientation.startswith('x') and 'y' in orientation):
            continue
        x_angle, y_angle = parse_orientation(orientation)  # ← now defined above
        original_points.append([x_angle, y_angle])
        for i, eps in enumerate(ellipticities):
            if i < len(reff_multipliers):
                original_values[i].append(float(eps))

    extended_points = original_points.copy()
    extended_values = {i: original_values[i].copy() for i in range(len(reff_multipliers))}

    if coordinate_system == 'angles':
        for idx, point in enumerate(original_points):
            x_angle, y_angle = point
            if x_angle == 0:
                extended_points.append([180, y_angle])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif x_angle == 180:
                extended_points.append([0, y_angle])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            if y_angle == 0:
                extended_points.append([x_angle, 360])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif y_angle in (360, 359):
                extended_points.append([x_angle, 0])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            if (x_angle == 0 and y_angle == 0):
                extended_points.append([180, 360])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 180 and y_angle == 0):
                extended_points.append([0, 360])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 0 and y_angle in (360, 359)):
                extended_points.append([180, 0])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 180 and y_angle in (360, 359)):
                extended_points.append([0, 0])
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])

    extended_points_filtered = {}
    for i in range(len(reff_multipliers)):
        if not extended_values[i]:
            continue
        points_array = np.array(extended_points)
        values_array = np.array(extended_values[i])
        valid_mask = ~np.isnan(values_array)
        points_array = points_array[valid_mask]
        values_array = values_array[valid_mask]
        if len(values_array) > 0:
            extended_points_filtered[i] = points_array
            extended_values[i] = values_array
        else:
            extended_values[i] = []

    interpolators = {}
    fallback_interpolators = {}

    for i in range(len(reff_multipliers)):
        if i not in extended_points_filtered or len(extended_values[i]) == 0:
            continue
        points_array = extended_points_filtered[i]
        values_array = np.array(extended_values[i])
        assert len(points_array) == len(values_array)

        if interpolation_method == 'rbf':
            if coordinate_system == 'vectors':
                interpolators[i] = Rbf(points_array[:, 0], points_array[:, 1],
                                       points_array[:, 2], values_array, function='multiquadric')
            else:
                interpolators[i] = Rbf(points_array[:, 0], points_array[:, 1],
                                       values_array, function='multiquadric')
        elif interpolation_method == 'nearest':
            interpolators[i] = NearestNDInterpolator(points_array, values_array)
        else:
            interpolators[i] = LinearNDInterpolator(points_array, values_array)
            fallback_interpolators[i] = NearestNDInterpolator(points_array, values_array)

    return interpolators, fallback_interpolators, list(reff_multipliers)


class HaloInterpolator:
    def __init__(self, sim_name, halo_id, halo_data, reff_multipliers,
                 interpolation_method='linear', coordinate_system='angles'):
        self._interpolators, self._fallback_interpolators, self.reff_multipliers = \
            _build_interpolators_cached(
                sim_name, halo_id, halo_data,
                tuple(reff_multipliers), interpolation_method, coordinate_system
            )
        self.interpolation_method = interpolation_method
        self.coordinate_system = coordinate_system

    @staticmethod
    def angles_to_vector(x_angle, y_angle):
        x_rad = np.radians(x_angle)
        y_rad = np.radians(y_angle)
        vx = np.sin(y_rad)
        vy = 0
        vz = -np.cos(y_rad)
        new_vy = vy * np.cos(x_rad) - vz * np.sin(x_rad)
        new_vz = vy * np.sin(x_rad) + vz * np.cos(x_rad)
        return [vx, new_vy, new_vz]

    def __call__(self, x_angle, y_angle, reff_index=0):
        if x_angle > 180:
            x_angle = x_angle % 180
        if y_angle > 360:
            y_angle = y_angle % 360

        if reff_index not in self._interpolators:  # ← self.
            raise ValueError(f"No data for reff_index {reff_index} "
                             f"(multiplier={self.reff_multipliers[reff_index]})")

        if self.interpolation_method == 'rbf':  # ← self.
            if self.coordinate_system == 'vectors':  # ← self.
                vx, vy, vz = self.angles_to_vector(x_angle, y_angle)  # ← self.
                return float(self._interpolators[reff_index](vx, vy, vz))  # ← self.
            else:
                return float(self._interpolators[reff_index](x_angle, y_angle))  # ← self.
        else:
            if self.coordinate_system == 'vectors':  # ← self.
                point = self.angles_to_vector(x_angle, y_angle)  # ← self.
            else:
                point = [x_angle, y_angle]

            result = self._interpolators[reff_index](point)  # ← self.

            if self.interpolation_method == 'linear' and np.isnan(result):  # ← self.
                result = self._fallback_interpolators[reff_index](point)  # ← self.

            return float(result)




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


def random_viewing_angles(n):
    """
    Generate n random viewing angles uniformly distributed on a sphere.

    This function is already vectorized.

    Parameters:
        n (int): Number of viewing angles to generate

    Returns:
        tuple: (X, Y) rotation angles in degrees for n viewing positions
    """
    phi = np.random.uniform(0, 2 * np.pi, n)
    nu = np.random.uniform(0, 1, n)
    theta = np.arccos(2 * nu - 1)
    X, Y = spherical_to_rotation_angles(theta, phi)
    return X, Y




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
        raise ValueError(f"Halo ID {halo_id} not found")

    return ellipse_dict[sim_name][halo_id]


# Add the check_existing_results function from your test code
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
            if all(k in results for k in ['samples', 'max_prob_params', 'full_chain', 'q_obs', 'sampler']):
                print("Loaded existing MCMC results.")
                return results['samples'], results['max_prob_params'], results['full_chain'], results['q_obs'], results['sampler']
            else:
                print("Existing results incomplete. Will rerun analysis.")
                return None

        except Exception as e:
            print(f"Error loading existing results: {e}")
            return None

    return None


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



class GalaxyEllipseCollection:
    """
    A collection of galaxy ellipse data with methods to generate observed axis ratio distributions
    and run shape inference using MCMC.
    """

    def __init__(self):
        """Initialize an empty collection of galaxy ellipses."""
        self.halos = {}  # Dictionary to store halo data {(sim_name, halo_id): {data}}
        self.interpolators = {}  # Dictionary to store interpolation functions
        self.reff_multipliers = {}  # Dictionary to store effective radius multipliers
        self.n_steps = 30000
        self.halo_data = {}  # Store raw halo data for reference

    def add_halo(self, sim_name, halo_id, halo_data, reff_multipliers=None,
                 interpolation_method='linear', coordinate_system='angles'):
        halo_key = (sim_name, halo_id)
        self.halo_data[halo_key] = halo_data  # Store raw data for reference
        self.halos[halo_key] = halo_data

        interpolator = HaloInterpolator(
            sim_name, halo_id, halo_data,
            reff_multipliers, interpolation_method, coordinate_system
        )
        self.interpolators[halo_key] = interpolator
        self.reff_multipliers[halo_key] = interpolator.reff_multipliers

    def copy_halo_from(self, other_collection, sim_name, halo_id):
        """Copy a halo and its prebuilt interpolator from another collection."""
        halo_key = (sim_name, halo_id)
        self.halos[halo_key] = other_collection.halos[halo_key]
        self.interpolators[halo_key] = other_collection.interpolators[halo_key]
        self.reff_multipliers[halo_key] = other_collection.reff_multipliers[halo_key]
        self.halo_data[halo_key] = other_collection.halos[halo_key]

    def generate_q_distribution_single_halo(self, sim_name, halo_id, n_angles, reff_index=0):
        """
        Generate q (axis ratio) distribution for a single halo with random viewing angles.

        Parameters:
            sim_name (str): Simulation name
            halo_id (int/str): Halo identifier
            n_angles (int): Number of random viewing angles
            reff_index (int): Index into reff_multipliers to use

        Returns:
            array: q values (axis ratios) for the specified halo
        """
        halo_key = (sim_name, halo_id)
        if halo_key not in self.halos:
            raise KeyError(f"Halo {halo_id} from simulation {sim_name} not found in collection")

        interpolator = self.interpolators[halo_key]

        # # Generate random viewing angles


        x_angles, y_angles = random_viewing_angles(n_angles)

        #print(f"phi: {x_angles[0:15]}\ntheta: {y_angles[0:15]}")

        # Get q values for each angle
        e = np.array([interpolator(x, y, reff_index) for x, y in zip(x_angles, y_angles)])
        q_values = 1 - e

        return q_values

    def generate_q_distribution_all_halos(self, angles_per_halo, reff_index=0):
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


        i = 0
        j = 0
        for halo_key in self.halos.keys():
            interpolator = self.interpolators[halo_key]

            # Generate random viewing angles for this halo
            x_angles, y_angles = random_viewing_angles(angles_per_halo)

            # Get q values for each angle
            try:
                e = np.array([interpolator(x, y, reff_index) for x, y in zip(x_angles, y_angles)])
                q = 1 - e
                q_values.extend(q)
                i = i + 1
            except Exception as ex:
                print(f"Error generating q values for halo {halo_key}: {ex}")
                j= j + 1
                #count number of successful and failed halos

        print(f'Generated q values from {i} halos, failed for {j} halos.')
        
        return np.array(q_values)

    def generate_q_distribution_all_halos_sideon(self, reff_index=0):
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

        i = 0
        j = 0
        for halo_key in self.halos.keys():
            interpolator = self.interpolators[halo_key]

            # Generate random viewing angles for this halo
            x_angles, y_angles = [0],[90] #side-on view
            # Get q values for each angle
            try:
                e = np.array([interpolator(x, y, reff_index) for x, y in zip(x_angles, y_angles)])
                q = 1 - e
                q_values.extend(q)
                i = i + 1
            except Exception as ex:
                print(f"Error generating q values for halo {halo_key}: {ex}")
                j = j + 1
                # count number of successful and failed halos

        print(f'Generated q values from {i} halos, failed for {j} halos.')

        return np.array(q_values)

    def run_inference_single_halo(self, sim_name, halo_id, n_angles, reff_index=0,
                                  n_walkers=32, n_steps=3000, burn_in=500, n_cores=None,
                                  output_prefix=None, output_dir="results", force_rerun=False):
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
                samples, max_prob_params, chain, q_obs, sampler = existing_results
                print(f"Using existing inference results for {sim_name} halo {halo_id}")

        if force_rerun or not existing_results:
            # Record start time
            start_time = time.time()

            # Generate q distribution
            q_obs = self.generate_q_distribution_single_halo(
                sim_name, halo_id, n_angles, reff_index
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

        #get true_params from halo_data
        halo_data = self.halos[(sim_name, halo_id)]
        ba_s = halo_data['ba_s']
        ca_s = halo_data['ca_s']
        true_params = np.array([ba_s, ca_s, 0, 0])

        # Plot results
        self.plot_results(
            samples=samples,
            max_prob_params=max_prob_params,
            true_params=true_params,
            q_obs=q_obs,
            chain=chain,
            burn_in=burn_in,
            label=f"{sim_name} Halo {halo_id}",
            color="blue",
            output_prefix=output_prefix,
            output_dir=str(full_output_dir)
        )

        return samples, max_prob_params, sampler, q_obs

    # Here's a fixed version of your run_inference_all_halos method to ensure consistency
    def run_inference_all_halos(self, n_angles_per_halo, reff_index=0, weighted=False,
                                n_walkers=32, n_steps=3000, burn_in=500, n_cores=None,
                                output_prefix="all_halos", output_dir="results",
                                force_rerun=False,
                                label="All Halos Combined", color="blue"):
        """
        Run shape inference on a distribution from all halos.

        Parameters:
            n_total_angles (int): Total number of random viewing angles
            reff_index (int): Index into reff_multipliers to use
            weighted (bool): If True, sample each halo with equal probability
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
        # Create full output directory path
        full_output_dir = Path(output_dir) / "all_halos"
        os.makedirs(full_output_dir, exist_ok=True)

        # Check for existing results
        existing_results = None
        if not force_rerun:
            existing_results = check_existing_results(str(full_output_dir), output_prefix)
            if existing_results:
                samples, max_prob_params, chain, q_obs, sampler = existing_results
                print(f"Using existing inference results for all halos combined")

        if force_rerun or not existing_results:
            # Record start time
            start_time = time.time()

            # Generate q distribution
            q_obs = self.generate_q_distribution_all_halos(
                n_angles_per_halo, reff_index)
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
        
        # Get true_params by averaging over all halos
        ba_s = []
        ca_s = []
        for halo_key in self.halos.keys():
            halo_data = self.halos[halo_key]
            ba_s.append(halo_data['ba_s'])
            ca_s.append(halo_data['ca_s'])
        ba_s_sigma = np.std(ba_s)
        ca_s_sigma = np.std(ca_s)

        ba_s = np.mean(ba_s)
        ca_s = np.mean(ca_s)

        true_params = np.array([ba_s, ca_s, ba_s_sigma, ca_s_sigma])
        model_samples = 10000

        q_true,_,_ = generate_model_projections([ba_s, ca_s, ba_s_sigma, ca_s_sigma], model_samples)
        q_model,_,_ = generate_model_projections(max_prob_params, model_samples)


        print(q_obs,q_true,q_model)

        recovery_results = assess_parameter_recovery(max_prob_params, true_params, q_obs=q_obs, q_model=q_model, q_true=q_true)

        # Save recovery results to text file
        with open(full_output_dir / f"{output_prefix}_recovery.txt", 'w') as f:
            f.write(recovery_results['output_text'])

        #check that color is valid, if not, set to none. (might already be none)
        if not is_color_like(color):
            color = None
        

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
            label=label,
            color=color,
            output_prefix=output_prefix,
            output_dir=str(full_output_dir)
        )

        return samples, max_prob_params, sampler, q_obs

    def plot_results(self, samples, max_prob_params, q_obs,q_model=None,q_true=None, true_params=None,
                     chain=None, burn_in=500, label="Model", color="blue",
                     output_prefix=None, output_dir="results"):
        """
        Plot inference results with model comparison.

        Parameters:
            samples (array): MCMC samples
            max_prob_params (array): Parameters with highest probability
            q_obs (array): Observed projected axis ratios
            true_params (array): True parameters (if known)
            chain (array): MCMC chain
            burn_in (int): Number of burn-in steps
            label (str): Label for plots
            color (str): Color for plots
            output_prefix (str): Prefix for output files
            output_dir (str): Directory to save results
        """
        if output_prefix is None:
            output_prefix = label.lower().replace(" ", "_")

        print("True params:", true_params)
        print("Max prob params:", max_prob_params)

        os.makedirs(output_dir, exist_ok=True)

        # Plot histogram of projections WITH model overlay
        fig_hist = plot_projected_distributions_with_model(
            [q_obs], [max_prob_params], [true_params],[label], [color],
            q_model=q_model, q_true=q_true,
            output_file=os.path.join(output_dir, f"{output_prefix}_projections_comparison.png"),
            title=f"Projected Axis Ratios: Observed vs Model for {label}",
            bin_width=0.1,
        )
        plt.close(fig_hist)


        fig_hist_obs = plot_projected_distributions(
            [q_obs], [label], [color],
            output_file=os.path.join(output_dir, f"{output_prefix}_projections_hist.png"),
            title=f"Projected Axis Ratios for {label}",
            bin_width=0.1, kde=False,
        )
        plt.close(fig_hist_obs)


        # Plot corner plot
        fig_corner = plot_corner(
            samples, max_prob_params, true_params,
            output_file=os.path.join(output_dir, f"{output_prefix}_corner.png"),
            title=f"Parameter Inference for {label}"
        )
        plt.close(fig_corner)

        # Plot chain evolution if chain is provided
        if chain is not None:
            fig_chain = plot_chain_evolution(
                chain, burn_in=burn_in,
                output_file=os.path.join(output_dir, f"{output_prefix}_chain.png"),
                title=f"MCMC Chain Evolution for {label}"
            )
            plt.close(fig_chain)

        # Create ellipsoid shapes plot
        fig_shapes = plot_ellipsoid_shapes(
            [samples], [max_prob_params], [true_params],
            [label], color,
            output_file=os.path.join(output_dir, f"{output_prefix}_ellipsoid_shapes.png"),
            title=f"Intrinsic Shapes: {label}",
            focus_on_max_prob=True
        )
        plt.close(fig_shapes)

        # Create ellipsoid shapes with all samples
        fig_shapes_all = plot_ellipsoid_shapes(
            [samples], [max_prob_params], [true_params],
            [label], ["blue"],
            output_file=os.path.join(output_dir, f"{output_prefix}_ellipsoid_shapes_all.png"),
            title=f"Intrinsic Shapes: {label}",
            focus_on_max_prob=False, show_samples=True, show_ellipses=False
        )
        plt.close(fig_shapes_all)


    def get_all_halo_keys(self):
        """Return a list of all (sim_name, halo_id) keys in the collection."""
        return list(self.halos.keys())

    def get_halo_count(self):
        """Return the number of halos in the collection."""
        return len(self.halos)

    def get_reff_multipliers(self, sim_name, halo_id):
        """Return the effective radius multipliers for a specific halo."""
        halo_key = (sim_name, halo_id)
        if halo_key not in self.reff_multipliers:
            raise KeyError(f"Halo {halo_id} from simulation {sim_name} not found in collection")
        return self.reff_multipliers[halo_key]