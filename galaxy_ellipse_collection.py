import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import pickle
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf

from shape_inference import (
    generate_projections,
    generate_ellipsoid_distribution,
    generate_projections_from_distribution,
    infer_intrinsic_shape,
    load_results
)

from shape_plotting import (
    plot_corner,
    plot_ellipsoid_shapes,
    plot_projected_distributions,
    plot_chain_evolution,
    plot_comparison_grid,
    plot_statistics
)


def interpolate_orientations(halo_ellipses, reff_multipliers=[2, 3, 4],
                             interpolation_method='linear', coordinate_system='angles'):
    """
    Function that interpolates ellipticity values for arbitrary viewing angles.
    Handles periodic boundary conditions.
    """

    # Function to extract angles from orientation string
    def parse_orientation(orientation):
        try:
            x_angle = int(orientation[1:4])  # Rotation around x-axis in degrees
            y_angle = int(orientation[5:8])  # Rotation around y-axis in degrees
            return x_angle, y_angle
        except (ValueError, IndexError):
            print(f"Invalid orientation format: {orientation}")
            return None, None

    # Function to convert angles to 3D unit vector if using vector coordinates
    def angles_to_vector(x_angle, y_angle):
        x_rad = np.radians(x_angle)
        y_rad = np.radians(y_angle)

        # Start with vector pointing along z-axis (0, 0, -1)
        # Rotate around y-axis (affects x and z)
        vx = np.sin(y_rad)
        vy = 0
        vz = -np.cos(y_rad)

        # Rotate around x-axis (affects y and z)
        new_vy = vy * np.cos(x_rad) - vz * np.sin(x_rad)
        new_vz = vy * np.sin(x_rad) + vz * np.cos(x_rad)

        return [vx, new_vy, new_vz]

    # Prepare data points and values with periodic boundary handling
    original_points = []
    extended_points = []  # Will include periodic boundary points
    original_values = {i: [] for i in range(len(reff_multipliers))}
    extended_values = {i: [] for i in range(len(reff_multipliers))}

    # First collect all original data points
    for orientation, ellipticities in halo_ellipses.items():
        # Skip non-orientation keys
        if not (orientation.startswith('x') and 'y' in orientation):
            continue

        x_angle, y_angle = parse_orientation(orientation)
        if x_angle is None:
            continue

        # Store original point
        if coordinate_system == 'vectors':
            point = angles_to_vector(x_angle, y_angle)
        else:  # 'angles' is default
            point = [x_angle, y_angle]

        original_points.append(point)

        # Store original ellipticity values
        for i, eps in enumerate(ellipticities):
            if i < len(reff_multipliers):
                original_values[i].append(float(eps))

    # Now create extended dataset with periodic boundaries
    extended_points = original_points.copy()
    for i in range(len(reff_multipliers)):
        extended_values[i] = original_values[i].copy()

    # Handle periodicity by adding mirrored points
    if coordinate_system == 'angles':
        for idx, point in enumerate(original_points):
            x_angle, y_angle = point

            # Add points for x-periodicity (0° = 180°)
            if x_angle == 0:
                new_point = [180, y_angle]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif x_angle == 180:
                new_point = [0, y_angle]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])

            # Add points for y-periodicity (0° = 360°)
            if y_angle == 0:
                new_point = [x_angle, 360]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif y_angle == 360 or y_angle == 359:
                new_point = [x_angle, 0]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])

            # Add points for corner cases
            if (x_angle == 0 and y_angle == 0):
                new_point = [180, 360]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 180 and y_angle == 0):
                new_point = [0, 360]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 0 and (y_angle == 360 or y_angle == 359)):
                new_point = [180, 0]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])
            elif (x_angle == 180 and (y_angle == 360 or y_angle == 359)):
                new_point = [0, 0]
                extended_points.append(new_point)
                for i in range(len(reff_multipliers)):
                    extended_values[i].append(original_values[i][idx])

    # filter out nan data points:
    # Filter out NaN values before creating interpolators
    for i in range(len(reff_multipliers)):
        if len(extended_values[i]) == 0:
            continue

        # Convert to numpy arrays for efficient filtering
        points_array = np.array(extended_points)
        values_array = np.array(extended_values[i])

        # Create boolean mask for non-NaN values
        valid_mask = ~np.isnan(values_array)

        # Apply mask to filter out NaN values
        points_array = points_array[valid_mask]
        values_array = values_array[valid_mask]

        # Store the filtered arrays back for interpolation
        extended_points_filtered = points_array
        extended_values[i] = values_array

    # Create interpolators based on selected method using filtered dataset
    interpolators = {}
    fallback_interpolators = {}

    for i in range(len(reff_multipliers)):
        if len(extended_values[i]) == 0:
            continue

        points_array = np.array(extended_points_filtered)
        values_array = np.array(extended_values[i])

        if interpolation_method == 'rbf':
            if coordinate_system == 'vectors':
                interpolators[i] = Rbf(
                    points_array[:, 0], points_array[:, 1], points_array[:, 2],
                    values_array, function='multiquadric'
                )
            else:
                interpolators[i] = Rbf(
                    points_array[:, 0], points_array[:, 1],
                    values_array, function='multiquadric'
                )
        elif interpolation_method == 'nearest':
            interpolators[i] = NearestNDInterpolator(points_array, values_array)
        else:  # 'linear' is default
            interpolators[i] = LinearNDInterpolator(points_array, values_array)
            fallback_interpolators[i] = NearestNDInterpolator(points_array, values_array)

    # Define the interpolation function with periodic boundary handling
    def interpolate(x_angle, y_angle, reff_index=0):
        """
        Estimate ellipticity at arbitrary viewing angles with periodic boundaries.
        """
        # Normalize angles to the periodic domain
        if x_angle > 180:
            x_angle = x_angle % 180
        if y_angle > 360:
            y_angle = y_angle % 360

        if reff_index not in interpolators:
            raise ValueError(f"No data for reff_index {reff_index} (multiplier={reff_multipliers[reff_index]})")

        # Interpolate based on selected method and coordinate system
        if interpolation_method == 'rbf':
            if coordinate_system == 'vectors':
                vx, vy, vz = angles_to_vector(x_angle, y_angle)
                return float(interpolators[reff_index](vx, vy, vz))
            else:
                return float(interpolators[reff_index](x_angle, y_angle))
        else:
            if coordinate_system == 'vectors':
                point = angles_to_vector(x_angle, y_angle)
            else:
                point = [x_angle, y_angle]

            result = interpolators[reff_index](point)

            # For linear interpolation, use fallback if out of convex hull
            if interpolation_method == 'linear' and np.isnan(result):
                result = fallback_interpolators[reff_index](point)

            return float(result)

    return interpolate, reff_multipliers


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

    def add_halo(self, sim_name, halo_id, halo_data, reff_multipliers=None,
                 interpolation_method='linear', coordinate_system='angles'):
        """
        Add a halo to the collection with its data and create interpolator.

        Parameters:
            sim_name (str): Simulation name
            halo_id (int/str): Halo identifier
            halo_data (dict): Dictionary containing ellipse data
            reff_multipliers (list): List of effective radius multipliers
            interpolation_method (str): Method for interpolation
            coordinate_system (str): Coordinate system for interpolation
        """
        halo_key = (sim_name, halo_id)
        self.halos[halo_key] = halo_data

        # Create interpolation function
        interpolate_func, reff_mults = interpolate_orientations(
            halo_data,
            reff_multipliers=reff_multipliers,
            interpolation_method=interpolation_method,
            coordinate_system=coordinate_system
        )

        self.interpolators[halo_key] = interpolate_func
        self.reff_multipliers[halo_key] = reff_mults

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



        for halo_key in self.halos.keys():
            interpolator = self.interpolators[halo_key]

            # Generate random viewing angles for this halo
            x_angles, y_angles = random_viewing_angles(angles_per_halo)

            # Get q values for each angle
            e = np.array([interpolator(x, y, reff_index) for x, y in zip(x_angles, y_angles)])
            q = 1 - e
            q_values.extend(q)



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
        full_output_dir = Path(output_dir) / f"{sim_name}_{halo_id}"
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
                                force_rerun=False):
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
        # Plot results
        self.plot_results(
            samples=samples,
            max_prob_params=max_prob_params,
            true_params=true_params,
            q_obs=q_obs,
            chain=chain,
            burn_in=burn_in,
            label="All Halos Combined",
            color="red",
            output_prefix=output_prefix,
            output_dir=str(full_output_dir)
        )

        return samples, max_prob_params, sampler, q_obs

    def plot_results(self, samples, max_prob_params, q_obs, true_params=None,
                     chain=None, burn_in=500, label="Model", color="blue",
                     output_prefix=None, output_dir="results"):
        """
        Plot inference results.

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

        print(true_params)
        print(max_prob_params)

        os.makedirs(output_dir, exist_ok=True)

        # Plot histogram of projections
        fig_hist = plot_projected_distributions(
            [q_obs], [label], [color],
            output_file=os.path.join(output_dir, f"{output_prefix}_projections_hist.png"),
            title=f"Projected Axis Ratios for {label}"
        )
        plt.close(fig_hist)

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
            [samples], [max_prob_params], [true_params],  # Use max_prob as "true" params
            [label], ["blue"],
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