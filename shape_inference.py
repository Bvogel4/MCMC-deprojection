"""
shape_inference.py - Core functionality for 3D shape inference from 2D projections.

This module implements the methodology from Kado-Fong et al. (2020) to infer
the intrinsic 3D shapes of ellipsoids from their 2D projections.
"""

import numpy as np
import emcee
import os
import time
from multiprocessing import Pool
from scipy.special import factorial as scipy_factorial
import pickle


def random_viewing_angles(n):
    """
    Generate n random viewing angles uniformly distributed on a sphere.

    Parameters:
        n (int): Number of viewing angles to generate

    Returns:
        tuple: (phi, theta) where phi is in [0, 2π] and theta is in [0, π]
    """
    phi = np.random.uniform(0, 2 * np.pi, n)
    nu = np.random.uniform(0, 1, n)
    theta = np.arccos(2 * nu - 1)
    return phi, theta


def projected_axis_ratio(phi, theta, B, C):
    """
    Calculate the projected axis ratio q for a triaxial ellipsoid with axis ratios B/A and C/A
    viewed from angles (phi, theta).

    Based on Simonneau et al. (1998) as used in Kado-Fong et al. (2020).

    Parameters:
        phi (float or array): Azimuthal viewing angle in radians
        theta (float or array): Polar viewing angle in radians
        B (float or array): B/A axis ratio (intermediate/major)
        C (float or array): C/A axis ratio (minor/major)

    Returns:
        float or array: Projected axis ratio q = b/a
    """
    # Calculate f and g as per equations in Simonneau et al. (1998)
    f = np.sqrt(
        (C * np.sin(theta) * np.cos(phi)) ** 2 +
        (B * C * np.sin(theta) * np.sin(phi)) ** 2 +
        (B * np.cos(theta)) ** 2
    )

    g = (
            np.cos(phi) ** 2 + np.cos(theta) ** 2 * np.sin(phi) ** 2 +
            B ** 2 * (np.sin(phi) ** 2 + np.cos(theta) ** 2 * np.cos(phi) ** 2) +
            (C * np.sin(theta)) ** 2
    )

    # Calculate h
    h = np.sqrt((g - 2 * f) / (g + 2 * f))

    # Calculate q
    q = (1 - h) / (1 + h)

    return q


def generate_projections(a, b, c, n_samples=10000):
    """
    Generate n_samples random projections of an ellipsoid with semi-axes a, b, c.

    Parameters:
        a, b, c (float): Semi-axes of the ellipsoid (a >= b >= c)
        n_samples (int): Number of random projections to generate

    Returns:
        array: Projected axis ratios q = b/a
    """
    # Calculate axis ratios
    B_A = b / a
    C_A = c / a

    # Generate random viewing angles
    phi, theta = random_viewing_angles(n_samples)

    # Calculate projected axis ratios
    q = projected_axis_ratio(phi, theta, B_A, C_A)

    return q


def generate_ellipsoid_distribution(n_ellipsoids, mu_B, sigma_B, mu_C, sigma_C):
    """
    Generate a distribution of ellipsoids with given parameters.

    Parameters:
        n_ellipsoids (int): Number of ellipsoids to generate
        mu_B, sigma_B (float): Mean and standard deviation of B/A
        mu_C, sigma_C (float): Mean and standard deviation of C/A

    Returns:
        tuple: Arrays of sampled B/A and C/A values
    """
    # Sample B/A values from a normal distribution
    B_A = np.random.normal(mu_B, sigma_B, n_ellipsoids)

    # Sample C/A values from a normal distribution
    C_A = np.random.normal(mu_C, sigma_C, n_ellipsoids)

    # Enforce physical constraints: 0 < C/A <= B/A <= 1
    mask = (B_A > 0) & (B_A <= 1) & (C_A > 0) & (C_A <= B_A)
    B_A = B_A[mask]
    C_A = C_A[mask]

    # If too many samples were filtered out, generate more
    while len(B_A) < n_ellipsoids:
        additional = n_ellipsoids - len(B_A)
        B_additional = np.random.normal(mu_B, sigma_B, additional)
        C_additional = np.random.normal(mu_C, sigma_C, additional)

        mask = (B_additional > 0) & (B_additional <= 1) & (C_additional > 0) & (C_additional <= B_additional)
        B_A = np.append(B_A, B_additional[mask])
        C_A = np.append(C_A, C_additional[mask])

    # Trim to exactly n_ellipsoids
    B_A = B_A[:n_ellipsoids]
    C_A = C_A[:n_ellipsoids]

    return B_A, C_A


def generate_projections_from_distribution(B_A, C_A, n_projections_per_ellipsoid=10):
    """
    Generate random projections from a distribution of ellipsoids.

    Parameters:
        B_A, C_A (array): Arrays of B/A and C/A values for the ellipsoids
        n_projections_per_ellipsoid (int): Number of random projections per ellipsoid

    Returns:
        array: Projected axis ratios q = b/a
    """
    n_ellipsoids = len(B_A)
    q_values = []

    for i in range(n_ellipsoids):
        # Generate random viewing angles for this ellipsoid
        phi, theta = random_viewing_angles(n_projections_per_ellipsoid)

        # Calculate projected axis ratios
        q = projected_axis_ratio(phi, theta, B_A[i], C_A[i])
        q_values.extend(q)

    return np.array(q_values)


def log_likelihood(params, q_obs):
    """
    Calculate the log likelihood using equation (6) from Kado-Fong et al.

    ln p(q|μB,μC,σB,σC) = ∑(ni ln(mi) - mi - ln(ni!))

    where ni is the observed count where 0.04i < q ≤ 0.04(i+1)
    and mi is the predicted count in the same range.

    Parameters:
        params (list): [mu_B, mu_C, sigma_B, sigma_C]
        q_obs (array): Observed projected axis ratios

    Returns:
        float: Log likelihood
    """

    mu_B, mu_C, sigma_B, sigma_C = params

    # Enforce physical constraints
    # mu_B and mu_C must be in [0, 1]
    if not (0 < mu_B <= 1 and 0 < mu_C <= 1):
        return -np.inf
    # mu_B must be greater than mu_C
    if mu_B <= mu_C:
        return -np.inf
    # sigma_B and sigma_C must be between 0 and 0.5
    if not (0 < sigma_B < 0.5 and 0 < sigma_C < 0.5):
        return -np.inf
    #prevent uncertainty from exceeding physical limits
    # B = C limits
    # we'll let these region extend to 1.5 sigma
    if mu_B - 1.5*sigma_B < mu_C + 1.5*sigma_C:
        return -np.inf
    # B = 0 limits
    if mu_B - 1.5*sigma_B < 0:
        return -np.inf
    # B = 1 limits
    if mu_B + 1.5*sigma_B > 1:
        return -np.inf
    # C = 0 limits
    if mu_C - 1.5*sigma_C < 0:
        return -np.inf
    # C = 1 limits
    if mu_C + 1.5*sigma_C > 1:
        return -np.inf

    
    

    # Generate model projections
    n_model_draws = len(q_obs) * 10  # number of draws to approximate the model distribution

    # Vectorized sample generation with efficient filtering
    # Generate more samples initially to account for filtering
    oversample_factor = 2  # Start with twice as many samples
    n_initial_samples = int(n_model_draws * oversample_factor)

    # Create arrays to store B and C values
    B_samples = np.random.normal(mu_B, sigma_B, n_initial_samples)
    C_samples = np.random.normal(mu_C, sigma_C, n_initial_samples)

    # Enforce physical constraints: 0 < C <= B <= 1
    mask = (B_samples > 0) & (B_samples <= 1) & (C_samples > 0) & (C_samples <= B_samples)

    B_samples = B_samples[mask]
    C_samples = C_samples[mask]

    # If we don't have enough samples after filtering, generate more efficiently
    if len(B_samples) < n_model_draws:
        samples_needed = n_model_draws - len(B_samples)

        # More efficient approach: use rejection sampling in batches
        while len(B_samples) < n_model_draws:
            batch_size = min(samples_needed * 2, n_initial_samples)  # Adaptive batch size

            B_additional = np.random.normal(mu_B, sigma_B, batch_size)
            C_additional = np.random.normal(mu_C, sigma_C, batch_size)

            mask = (B_additional > 0) & (B_additional <= 1) & (C_additional > 0) & (C_additional <= B_additional)

            B_samples = np.append(B_samples, B_additional[mask])
            C_samples = np.append(C_samples, C_additional[mask])

            samples_needed = n_model_draws - len(B_samples)
            if samples_needed <= 0:
                break

    # Trim to exactly n_model_draws
    B_samples = B_samples[:n_model_draws]
    C_samples = C_samples[:n_model_draws]

    # Generate random viewing angles (vectorized)
    phi, theta = random_viewing_angles(len(B_samples))

    # Calculate projected axis ratios
    q_model = projected_axis_ratio(phi, theta, B_samples, C_samples)

    # Choose a bin size from a uniform distribution between [0.03 and 0.1]
    #bin_size = np.random.uniform(0.03, 0.1)

    # Create bins for histogram
    #bins = np.arange(0, 1 + bin_size, bin_size)

    #bins from 0-1 in 0.04 increments
    bins = np.arange(0, 1, 0.04)

    # Count observed and model values in each bin
    n_obs, _ = np.histogram(q_obs, bins=bins)
    n_model, _ = np.histogram(q_model, bins=bins)

    # Normalize model counts to match observed counts
    n_model = n_model / np.sum(n_model) * np.sum(n_obs)

    # Vectorized stirling approximation
    def vectorized_stirling(n_array):
        result = np.zeros_like(n_array, dtype=float)
        small_mask = n_array < 30
        large_mask = ~small_mask

        # For small values, use scipy factorial
        if np.any(small_mask):
            small_n = n_array[small_mask]
            result[small_mask] = np.array([np.log(scipy_factorial(n)) for n in small_n])

        # For large values, use Stirling's approximation
        if np.any(large_mask):
            large_n = n_array[large_mask]
            result[large_mask] = (large_n + 0.5) * np.log(large_n) - large_n + 0.5 * np.log(2 * np.pi)

        return result

    # Vectorized log-likelihood calculation
    ni = n_obs
    mi = n_model

    # Create masks for different cases
    mask_both_positive = (ni > 0) & (mi > 0)
    mask_ni_positive_mi_zero = (ni > 0) & (mi == 0)
    mask_ni_zero_mi_positive = (ni == 0) & (mi > 0)

    # Initialize log likelihood
    log_like = 0

    # Case 1: Both positive
    if np.any(mask_both_positive):
        ni_pos = ni[mask_both_positive]
        mi_pos = mi[mask_both_positive]
        log_like += np.sum(ni_pos * np.log(mi_pos) - mi_pos - vectorized_stirling(ni_pos))

    # Case 2: Model incorrectly predicts impossible event
    if np.any(mask_ni_positive_mi_zero):
        ni_pos_mi_zero = ni[mask_ni_positive_mi_zero]
        log_like += -np.sum(ni_pos_mi_zero)

    # Case 3: Model predicts events that didn't occur
    if np.any(mask_ni_zero_mi_positive):
        mi_pos_ni_zero = mi[mask_ni_zero_mi_positive]
        log_like += -np.sum(mi_pos_ni_zero)

    # Case 4: Agreement on zero counts (adds nothing)


    return log_like



def infer_intrinsic_shape(q_obs, n_walkers=32, n_steps=3000, burn_in=500,
                          n_cores=None, output_prefix=None, output_dir="results"):
    """
    Infer the intrinsic shape distribution from observed projected axis ratios.

    Parameters:
        q_obs (array): Observed projected axis ratios
        n_walkers (int): Number of MCMC walkers
        n_steps (int): Number of MCMC steps
        burn_in (int): Number of burn-in steps to discard
        initial_guess (list): Initial parameter guess [mu_B, mu_C, sigma_B, sigma_C]
        n_cores (int): Number of CPU cores to use
        output_prefix (str): Prefix for output files
        output_dir (str): Directory to save results

    Returns:
        tuple: (samples, max_prob_params, sampler) - MCMC samples, parameters with highest probability, and the sampler
    """
    # Set initial guess if not provided

    ndim = 4  # Number of parameters

    # # Initialize walkers in a small ball around the initial guess
    # pos = [initial_guess + 1e-2 * np.random.randn(ndim) for _ in range(n_walkers)]

    # intial walker positions uniformly distributed in the valid b/a vs c/a space
    #first guess of where to start based on peak of q_obs
    q_obs_peak = np.mean(q_obs)
    q_obs_std = np.std(q_obs)
    mu_B = q_obs_peak + q_obs_std
    mu_C = q_obs_peak - q_obs_std
    sigma_B = q_obs_std/2
    sigma_C = q_obs_std/2
    #make sure intial guess is within physical limits
    if mu_B > 0.9:
        mu_B = .9
    if mu_C < 0.1:
        mu_C = 0.1
    if sigma_B > 0.4:
        sigma_B = 0.4
    if sigma_C > 0.4:
        sigma_C = 0.4
    #start all walkers at the same initial guess
    pos = np.array([[mu_B, mu_C, sigma_B, sigma_C] for _ in range(n_walkers)])
    #add a little variation to each walker
    pos += 0.1 * np.random.randn(n_walkers, ndim)
    #ensure walkers are within physical limits
    for i in range(n_walkers):
        if pos[i, 0] > 1:
            pos[i, 0] = 1
        if pos[i, 1] < 0:
            pos[i, 1] = 0
        if pos[i, 2] > 0.5:
            pos[i, 2] = 0.5
        if pos[i, 3] > 0.5:
            pos[i, 3] = 0.5
        if pos[i, 0] < pos[i, 1]:
            pos[i,0],pos[i,1] = pos[i,1],pos[i,0]

    # Set up the sampler with multiprocessing if requested
    if n_cores is not None and n_cores > 1:
        with Pool(processes=n_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=[q_obs], pool=pool)

            # Run MCMC
            print(f"Running MCMC with {n_cores} processes...")
            start_time = time.time()
            sampler.run_mcmc(pos, n_steps, progress=True)
            end_time = time.time()
            print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    else:
        # Run without multiprocessing
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=[q_obs])

        # Run MCMC
        print("Running MCMC...")
        start_time = time.time()
        sampler.run_mcmc(pos, n_steps, progress=True)
        end_time = time.time()
        print(f"MCMC completed in {end_time - start_time:.2f} seconds")

    # Discard burn-in and get samples
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    # Find the parameters with highest probability
    max_prob_idx = np.argmax(log_probs)
    max_prob_params = samples[max_prob_idx]

    # Save results if output_prefix is provided
    if output_prefix:
        os.makedirs(output_dir, exist_ok=True)

        # Save samples and parameters
        np.save(f"{output_dir}/{output_prefix}_samples.npy", samples)
        np.save(f"{output_dir}/{output_prefix}_max_prob_params.npy", max_prob_params)
        np.save(f"{output_dir}/{output_prefix}_log_probs.npy", log_probs)

        # Save full chain for diagnostics
        full_chain = sampler.get_chain()
        np.save(f"{output_dir}/{output_prefix}_full_chain.npy", full_chain)

        # Save observed q values
        np.save(f"{output_dir}/{output_prefix}_q_obs.npy", q_obs)

        # Save all results in a pickle file for easy loading
        results = {
            'samples': samples,
            'max_prob_params': max_prob_params,
            'log_probs': log_probs,
            'full_chain': full_chain,
            'q_obs': q_obs,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'burn_in': burn_in,
        }

        with open(f"{output_dir}/{output_prefix}_results.pkl", 'wb') as f:
            pickle.dump(results, f)

    return samples, max_prob_params, sampler


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