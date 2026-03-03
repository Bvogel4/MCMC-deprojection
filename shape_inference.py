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
from scipy import stats
#supress warning from emcee
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="emcee")


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
    return phi,theta


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


import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri  # Faster than stats.norm.cdf/ppf


def generate_ellipsoid_distribution(n_samples, mu_B, sigma_B, mu_C, sigma_C):

    # Sample B from truncated normal with bounds (0, 1)
    a_B = (0 - mu_B) / sigma_B
    b_B = (1 - mu_B) / sigma_B
    B_samples = stats.truncnorm.rvs(a_B, b_B, loc=mu_B, scale=sigma_B, size=n_samples)

    # Vectorized C sampling using inverse CDF method
    # For truncated normal: X = Φ⁻¹(Φ(a) + U·(Φ(b) - Φ(a))) · σ + μ
    a_C = (0 - mu_C) / sigma_C  # scalar lower bound (standardized)
    b_C = (B_samples - mu_C) / sigma_C  # array of upper bounds (standardized)

    # CDF values at bounds (ndtr is ~3-5x faster than stats.norm.cdf)
    Phi_a = ndtr(a_C)  # scalar
    Phi_b = ndtr(b_C)  # array

    # Uniform samples for inverse CDF transform
    u = np.random.random(n_samples)

    # Inverse CDF transform
    cdf_vals = Phi_a + u * (Phi_b - Phi_a)
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)  # Numerical stability

    C_standardized = ndtri(cdf_vals)  # ndtri is faster than stats.norm.ppf
    C_samples = C_standardized * sigma_C + mu_C

    # Handle edge cases where bounds are invalid (B_val < mu_C effectively)
    invalid = Phi_b <= Phi_a
    C_samples[invalid] = B_samples[invalid] * 0.99

    return  B_samples, C_samples


def generate_model_projections(params, n_samples=10000):
    """
    Generate model projected axis ratios using truncated normal distributions.

    Vectorized version using inverse CDF method for conditional sampling.
    """
    mu_B, mu_C, sigma_B, sigma_C = params

    B_samples, C_samples = generate_ellipsoid_distribution(n_samples, mu_B, sigma_B, mu_C, sigma_C)

    # Generate random viewing angles
    phi, theta = random_viewing_angles(n_samples)

    # Calculate projected axis ratios
    q_model = projected_axis_ratio(phi, theta, B_samples, C_samples)

    return q_model, B_samples, C_samples

def generate_projections_from_distribution(B_A, C_A, n_projections_per_ellipsoid=10):

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
    if not (0 < mu_C < mu_B <= 1):
        return -np.inf
    # sigma_B and sigma_C must be between 0 and 0.8
    if not (0 < sigma_B < 0.8 and 0 < sigma_C < 0.8):
        return -np.inf

    from scipy import stats

    n_model_draws = len(q_obs) * 10  # number of draws to approximate the model distribution

    q_model = generate_model_projections(params, n_samples=n_model_draws)


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



def infer_intrinsic_shape(q_obs, n_walkers=128, n_steps=5000, burn_in=500,
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
    if sigma_B > 0.5:
        sigma_B = 0.5
    if sigma_C > 0.5:
        sigma_C = 0.5
    #start all walkers at the same initial guess
    pos = np.array([[mu_B, mu_C, sigma_B, sigma_C] for _ in range(n_walkers)])
    #add a little variation to each walker
    # Define separate variation scales for means and sigmas
    mu_variation = 0.05  # Less variation for mu parameters
    sigma_variation = 0.2  # More variation for sigma parameters

    # Create random variations with appropriate scales for each parameter
    variations = np.zeros((n_walkers, ndim))
    variations[:, 0] = mu_variation * np.random.randn(n_walkers)  # mu_B
    variations[:, 1] = mu_variation * np.random.randn(n_walkers)  # mu_C
    variations[:, 2] = sigma_variation * np.random.randn(n_walkers)  # sigma_B
    variations[:, 3] = sigma_variation * np.random.randn(n_walkers)  # sigma_C

    # Add variations to positions
    pos = np.array([[mu_B, mu_C, sigma_B, sigma_C] for _ in range(n_walkers)])
    pos += variations
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
        if pos[i, 2] < 0:
            pos[i, 2] = 0
        if pos[i, 3] < 0:
            pos[i, 3] = 0
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