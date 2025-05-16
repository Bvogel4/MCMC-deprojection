"""
Test script for 3D shape inference from known 3d Shapes, where projections are calculated mathematically,
see Xu and Randall 2020

This script uses the functionality from shape_inference.py and shape_plotting.py
to generate data, run inference, and create visualizations.
"""

import os
import numpy as np
import pandas as pd
import time
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import stats
import pickle

# Import custom modules
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

# Configuration parameters
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# MCMC parameters
N_STEPS = 3000  # Number of MCMC steps
BURN_IN = 1500  # Number of burn-in steps to discard
N_CORES = 32  # Number of CPU cores to use for parallel processing
N_WALKERS = N_CORES # Number of MCMC walkers

# Data generation parameters
N_ELLIPSOIDS = 100  # Number of ellipsoids to generate for distribution
N_PROJECTIONS = 8000  # Total number of projections to generate

# Output directory
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the test ellipsoid shapes from Kado-Fong et al.
KADO_FONG_SHAPES = {
    "disk": {
        "params": (1.0, 0.9, 0.2),  # (A, B, C)
        "color": "blue",
        "label": "Disk"
    },
    "spheroid": {
        "params": (1.0, 0.9, 0.9),  # (A, B, C)
        "color": "red",
        "label": "Spheroid"
    },
    "prolate": {
        "params": (1.0, 0.1, 0.1),  # (A, B, C)
        "color": "green",
        "label": "Prolate"
    },
    "Triaxial": {
        "params": (1.0, 0.75, 0.4),  # (A, B, C)
        "color": "purple",
        "label": "Triaxial"
    }

}

# Define CDM and SIDM model parameters
CDM_PARAMS = (0.752, 0.293, 0.15, 0.15)  # (mu_B, mu_C, sigma_B, sigma_C)
SIDM_PARAMS = (0.859, 0.354, 0.1, 0.1)  # (mu_B, mu_C, sigma_B, sigma_C)

# Custom models
CUSTOM_MODELS = {
    "disk_like": (0.8, 0.2, 0.1, 0.1),
    "spheroid_like": (0.9, 0.8, 0.1, 0.1),
}


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
            if all(k in results for k in ['samples', 'max_prob_params', 'full_chain']):
                print("Loaded existing MCMC results.")
                return results['samples'], results['max_prob_params'], results['full_chain']
            else:
                print("Existing results incomplete. Will rerun analysis.")
                return None

        except Exception as e:
            print(f"Error loading existing results: {e}")
            return None

    return None


def run_shape_analysis(
        models_dict,
        test_name,
        output_dir=OUTPUT_DIR,
        use_abc_format=False,
        force_rerun=False,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        burn_in=BURN_IN,
        n_cores=N_CORES,
        n_ellipsoids=N_ELLIPSOIDS,
        n_projections=N_PROJECTIONS
):
    """
    Run shape analysis for a dictionary of models.

    Parameters:
        models_dict (dict): Dictionary of models to analyze
        test_name (str): Name of the test (used for output directory)
        output_dir (str): Base output directory
        use_abc_format (bool): Whether models are in ABC format (True) or mu_B/mu_C format (False)
        force_rerun (bool): Whether to force rerunning inference even if results exist
        n_walkers (int): Number of MCMC walkers
        n_steps (int): Number of MCMC steps
        burn_in (int): Number of burn-in steps to discard
        n_cores (int): Number of CPU cores for parallel processing
        n_ellipsoids (int): Number of ellipsoids to generate
        n_projections (int): Total number of projections to generate

    Returns:
        dict: Dictionary of results for all models
    """
    test_dir = Path(output_dir) / test_name
    test_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nRunning {test_name} analysis...")
    print(f"Using {n_projections} projections, {n_walkers} walkers, {n_steps} steps, {burn_in} burn-in")

    # Store results
    results = {}
    q_obs_list = []
    labels = []
    colors = []

    # For each model
    for model_name, model_info in models_dict.items():
        print(f"\nProcessing {model_name} model...")
        model_dir = test_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Extract parameters and metadata
        if isinstance(model_info, tuple):
            # Simple tuple format
            params = model_info
            color = plt.cm.tab10.colors[len(results) % 10]  # Auto-assign color
            label = model_name.replace('_', ' ').title()
        else:
            # Dictionary format
            params = model_info["params"]
            color = model_info.get("color", plt.cm.tab10.colors[len(results) % 10])
            label = model_info.get("label", model_name.replace('_', ' ').title())

        # Convert from ABC format to muB/muC format if needed
        if use_abc_format:
            a, b, c = params
            mu_B_true = b / a
            mu_C_true = c / a
            true_params = (mu_B_true, mu_C_true, 0.0, 0.0)  # Zero dispersion for test cases
        else:
            mu_B, mu_C, sigma_B, sigma_C = params
            true_params = (mu_B, mu_C, sigma_B, sigma_C)

        # Check for existing results
        if force_rerun:
            existing_results = None 
        else:
            existing_results = check_existing_results(model_dir, model_name)

        if existing_results:
            samples, max_prob_params, chain = existing_results

            # Check if we need to load q_obs
            q_obs_path = model_dir / f"{model_name}_q_obs.npy"
            if q_obs_path.exists():
                q_obs = np.load(q_obs_path)
            else:
                # We need to regenerate projections
                print("Existing q_obs not found. Regenerating projections...")
                if use_abc_format:
                    a, b, c = params
                    q_obs = generate_projections(a, b, c, n_samples=n_projections)
                else:
                    mu_B, mu_C, sigma_B, sigma_C = params
                    B_A_values, C_A_values = generate_ellipsoid_distribution(
                        n_ellipsoids, mu_B, sigma_B, mu_C, sigma_C
                    )
                    q_obs = generate_projections_from_distribution(
                        B_A_values, C_A_values,
                        n_projections_per_ellipsoid=n_projections // n_ellipsoids
                    )

                # Save the projections
                np.save(model_dir / f"{model_name}_q_obs.npy", q_obs)
        else:
            # Generate new data and run inference
            start_time = time.time()

            if use_abc_format:
                # Generate projections directly from ABC parameters
                a, b, c = params
                print(f"Generating {n_projections} projections...")
                q_obs = generate_projections(a, b, c, n_samples=n_projections)
            else:
                # Generate ellipsoid distribution and then projections
                mu_B, mu_C, sigma_B, sigma_C = params
                print(f"Generating {n_ellipsoids} ellipsoids...")
                B_A_values, C_A_values = generate_ellipsoid_distribution(
                    n_ellipsoids, mu_B, sigma_B, mu_C, sigma_C
                )

                # Save ellipsoid parameters
                np.save(model_dir / f"{model_name}_B_A_values.npy", B_A_values)
                np.save(model_dir / f"{model_name}_C_A_values.npy", C_A_values)

                # Generate projections
                print(f"Generating projections...")
                q_obs = generate_projections_from_distribution(
                    B_A_values, C_A_values,
                    n_projections_per_ellipsoid=n_projections // n_ellipsoids
                )

            # Save projections
            np.save(model_dir / f"{model_name}_q_obs.npy", q_obs)

            end_time = time.time()
            print(f"Data generation completed in {end_time - start_time:.2f} seconds")

            # Run inference
            print(f"Running inference for {model_name}...")
            samples, max_prob_params, sampler = infer_intrinsic_shape(
                q_obs,
                n_walkers=n_walkers,
                n_steps=n_steps,
                burn_in=burn_in,
                n_cores=n_cores,
                output_prefix=model_name,
                output_dir=str(model_dir)
            )

            # Extract the full chain for plotting
            chain = sampler.get_chain()
            # Get the autocorrelation time and effective sample size
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"Autocorrelation times: {tau}")
            print(f"Number of effective samples: {N_STEPS / np.max(tau)}")

        # Store for combined plots
        q_obs_list.append(q_obs)
        labels.append(label)
        colors.append(color)

        # Plot histogram of projections
        fig_hist = plot_projected_distributions(
            [q_obs], [label], [color],
            output_file=str(model_dir / "projections_hist.png"),
            title=f"Projected Axis Ratios for {label}"
        )
        plt.close(fig_hist)

        # Plot corner plot
        fig_corner = plot_corner(
            samples, max_prob_params, true_params,
            output_file=str(model_dir / "corner.png"),
            title=f"Parameter Inference for {label}"
        )
        plt.close(fig_corner)

        # Plot chain evolution
        fig_chain = plot_chain_evolution(
            chain, burn_in=burn_in,
            output_file=str(model_dir / "chain.png"),
            title=f"MCMC Chain Evolution for {label}"
        )
        plt.close(fig_chain)

        # Print results
        if use_abc_format:
            print(f"True B/A: {true_params[0]:.3f}, True C/A: {true_params[1]:.3f}")
        else:
            print(
                f"True params: μB={true_params[0]:.3f}, μC={true_params[1]:.3f}, σB={true_params[2]:.3f}, σC={true_params[3]:.3f}")

        print(f"Inferred B/A: {max_prob_params[0]:.3f}, Inferred C/A: {max_prob_params[1]:.3f}")
        print(f"Inferred σB: {max_prob_params[2]:.3f}, Inferred σC: {max_prob_params[3]:.3f}")

        # Store results
        results[model_name] = {
            "true_params": true_params,
            "max_prob_params": max_prob_params,
            "samples": samples,
            "q_obs": q_obs,
            "color": color,
            "label": label
        }

    # Create combined plots
    if len(results) > 1:
        # Combined histogram of projections
        fig_hist_combined = plot_projected_distributions(
            q_obs_list, labels, colors,
            output_file=str(test_dir / "combined_projections_hist.png"),
            title=f"{test_name.replace('_', ' ').title()}: Projected Axis Ratios"
        )
        plt.close(fig_hist_combined)

        # Combined B/A vs C/A plot
        samples_list = [results[model_name]["samples"] for model_name in models_dict]
        max_prob_list = [results[model_name]["max_prob_params"] for model_name in models_dict]
        true_params_list = [results[model_name]["true_params"] for model_name in models_dict]

        #focus on maximum probablity:
        fig_shapes = plot_ellipsoid_shapes(
            samples_list, max_prob_list, true_params_list,
            labels, colors,
            output_file=str(test_dir / "max_prob_combined_shapes.png"),
            title=f"{test_name.replace('_', ' ').title()}: Intrinsic Shapes"
        )
        plt.close(fig_shapes)

        # Combined ellipsoid shapes
        fig_shapes_all = plot_ellipsoid_shapes(
            samples_list, max_prob_list, true_params_list,
            labels, colors,
            output_file=str(test_dir / "all_combined_shapes.png"),
            title=f"{test_name.replace('_', ' ').title()}: Intrinsic Shapes",
            focus_on_max_prob=False, show_samples=True, show_ellipses=False
        )
        plt.close(fig_shapes_all)

        # Create comparison grid
        fig_grid = plot_comparison_grid(
            results,
            output_file=str(test_dir / "comparison_grid.png"),
            title=f"{test_name.replace('_', ' ').title()}: Comparison"
        )
        plt.close(fig_grid)

        # Create statistics plot
        fig_stats = plot_statistics(
            results,
            output_file=str(test_dir / "statistics.png"),
            title=f"{test_name.replace('_', ' ').title()}: Parameter Statistics"
        )
        plt.close(fig_stats)

        # If we have exactly two models, perform statistical tests
        if len(results) == 2:
            model_names = list(results.keys())
            samples_1 = results[model_names[0]]["samples"]
            samples_2 = results[model_names[1]]["samples"]

            # B/A difference
            ks_stat_B, p_value_B = stats.ks_2samp(samples_1[:, 0], samples_2[:, 0])

            # C/A difference
            ks_stat_C, p_value_C = stats.ks_2samp(samples_1[:, 1], samples_2[:, 1])

            print("\nStatistical Tests:")
            print(f"B/A KS test: statistic={ks_stat_B:.4f}, p-value={p_value_B:.4f}")
            print(f"C/A KS test: statistic={ks_stat_C:.4f}, p-value={p_value_C:.4f}")

    # Create a summary table
    summary_data = []

    for model_name, result in results.items():
        true_params = result["true_params"]
        max_prob_params = result["max_prob_params"]
        samples = result["samples"]

        mean_params = np.mean(samples, axis=0)
        std_params = np.std(samples, axis=0)

        summary_data.append({
            "Model": model_name,
            "True B/A": true_params[0],
            "Inferred B/A": max_prob_params[0],
            "B/A Mean": mean_params[0],
            "B/A Std": std_params[0],
            "True C/A": true_params[1],
            "Inferred C/A": max_prob_params[1],
            "C/A Mean": mean_params[1],
            "C/A Std": std_params[1],
            "True σB": true_params[2],
            "Inferred σB": max_prob_params[2],
            "σB Mean": mean_params[2],
            "σB Std": std_params[2],
            "True σC": true_params[3],
            "Inferred σC": max_prob_params[3],
            "σC Mean": mean_params[3],
            "σC Std": std_params[3]
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(test_dir / "summary.csv", index=False)

    # Save all results in a single file
    with open(test_dir / f"{test_name}_all_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    return results


def main():
    """
    Main function to run all analyses.
    """
    print("3D Shape Inference from 2D Projections")
    print("======================================")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"MCMC parameters: {N_WALKERS} walkers, {N_STEPS} steps, {BURN_IN} burn-in steps, {N_CORES} cores")
    print(f"Data generation: {N_ELLIPSOIDS} ellipsoids, {N_PROJECTIONS} total projections")

    # Choose which tests to run
    run_kado_fong = True
    run_cdm_sidm = True
    run_custom = False
    force_rerun = True  # Set to True to force rerunning analyses even if results exist

    # 1. Recreate the test from Kado-Fong et al.
    if run_kado_fong:
        kado_fong_results = run_shape_analysis(
            KADO_FONG_SHAPES,
            "kado_fong_test",
            use_abc_format=True,
            force_rerun=force_rerun
        )

    # 2. Compare CDM and SIDM models
    if run_cdm_sidm:
        cdm_sidm_models = {
            "cdm": {"params": CDM_PARAMS, "color": "blue", "label": "CDM"},
            "sidm": {"params": SIDM_PARAMS, "color": "red", "label": "SIDM"}
        }

        cdm_sidm_results = run_shape_analysis(
            cdm_sidm_models,
            "cdm_sidm_comparison",
            use_abc_format=False,
            force_rerun=force_rerun
        )

    # 3. Run custom models
    if run_custom:
        # Convert CUSTOM_MODELS to the expected format
        custom_model_dict = {}
        for i, (name, params) in enumerate(CUSTOM_MODELS.items()):
            custom_model_dict[name] = {
                "params": params,
                "color": plt.cm.tab10.colors[i % 10],
                "label": name.replace('_', ' ').title()
            }

        custom_results = run_shape_analysis(
            custom_model_dict,
            "custom_models",
            use_abc_format=False,
            force_rerun=force_rerun
        )

    print("\nAll analyses completed successfully!")


if __name__ == "__main__":
    main()