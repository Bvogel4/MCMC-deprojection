"""
shape_plotting.py - Visualization functions for 3D shape inference.

This module contains functions for visualizing results from 3D shape inference,
including corner plots, ellipsoid shapes, projection distributions, and MCMC diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
import os
from shape_inference import generate_model_projections
from matplotlib.patches import Ellipse

from scipy.stats import ks_2samp


# KF observational data
KF_DATA = {
    'KF High Mass': {
        '1R_eff': {'center': (0.863, 0.297), 'std': (0.075, 0.103)},
        '2R_eff': {'center': (0.91, 0.352), 'std': (0.04, 0.155)}
    },
    'KF Medium Mass': {
        '1R_eff': {'center': (0.857, 0.323), 'std': (0.05, 0.118)},
        '2R_eff': {'center': (0.902, 0.371), 'std': (0.05, 0.163)}
    },
    'KF Low Mass': {
        '1R_eff': {'center': (0.753, 0.459), 'std': (0.087, 0.194)},
        '2R_eff': {'center': (0.778, 0.446), 'std': (0.102, 0.213)}
    }
}

def plot_corner(samples, max_prob_params, true_params=None, output_file=None, title=None):
    """
    Create a corner plot of the MCMC samples.

    Parameters:
        samples (array): MCMC samples
        max_prob_params (array): Parameters with highest probability
        true_params (array): True parameters (optional)
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot

    Returns:
        figure: Corner plot figure
    """

    def filter_outliers(samples, sigma=3.0):
        """Filter out samples that are more than sigma standard deviations from the mean."""
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        mask = np.all(np.abs(samples - means) < sigma * stds, axis=1)
        return samples[mask]

    #samples = filter_outliers(samples)

    labels = ["B/A", "C/A", r"$\sigma_B$", r"$\sigma_C$"]

    fig = corner.corner(
        samples,
        labels=labels,
        truths=true_params,
        range=[(0,1),(0,1),(0,0.5),(0,0.5)],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        hist_kwargs={"density": True},
        levels=(0.68, 0.95),  # Show 1-sigma and 2-sigma contours
        plot_datapoints=True,  # Don't plot the individual points
        fill_contours=True,  # Fill the contours
        smooth=1.0  # Apply some smoothing
    )

    # fig = corner.corner(
    #     samples,
    #     labels=labels,
    #     quantiles=[0.16, 0.5, 0.84],
    #     show_titles=True,
    #     title_kwargs={"fontsize": 12},
    #     title_fmt=".3f",
    # )

    # # Add maximum probability values
    # corner.overplot_points(fig, max_prob_params.reshape(1, -1), marker="s", color="C1", markersize=10, label="Max Probability")
    #
    # # Add true values if provided
    # if true_params is not None:
    #     corner.overplot_points(fig, np.array(true_params).reshape(1, -1), marker="*", color="r", markersize=20, label="True")
    #
    # # Add a title if provided
    # if title:
    #     plt.suptitle(title, fontsize=16, y=1.02)

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_ellipsoid_shapes(samples_list, max_prob_list, true_params_list=None, labels=None,
                          color=None, output_file=None, title=None, reference_shapes=True,
                          focus_on_max_prob=True, show_samples=False, show_ellipses=True):
    """
    Plot B/A vs C/A with focus on the maximum probability parameters.

    Parameters:
        samples_list (list): List of MCMC sample arrays
        max_prob_list (list): List of maximum probability parameters
        true_params_list (list): List of true parameters (optional)
        labels (list): List of labels for each ellipsoid shape
        color (list): List of colors for each ellipsoid shape
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot
        reference_shapes (bool): Whether to add reference shapes
        focus_on_max_prob (bool): Whether to focus on max probability point
        show_samples (bool): Whether to show the scatter of all samples
        show_ellipses (bool): Whether to show error ellipses

    Returns:
        figure: Ellipsoid shapes plot figure
    """


    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot diagonal line at B/A = C/A
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # Set default color if not provided
    if color is None:
        color = plt.cm.tab10.colors[:len(samples_list)]

    # Set default labels if not provided
    if labels is None:
        labels = [f"Shape {i + 1}" for i in range(len(samples_list))]

    # Set true_params_list to None for all if not provided
    if true_params_list is None:
        true_params_list = [None] * len(samples_list)

    # For each ellipsoid shape
    for i, (samples, max_prob, true_params, label, color) in enumerate(zip(
            samples_list, max_prob_list, true_params_list, labels, color)):
        label = label.split('.')[0]

        # Extract B/A and C/A values
        B_A_samples = samples[:, 0]
        C_A_samples = samples[:, 1]

        # Scatter plot of all samples (only if not focusing solely on max prob)
        if show_samples and not focus_on_max_prob:
            ax.scatter(B_A_samples, C_A_samples, alpha=0.01, color=color, label=None)

        # Plot true parameters if provided
        if true_params is not None:
            ax.scatter(true_params[0], true_params[1], marker='o', s=75, color='k',
                       edgecolors='white', linewidth=1.5, zorder=5)

        # Plot maximum probability parameters with emphasis
        if focus_on_max_prob:
            # Make the max probability point more prominent
            ax.scatter(max_prob[0], max_prob[1], marker='o', s=75, color=color,
                       edgecolors='white', linewidth=2)

            # Add a text annotation with the exact values
            # ax.annotate(f"B/A: {max_prob[0]:.3f}\nC/A: {max_prob[1]:.3f}",
            #             xy=(max_prob[0], max_prob[1]),
            #             xytext=(10, 10), textcoords='offset points',
            #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        else:
            # Standard display of max probability point
            ax.scatter(max_prob[0], max_prob[1], marker='o', s=75, color=color,
                       edgecolors='white', linewidth=1, label=f"{label} (Inferred)")

        if show_ellipses:

            # plot true ellipses for sigma_B and sigma_C
            if true_params is not None:
                sigma_B = true_params[2]
                sigma_C = true_params[3]
                # add ellipses to plot
                # one has a value over 0.05, and the ratio of the smallest to the largest is not too extreme (<10)
                if sigma_B > 0.05 or sigma_C > 0.05 and (sigma_B / sigma_C < 10 and sigma_C / sigma_B < 10):
                    # add ellipses to plot
                    ellipse = Ellipse(xy=(true_params[0], true_params[1]),
                                      width=2 * sigma_B, height=2 * sigma_C,
                                      edgecolor='k', facecolor='none', linestyle=':', alpha=1, linewidth=2.5)
                    ax.add_patch(ellipse)

                    # Add dummy label for true ellipse
                    ax.plot(-1, -1, c='k', linestyle=':', linewidth=2.5,
                            label=f"Intrinsic Distribution ({label})")

            # plot error ellipses from sigma_B and sigma_C output from the MCMC
            sigma_B = max_prob[2]
            sigma_C = max_prob[3]
            # only add if sigma_B and sigma_C have some physical meaning
            # one has a value over 0.05, and the ratio of the smallest to the largest is not too extreme (<10)
            if sigma_B > 0.05 or sigma_C > 0.05 and (sigma_B / sigma_C < 10 and sigma_C / sigma_B < 10):
                # add ellipses to plot
                ellipse = Ellipse(xy=(max_prob[0], max_prob[1]),
                                  width=2 * sigma_B, height=2 * sigma_C,
                                  edgecolor=color, facecolor='none', linestyle='-', alpha=1, linewidth=2.5)
                ax.add_patch(ellipse)

                # Add dummy label for inferred ellipse
                ax.plot(-1, -1, c=color, linestyle='-', linewidth=2.5,
                        label=f"Inferred Distribution ({label})")

        # Add data from KF_DATA
        # colors purple, maroon, lime green
        ell_colors = {
            'KF Low Mass': 'indigo',
            'KF Medium Mass': 'darkred',
            'KF High Mass': 'limegreen'
        }

        for mass_range in KF_DATA:
            a, b = KF_DATA[mass_range]['2R_eff']['center']
            a_std, b_std = KF_DATA[mass_range]['2R_eff']['std']
            ellipse = Ellipse(xy=(a, b), width=2 * a_std, height=2 * b_std,
                              facecolor='none', edgecolor=ell_colors[mass_range],
                              linestyle='--', linewidth=2.5)
            ax.add_patch(ellipse)

            # Add dummy label for KF_DATA ellipse
            ax.plot(-1, -1, c=ell_colors[mass_range], linestyle='--', linewidth=2.5,
                    label=f'{mass_range}')
    # Set limits and labels
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'$Q = B/A$ $(2R_{eff})$', fontsize=28)
    ax.set_ylabel(r'$S = C/A$ $(2R_{eff})$', fontsize=28)
    ax.tick_params(which='both', labelsize=18)

    #make sure plot is square
    ax.set_aspect('equal', adjustable='box')

    # Add title if provided
    # if title:
    #     if focus_on_max_prob:
    #         ax.set_title(f"{title} - Maximum Likelihood Solution", fontsize=16)
    #     else:
    #         ax.set_title(title, fontsize=16)

    # Add legend (adjust position as needed)
    ax.legend(loc='upper left', ncol=1, fontsize=20)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        if focus_on_max_prob and output_file:
            # Modify the output filename to indicate focus on max prob
            output_base, output_ext = os.path.splitext(output_file)
            output_file = f"{output_base}_max_likelihood{output_ext}"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_projected_distributions(q_obs_list, labels=None, colors=None, bin_width=0.04,
                                 output_file=None, title=None, kde=True):
    """
    Plot histograms of projected axis ratios for multiple distributions.

    Parameters:
        q_obs_list (list): List of observed projected axis ratio arrays
        labels (list): List of labels for each distribution
        colors (list): List of colors for each distribution
        bin_width (float): Width of histogram bins
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot
        kde (bool): Whether to plot kernel density estimate

    Returns:
        figure: Histogram plot figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set default colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors[:len(q_obs_list)]

    # Set default labels if not provided
    if labels is None:
        labels = [f"Distribution {i + 1}" for i in range(len(q_obs_list))]

    # Calculate bin edges
    bins = np.arange(0, 1.01, bin_width)

    # Plot histogram for each distribution
    for i, (q_obs, label, color) in enumerate(zip(q_obs_list, labels, colors)):
        ax.hist(q_obs, bins=bins, alpha=0.6, color=color, label=label, density=True, histtype='step', linewidth=2)

        # Add kernel density estimate
        if kde and len(q_obs) > 100:  # Only add KDE if enough points
            x_grid = np.linspace(0, 1, 500)
            kde_obj = gaussian_kde(q_obs)
            y_kde = kde_obj(x_grid)
            ax.plot(x_grid, y_kde, color=color, linestyle='-', linewidth=2)

    # Set labels and title
    ax.set_xlabel('Projected Axis Ratio (q = b/a)', fontsize=28)
    ax.set_ylabel('Density', fontsize=28)

    # if title:
    #     ax.set_title(title, fontsize=16)

    # Add legend
    ax.legend(fontsize=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig




def plot_projected_distributions_with_model(q_obs_list, model_params_list=None, true_params_list=None,
                                            labels=None, colors=None, bin_width=0.04,
                                            output_file=None, title=None, kde=False,
                                            model_samples=10000):
    """
    Plot histograms of projected axis ratios for multiple distributions with model and true distribution overlays.

    Parameters:
        q_obs_list (list): List of observed projected axis ratio arrays
        model_params_list (list): List of model parameters [mu_B, mu_C, sigma_B, sigma_C] for each distribution
        true_params_list (list): List of true/intrinsic parameters [B, C] for each distribution
        labels (list): List of labels for each distribution
        colors (list): List of colors for each distribution
        bin_width (float): Width of histogram bins
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot
        kde (bool): Whether to plot kernel density estimate
        model_samples (int): Number of samples to generate for model and true distributions

    Returns:
        figure: Histogram plot figure
    """

    fig, ax = plt.subplots(figsize=(10, 7))  # Slightly larger to accommodate KS test results

    # Set default colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors[:len(q_obs_list)]

    # Set default labels if not provided
    if labels is None:
        labels = [f"Distribution {i + 1}" for i in range(len(q_obs_list))]

    # Calculate bin edges
    bins = np.arange(0, 1.01, bin_width)
    true_color = 'red'

    # Store KS test results for plotting
    ks_results_text = []

    # Plot histogram for each distribution
    for i, (q_obs, label, color) in enumerate(zip(q_obs_list, labels, colors)):
        if color == 'red':
            color = 'purple'

        # Plot observed data
        ax.hist(q_obs, bins=bins, alpha=1, color=color, label=f"{label} (Observed)",
                density=True, histtype='step', linewidth=2.5)

        # Add kernel density estimate for observed data
        if kde and len(q_obs) > 100:
            x_grid = np.linspace(0, 1, 500)
            kde_obj = gaussian_kde(q_obs)
            y_kde = kde_obj(x_grid)
            ax.plot(x_grid, y_kde, color=color, linestyle='--', linewidth=2, alpha=1)

        # Initialize variables for KS tests
        q_model = None
        q_true = None

        # Plot model if parameters are provided
        if model_params_list is not None and i < len(model_params_list):
            if model_params_list[i] is not None:
                # Generate model projections
                q_model = generate_model_projections(model_params_list[i], model_samples)

                # Plot model histogram
                ax.hist(q_model, bins=bins, alpha=1, color='k',
                        label=f"{label} (Model)", density=True, histtype='step',
                        linewidth=2.5, linestyle='-')

                # Add KDE for model if requested
                if kde and len(q_model) > 100:
                    kde_model = gaussian_kde(q_model)
                    y_kde_model = kde_model(x_grid)
                    ax.plot(x_grid, y_kde_model, color='k', linestyle='--',
                            linewidth=2, alpha=1)

        if true_params_list is not None and i < len(true_params_list):
            if true_params_list[i] is not None:
                # Generate true projections from intrinsic shape
                B_true, C_true, B_err, C_err = true_params_list[i]
                q_true = generate_model_projections([B_true, C_true, B_err, C_err], model_samples)

                # Plot true distribution histogram
                ax.hist(q_true, bins=bins, alpha=1, color=true_color,
                        label=f"{label} (True)", density=True, histtype='step',
                        linewidth=2.5, linestyle='-')

                # Add KDE for true distribution if requested
                if kde and len(q_true) > 100:
                    kde_true = gaussian_kde(q_true)
                    y_kde_true = kde_true(x_grid)
                    ax.plot(x_grid, y_kde_true, color=true_color, linestyle=':',
                            linewidth=2, alpha=1)

        # Perform KS tests between distribution pairs
        print(f"\n{'=' * 60}")
        print(f"KS Test Results for {label}")
        print('=' * 60)

        ks_text_lines = [f"{label}:"]

        # Observed vs Model
        if q_model is not None:
            ks_stat_obs_model, p_val_obs_model = ks_2samp(q_obs, q_model)
            print(f"Observed vs Model:")
            print(f"  KS statistic: {ks_stat_obs_model:.4f}")
            print(f"  p-value: {p_val_obs_model:.4e}")
            print(
                f"  Interpretation: {'Distributions are similar (fail to reject H0)' if p_val_obs_model > 0.05 else 'Distributions are different (reject H0)'}")
            ks_text_lines.append(f"  Obs vs Model: D={ks_stat_obs_model:.3f}, p={p_val_obs_model:.2e}")

        # Observed vs True
        if q_true is not None:
            ks_stat_obs_true, p_val_obs_true = ks_2samp(q_obs, q_true)
            print(f"\nObserved vs True:")
            print(f"  KS statistic: {ks_stat_obs_true:.4f}")
            print(f"  p-value: {p_val_obs_true:.4e}")
            print(
                f"  Interpretation: {'Distributions are similar (fail to reject H0)' if p_val_obs_true > 0.05 else 'Distributions are different (reject H0)'}")
            ks_text_lines.append(f"  Obs vs True: D={ks_stat_obs_true:.3f}, p={p_val_obs_true:.2e}")

        # Model vs True
        if q_model is not None and q_true is not None:
            ks_stat_model_true, p_val_model_true = ks_2samp(q_model, q_true)
            print(f"\nModel vs True:")
            print(f"  KS statistic: {ks_stat_model_true:.4f}")
            print(f"  p-value: {p_val_model_true:.4e}")
            print(
                f"  Interpretation: {'Distributions are similar (fail to reject H0)' if p_val_model_true > 0.05 else 'Distributions are different (reject H0)'}")
            ks_text_lines.append(f"  Model vs True: D={ks_stat_model_true:.3f}, p={p_val_model_true:.2e}")

        if len(ks_text_lines) > 1:  # Only add if we have actual results
            ks_results_text.extend(ks_text_lines)

    # Set labels and title
    ax.set_xlabel(r'Projected Axis Ratio ($q = b/a$)', fontsize=30)
    ax.set_ylabel('Density', fontsize=30)

    # Make axis ticks larger
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Add legend
    ax.legend(fontsize=16, loc='upper left')

    # Add KS test results as text box on the plot
    if ks_results_text:
        ks_text = '\n'.join(ks_results_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.98, ks_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig

def plot_chain_evolution(sampler, burn_in=None, output_file=None, title=None):
    """
    Plot the evolution of the MCMC chain to assess convergence.

    Parameters:
        sampler: emcee sampler object or chain array
        burn_in (int): Number of burn-in steps to mark
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot

    Returns:
        figure: Chain evolution plot figure
    """
    # Extract the chain
    if hasattr(sampler, 'get_chain'):
        chain = sampler.get_chain()
    else:
        # Assume sampler is already the chain
        chain = sampler

    n_steps, n_walkers, n_dim = chain.shape

    # Create the figure
    fig, axes = plt.subplots(n_dim, figsize=(12, 9), sharex=True)

    # Parameter labels
    labels = ["B/A", "C/A", r"$\sigma_B$", r"$\sigma_C$"]

    # Plot each parameter
    for i in range(n_dim):
        ax = axes[i]

        # Plot all walkers
        for j in range(n_walkers):
            ax.plot(chain[:, j, i], alpha=0.1, color='k')

        # Plot the median of all walkers
        ax.plot(np.median(chain[:, :, i], axis=1), color='C0', linewidth=2)

        # Add burn-in line if provided
        if burn_in is not None:
            ax.axvline(burn_in, color='r', linestyle='--', alpha=0.5)

        # Add labels
        ax.set_ylabel(labels[i])

        # Add grid
        ax.grid(True, alpha=0.3)

    # Add labels to the bottom plot
    axes[-1].set_xlabel("Step Number")

    # Add title if provided
    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_comparison_grid(results_dict, output_file=None, title=None):
    """
    Create a comprehensive comparison grid of multiple shape distributions.

    Parameters:
        results_dict (dict): Dictionary mapping shape names to result dictionaries
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot

    Returns:
        figure: Comparison grid figure
    """
    n_shapes = len(results_dict)

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # Create layout for comparison plot
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

    # Plot 1: Projected axis ratio distributions
    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare data for plotting
    q_obs_list = []
    labels = []
    colors = plt.cm.tab10.colors[:n_shapes]

    for i, (shape_name, results) in enumerate(results_dict.items()):
        q_obs_list.append(results['q_obs'])
        labels.append(shape_name)

    # Plot histograms
    bins = np.arange(0, 1.01, 0.04)
    for i, (q_obs, label, color) in enumerate(zip(q_obs_list, labels, colors)):
        ax1.hist(q_obs, bins=bins, alpha=0.6, color=color, label=label, density=True, histtype='step', linewidth=2)

        # Add kernel density estimate
        x_grid = np.linspace(0, 1, 500)
        kde = gaussian_kde(q_obs)
        y_kde = kde(x_grid)
        ax1.plot(x_grid, y_kde, color=color, linestyle='-', linewidth=2)

    ax1.set_xlabel('Projected Axis Ratio (q = b/a)')
    ax1.set_ylabel('Density')
    ax1.set_title('Projected Axis Ratio Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: B/A vs C/A with error ellipses
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot diagonal line
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # Prepare data for plotting
    samples_list = []
    max_prob_list = []
    true_params_list = []

    for shape_name, results in results_dict.items():
        samples_list.append(results['samples'])
        max_prob_list.append(results['max_prob_params'])
        if 'true_params' in results:
            true_params_list.append(results['true_params'])
        else:
            true_params_list.append(None)

    # Plot each distribution
    for i, (samples, max_prob, true_params, label, color) in enumerate(zip(
            samples_list, max_prob_list, true_params_list, labels, colors)):

        # Extract B/A and C/A values
        B_A_samples = samples[:, 0]
        C_A_samples = samples[:, 1]

        # Plot samples
        ax2.scatter(B_A_samples, C_A_samples, alpha=0.01, color=color)

        # Plot true parameters if available
        if true_params is not None:
            ax2.scatter(true_params[0], true_params[1], marker='*', s=200, color=color,
                        edgecolors='black', linewidth=1.5, label=f"{label} (True)")

        # Plot maximum probability parameters
        ax2.scatter(max_prob[0], max_prob[1], marker='o', s=100, color=color,
                    edgecolors='black', linewidth=1, label=f"{label}")

        # Add error ellipses
        cov = np.cov(B_A_samples, C_A_samples)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Calculate the angle of the largest eigenvector
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        # Plot 1-sigma and 2-sigma ellipses
        for n_sigma in [1, 2]:
            ellipse = Ellipse(
                xy=(np.mean(B_A_samples), np.mean(C_A_samples)),
                width=2 * n_sigma * np.sqrt(eigvals[0]),
                height=2 * n_sigma * np.sqrt(eigvals[1]),
                angle=angle,
                facecolor='none',
                edgecolor=color,
                alpha=0.8,
                linewidth=2 if n_sigma == 1 else 1,
                linestyle='-' if n_sigma == 1 else '--',
            )
            ax2.add_patch(ellipse)

    # Add reference shapes
    ax2.scatter(0.9, 0.1, marker='d', s=100, color='blue', edgecolor='black', label='Disk')
    ax2.scatter(0.9, 0.9, marker='d', s=100, color='red', edgecolor='black', label='Spheroid')
    ax2.scatter(0.1, 0.1, marker='d', s=100, color='green', edgecolor='black', label='Prolate')

    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('B/A')
    ax2.set_ylabel('C/A')
    ax2.set_title('Intrinsic Shape Distributions')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter comparison table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('tight')
    ax3.axis('off')

    # Create table data
    table_data = []
    table_cols = ['Shape', 'B/A (True)', 'B/A (Inferred)', 'C/A (True)', 'C/A (Inferred)',
                  'σB (True)', 'σB (Inferred)', 'σC (True)', 'σC (Inferred)']

    for shape_name, results in results_dict.items():
        max_prob = results['max_prob_params']
        means = np.mean(results['samples'], axis=0)
        stds = np.std(results['samples'], axis=0)

        row = [shape_name]

        # Add true values if available
        if 'true_params' in results:
            true_params = results['true_params']
            row.extend([
                f"{true_params[0]:.3f}",
                f"{max_prob[0]:.3f} ± {stds[0]:.3f}",
                f"{true_params[1]:.3f}",
                f"{max_prob[1]:.3f} ± {stds[1]:.3f}",
                f"{true_params[2]:.3f}",
                f"{max_prob[2]:.3f} ± {stds[2]:.3f}",
                f"{true_params[3]:.3f}",
                f"{max_prob[3]:.3f} ± {stds[3]:.3f}"
            ])
        else:
            row.extend([
                "N/A",
                f"{max_prob[0]:.3f} ± {stds[0]:.3f}",
                "N/A",
                f"{max_prob[1]:.3f} ± {stds[1]:.3f}",
                "N/A",
                f"{max_prob[2]:.3f} ± {stds[2]:.3f}",
                "N/A",
                f"{max_prob[3]:.3f} ± {stds[3]:.3f}"
            ])

        table_data.append(row)

    # Create table
    table = ax3.table(cellText=table_data, colLabels=table_cols, loc='center', cellLoc='center')

    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Adjust column widths
    for i in range(len(table_cols)):
        table.auto_set_column_width(i)

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def plot_statistics(results_dict, output_file=None, title=None):
    """
    Plot statistical measures of the inference results.

    Parameters:
        results_dict (dict): Dictionary mapping shape names to result dictionaries
        output_file (str): Path to save the plot (optional)
        title (str): Title for the plot

    Returns:
        figure: Statistics plot figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Parameter names
    param_names = ['B/A', 'C/A', r'$\sigma_B$', r'$\sigma_C$']

    # Colors for each shape
    colors = plt.cm.tab10.colors[:len(results_dict)]

    # For each parameter
    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # For each shape
        x_pos = []
        y_vals = []
        y_errs = []
        labels = []
        true_vals = []

        for j, (shape_name, results) in enumerate(results_dict.items()):
            samples = results['samples']
            mean_val = np.mean(samples[:, i])
            std_val = np.std(samples[:, i])

            x_pos.append(j)
            y_vals.append(mean_val)
            y_errs.append(std_val)
            labels.append(shape_name)

            # Add true value if available
            if 'true_params' in results:
                true_vals.append(results['true_params'][i])
            else:
                true_vals.append(None)

        # Plot bar chart
        bars = ax.bar(x_pos, y_vals, yerr=y_errs, alpha=0.7, capsize=5,
                      color=colors[:len(x_pos)], tick_label=labels)

        # Add true values as horizontal lines
        for j, true_val in enumerate(true_vals):
            if true_val is not None:
                ax.axhline(y=true_val, xmin=j / len(x_pos) - 0.5 / len(x_pos),
                           xmax=(j + 1) / len(x_pos) - 0.5 / len(x_pos),
                           color='r', linestyle='--', linewidth=2)

                # Add text annotation
                ax.text(j, true_val, f"True: {true_val:.3f}",
                        ha='center', va='bottom', rotation=0, fontsize=8)

        # Set labels and title
        ax.set_xlabel('Shape')
        ax.set_ylabel(f'{param_name} Value')
        ax.set_title(f'{param_name} Parameter')

        # Add grid
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val, err in zip(bars, y_vals, y_errs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                    f"{val:.3f}±{err:.3f}", ha='center', va='bottom', rotation=90, fontsize=8)

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


