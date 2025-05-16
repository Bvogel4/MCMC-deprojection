import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from galaxy_ellipse_collection import GalaxyEllipseCollection



#load tangos data
# Set environment variables
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all.db'
os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/bk639/data/CDM_z0'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
#add python path /home/bk639/MorphologyMeasurements/Code/tangos
import sys
sys.path.append('/home/bk639/mytangosproperty')
import tangos
tangos_sims = tangos.all_simulations()

# Import your existing functions
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

# Configuration parameters (added from test code)
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# MCMC parameters (added from test code)
N_STEPS = 3000  # Number of MCMC steps
BURN_IN = 300  # Number of burn-in steps to discard
N_CORES = 32  # Number of CPU cores to use for parallel processing
N_WALKERS = 64  # Number of MCMC walkers
N_ANGLES_PER_HALO = 2000  # Number of angles to sample for each halo
N_ANGLES_PER_HALO_ALL = 500  # Number of angles to sample for each halo when running all combined




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



# Enhanced run_all_individual_halos with saving/loading functionality
def run_all_individual_halos(galaxy_collection, n_angles=1000, reff_index=0, force_rerun=False):
    """Run inference on each halo individually and collect results."""
    print(f"\nRunning inference for each halo separately...")

    results = {}
    summary_data = []

    for sim_name, halo_id in galaxy_collection.get_all_halo_keys():
        print(f"Processing halo {halo_id} from simulation {sim_name}...")
        try:
            samples, max_prob_params, sampler, q_obs = galaxy_collection.run_inference_single_halo(sim_name, halo_id,
                                                        n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN,
                                                        n_cores=N_CORES,output_dir=f'results/{sim_name}/{halo_id}',
                                                        force_rerun=force_rerun, n_angles=n_angles)

            # Store results
            results[(sim_name, halo_id)] = {
                'samples': samples,
                'max_prob_params': max_prob_params,
                'q_obs': q_obs,
                'sampler': sampler
            }

            # Add to summary data for CSV
            mean_params = np.mean(samples, axis=0)
            std_params = np.std(samples, axis=0)

            summary_data.append({
                "Simulation": sim_name,
                "Halo_ID": halo_id,
                "Inferred B/A": max_prob_params[0],
                "B/A Mean": mean_params[0],
                "B/A Std": std_params[0],
                "Inferred C/A": max_prob_params[1],
                "C/A Mean": mean_params[1],
                "C/A Std": std_params[1],
                "Inferred σB": max_prob_params[2],
                "σB Mean": mean_params[2],
                "σB Std": std_params[2],
                "Inferred σC": max_prob_params[3],
                "σC Mean": mean_params[3],
                "σC Std": std_params[3]
            })

        except Exception as e:
            print(f"Error processing halo {halo_id} from {sim_name}: {e}")

    # Save summary as CSV
    summary_dir = Path("results/summary")
    summary_dir.mkdir(exist_ok=True)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_dir / "all_halos_summary.csv", index=False)

    # Save all results in a single pickle file
    with open(summary_dir / "all_individual_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    return results



# Add a new method to create a summary table of all results
def create_summary_table(results_dict, output_file="results/summary/summary_table.csv"):
    """
    Create a summary table of all results.

    Parameters:
        results_dict (dict): Dictionary of results keyed by (sim_name, halo_id)
        output_file (str): Path to output CSV file
    """
    summary_data = []

    for key, result in results_dict.items():
        # Extract simulation name and halo ID
        if isinstance(key, tuple):
            sim_name, halo_id = key
        else:
            sim_name = "Combined"
            halo_id = key

        max_prob_params = result['max_prob_params']
        samples = result['samples']

        mean_params = np.mean(samples, axis=0)
        std_params = np.std(samples, axis=0)

        # Calculate 16th, 50th, and 84th percentiles for each parameter
        percentiles = np.percentile(samples, [16, 50, 84], axis=0)

        summary_data.append({
            "Simulation": sim_name,
            "Halo_ID": halo_id,
            "B/A_max_prob": max_prob_params[0],
            "B/A_mean": mean_params[0],
            "B/A_std": std_params[0],
            "B/A_16th": percentiles[0, 0],
            "B/A_50th": percentiles[1, 0],
            "B/A_84th": percentiles[2, 0],
            "C/A_max_prob": max_prob_params[1],
            "C/A_mean": mean_params[1],
            "C/A_std": std_params[1],
            "C/A_16th": percentiles[0, 1],
            "C/A_50th": percentiles[1, 1],
            "C/A_84th": percentiles[2, 1],
            "sigmaB_max_prob": max_prob_params[2],
            "sigmaB_mean": mean_params[2],
            "sigmaB_std": std_params[2],
            "sigmaB_16th": percentiles[0, 2],
            "sigmaB_50th": percentiles[1, 2],
            "sigmaB_84th": percentiles[2, 2],
            "sigmaC_max_prob": max_prob_params[3],
            "sigmaC_mean": mean_params[3],
            "sigmaC_std": std_params[3],
            "sigmaC_16th": percentiles[0, 3],
            "sigmaC_50th": percentiles[1, 3],
            "sigmaC_84th": percentiles[2, 3]
        })

    # Create DataFrame and save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)

    return summary_df


# ======================================
# Main execution
# ======================================
if __name__ == "__main__":
    print("3D Shape Inference from Galaxy Ellipse Data")
    print("===========================================")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"MCMC parameters: {N_WALKERS} walkers, {N_STEPS} steps, {BURN_IN} burn-in steps, {N_CORES} cores,{N_ANGLES_PER_HALO} angles per halo")

    # ======================================
    # Step 1: Create a collection and add halos
    # ======================================
    galaxy_collection = GalaxyEllipseCollection()

    # Load ellipse data from pickle file
    with open('ellipse_data.pickle', 'rb') as f:
        ellipse_dict = pickle.load(f)

    # Add all halos to the collection
    for sim in ellipse_dict.keys():
        for halo_ref in ellipse_dict[sim].keys():

            #from tangos, load ba_s and ca_s
            # Get the halo object from tangos
            try:
                halo = tangos.get_halo(halo_ref)
                hid = halo_name.split('_')[1]
                # Get the properties
                reff = halo['image_reffs'][0]
                ba_s_smoothed = halo.calculate('ba_s_smoothed()')
                ca_s_smoothed = halo.calculate('ca_s_smoothed()')
                ba_s = ba_s_smoothed(2*reff)
                ca_s = ca_s_smoothed(2*reff)
                #make sure these have sane values between 0 and 1
                assert 0 <= ba_s <= 1, f"simulation {sim}, halo {hid}: ba_s out of bounds: {ba_s}"
                assert 0 <= ca_s <= 1, f"simulation {sim}, halo {hid}: ca_s out of bounds: {ca_s}"
            except Exception as e:
                print(f"Error loading halo {hid} from simulation {sim}: {e}")
                continue



            print(f"Loading halo {hid} from simulation {sim}...")
            halo_data = load_and_process_halo_data(
                sim_name=sim,
                halo_id=hid,
                pickle_filename='ellipse_data.pickle'
            )
            #add the ba_s and ca_s to the halo data
            halo_data['ba_s'] = ba_s
            halo_data['ca_s'] = ca_s


            # Add the halo to our collection
            galaxy_collection.add_halo(
                sim_name=sim,
                halo_id=hid,
                halo_data=halo_data,
                reff_multipliers=[2, 3, 4],
                interpolation_method='linear',
                coordinate_system='angles'
            )
            

    print(f"Added {galaxy_collection.get_halo_count()} halos to the collection")

    # Set force_rerun to False to use existing results if available
    force_rerun = False

    # # Run inference on all halos combined
    # all_samples, all_max_params,all_sampler, all_q_obs = galaxy_collection.run_inference_all_halos(n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN, n_cores=N_CORES,
    #                                                                                         n_angles_per_halo=N_ANGLES_PER_HALO_ALL, force_rerun=force_rerun)
    # #summary table
    # summary_table = create_summary_table(
    #     {
    #         'Combined': {
    #             'samples': all_samples,
    #             'max_prob_params': all_max_params,
    #             'q_obs': all_q_obs
    #         }
    #     },
    #     output_file="results/summary/combined_summary.csv"
    # )
    print('Running halos individually...')
    # run individual halos
    individual_results = run_all_individual_halos(galaxy_collection, n_angles=N_ANGLES_PER_HALO, force_rerun=force_rerun)
    # summary table
    individual_summary_table = create_summary_table(
        individual_results,
        output_file="results/summary/individual_summary.csv"
    )


    print("\nAnalysis completed successfully!")