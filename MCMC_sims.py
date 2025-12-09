import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
from galaxy_ellipse_collection import GalaxyEllipseCollection

# load tangos data
# Set environment variables
# Set environment variables
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data/test_dbs/FIRE_test.db'
#os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/bk639/data/CDM_z0'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
#add python path /home/bk639/MorphologyMeasurements/Code/tangos
import sys
sys.path.append('/home/bk639/FIRE_analsis_tools/')
import tangos

tangos_sims = tangos.all_simulations()

band = 'v'
image_type = 'stars'
radius = 2.0  # in Reff

# Configuration parameters (added from test code)
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# MCMC parameters (added from test code)
N_STEPS = 3000  # Number of MCMC steps
BURN_IN = 300 # Number of burn-in steps to discard
N_CORES = 32  # Number of CPU cores to use for parallel processing
N_WALKERS = 64  # Number of MCMC walkers
N_ANGLES_PER_HALO = 2000  # Number of angles to sample for each halo
N_ANGLES_PER_HALO_ALL = 5000 # Number of angles to sample for each halo when running all combined

# Set force_rerun to False to use existing results if available
force_rerun = True


# Disky/Non-disky classification thresholds
BA_THRESHOLD = 0.65
CA_THRESHOLD = 0.4


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


def run_all_individual_halos(galaxy_collection, n_angles=1000, reff_index=0, force_rerun=False, output_suffix=""):
    """Run inference on each halo individually and collect results."""
    print(f"\nRunning inference for each halo separately{' (' + output_suffix + ')' if output_suffix else ''}...")

    results = {}
    summary_data = []

    # Save summary as CSV
    summary_dir = Path("results/summary")
    summary_dir.mkdir(exist_ok=True)

    for sim_name, halo_id in galaxy_collection.get_all_halo_keys():
        print(f"Processing halo {halo_id} from simulation {sim_name}...")
        try:
            samples, max_prob_params, sampler, q_obs = galaxy_collection.run_inference_single_halo(
                sim_name, halo_id,
                n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN,
                n_cores=N_CORES,
                output_dir=f'results/single_halos/{sim_name}_{halo_id}_{reff_index}{("_" + output_suffix) if output_suffix else ""}',
                force_rerun=force_rerun, n_angles=n_angles
            )

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

            # Save the results to a file
            output_filename = f"all_individual_results_3Reff{('_' + output_suffix) if output_suffix else ''}.pkl"
            with open(summary_dir / output_filename, 'wb') as f:
                pickle.dump(results, f)

        except Exception as e:
            print(f"Error processing halo {halo_id} from {sim_name}: {e}")

    summary_df = pd.DataFrame(summary_data)
    csv_filename = f"all_halos_summary{('_' + output_suffix) if output_suffix else ''}.csv"
    summary_df.to_csv(summary_dir / csv_filename, index=False)

    return results


def create_summary_table(results_dict, output_file="results/summary/summary_table.csv"):
    """Create a summary table of all results."""
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
    print(
        f"MCMC parameters: {N_WALKERS} walkers, {N_STEPS} steps, {BURN_IN} burn-in steps, {N_CORES} cores, {N_ANGLES_PER_HALO} angles per halo")

    # ======================================
    # Step 1: Create collections for all, disky, and non-disky galaxies
    # ======================================
    galaxy_collection = GalaxyEllipseCollection()
    disky_collection = GalaxyEllipseCollection()
    non_disky_collection = GalaxyEllipseCollection()

    # Track statistics
    disky_count = 0
    non_disky_count = 0
    skipped_count = 0

    # Load ellipse data from pickle file
    with open('ellipse_data.pickle', 'rb') as f:
        ellipse_dict = pickle.load(f)

    # Add all halos to the appropriate collections
    for sim in ellipse_dict.keys():
        for halo_ref in ellipse_dict[sim].keys():

            # From tangos, load ba_s and ca_s
            try:
                halo = tangos.get_halo(halo_ref)
                hid = halo.basename.split('_')[1]
                # Get the properties
                reff = halo[f'image_reffs_{band}'][0]
                ba_s_smoothed = halo.calculate('ba_s_smoothed()')
                ca_s_smoothed = halo.calculate('ca_s_smoothed()')
                ba_s = ba_s_smoothed(2 * reff)
                ca_s = ca_s_smoothed(2 * reff)
                # Make sure these have sane values between 0 and 1
                assert 0 <= ba_s <= 1, f"simulation {sim}, halo {hid}: ba_s out of bounds: {ba_s}"
                assert 0 <= ca_s <= 1, f"simulation {sim}, halo {hid}: ca_s out of bounds: {ca_s}"
            except Exception as e:
                print(f"Error loading halo {hid} from simulation {sim}: {e}")
                traceback.print_exc()
                skipped_count += 1
                continue

            print(f"Loading halo {hid} from simulation {sim} (ba_s={ba_s:.3f}, ca_s={ca_s:.3f})...")

            halo_data = load_and_process_halo_data(
                sim_name=sim,
                halo_id=halo_ref,
                pickle_filename=f'ellipse_data_{band}_{image_type}.pickle'
            )
            # Add the ba_s and ca_s to the halo data
            halo_data['ba_s'] = ba_s
            halo_data['ca_s'] = ca_s

            # Add the halo to the main collection
            galaxy_collection.add_halo(
                sim_name=sim,
                halo_id=hid,
                halo_data=halo_data,
                reff_multipliers=[radius],
                interpolation_method='linear',
                coordinate_system='angles'
            )

            # Classify and add to appropriate collection
            if ba_s > BA_THRESHOLD and ca_s < CA_THRESHOLD:
                # Disky galaxy
                disky_collection.add_halo(
                    sim_name=sim,
                    halo_id=hid,
                    halo_data=halo_data,
                    reff_multipliers=[radius],
                    interpolation_method='linear',
                    coordinate_system='angles'
                )
                disky_count += 1
                print(f"  → Classified as DISKY")
            else:
                # Non-disky galaxy
                non_disky_collection.add_halo(
                    sim_name=sim,
                    halo_id=hid,
                    halo_data=halo_data,
                    reff_multipliers=[radius],
                    interpolation_method='linear',
                    coordinate_system='angles'
                )
                non_disky_count += 1
                print(f"  → Classified as NON-DISKY")

    print(f"\nCollection Summary:")
    print(f"Total halos added: {galaxy_collection.get_halo_count()}")
    print(f"Disky galaxies (BA > {BA_THRESHOLD}, CA > {CA_THRESHOLD}): {disky_count}")
    print(f"Non-disky galaxies: {non_disky_count}")
    print(f"Skipped (errors): {skipped_count}")

    # ======================================
    # Step 2: Run inference on all halos (original analysis)
    # ======================================

    # print('\n' + '=' * 50)
    # print('Running inference on ALL halos individually...')
    # individual_results = run_all_individual_halos(
    #     galaxy_collection,
    #     n_angles=N_ANGLES_PER_HALO,
    #     force_rerun=force_rerun,
    #     reff_index=1,
    #     output_suffix="all"
    # )
    #
    # individual_summary_table = create_summary_table(
    #     individual_results,
    #     output_file=f"results/summary/individual_summary_{radius}_{band}_{image_type}reff_all.csv"
    # )
    # 
    print('\nRunning inference on ALL halos combined...')
    all_samples, all_max_params, all_sampler, all_q_obs = galaxy_collection.run_inference_all_halos(
        n_steps=N_STEPS,
        n_walkers=N_WALKERS,
        burn_in=BURN_IN,
        n_cores=N_CORES,
        n_angles_per_halo=N_ANGLES_PER_HALO_ALL,
        force_rerun=force_rerun,
        output_dir='results/combined_all',
        label = 'All galaxies',
        color = 'green'
    )

    summary_table = create_summary_table(
        {'Combined_All': {
            'samples': all_samples,
            'max_prob_params': all_max_params,
            'q_obs': all_q_obs
        }},
        output_file="results/summary/combined_summary_all.csv"
    )

    # ======================================
    # Step 3: Run inference on disky galaxies
    # ======================================
    if disky_count > 0:
        print('\n' + '=' * 50)
        print(f'Running inference on DISKY galaxies ({disky_count} halos)...')

        # Run on all disky galaxies combined
        disky_samples, disky_max_params, disky_sampler, disky_q_obs = disky_collection.run_inference_all_halos(
            n_steps=N_STEPS,
            n_walkers=N_WALKERS,
            burn_in=BURN_IN,
            n_cores=N_CORES,
            n_angles_per_halo=N_ANGLES_PER_HALO_ALL,
            force_rerun=force_rerun,
            output_dir='results/combined_disky',
            label = 'Disky',
            color = 'blue'
        )

        disky_summary_table = create_summary_table(
            {'Combined_Disky': {
                'samples': disky_samples,
                'max_prob_params': disky_max_params,
                'q_obs': disky_q_obs
            }},
            output_file="results/summary/combined_summary_disky.csv"
        )

        print(f"Disky galaxies analysis complete!")
    else:
        print(f"\nNo disky galaxies found with BA > {BA_THRESHOLD} and CA > {CA_THRESHOLD}")

    # ======================================
    # Step 4: Run inference on non-disky galaxies
    # ======================================
    if non_disky_count > 0:
        print('\n' + '=' * 50)
        print(f'Running inference on NON-DISKY galaxies ({non_disky_count} halos)...')

        # Run on all non-disky galaxies combined
        non_disky_samples, non_disky_max_params, non_disky_sampler, non_disky_q_obs = non_disky_collection.run_inference_all_halos(
            n_steps=N_STEPS,
            n_walkers=N_WALKERS,
            burn_in=BURN_IN,
            n_cores=N_CORES,
            n_angles_per_halo=N_ANGLES_PER_HALO_ALL,
            force_rerun=force_rerun,
            output_dir='results/combined_non_disky',
            label = 'Nondisky',
            color = 'red'
        )

        non_disky_summary_table = create_summary_table(
            {'Combined_Non_Disky': {
                'samples': non_disky_samples,
                'max_prob_params': non_disky_max_params,
                'q_obs': non_disky_q_obs
            }},
            output_file="results/summary/combined_summary_non_disky.csv"
        )

        print(f"Non-disky galaxies analysis complete!")
    else:
        print(f"\nNo non-disky galaxies found")

    # ======================================
    # Step 5: Create comparison summary
    # ======================================
    print('\n' + '=' * 50)
    print('Creating comparison summary...')

    comparison_data = []

    # Add results for all galaxies
    comparison_data.append({
        'Category': 'All Galaxies',
        'N_Halos': galaxy_collection.get_halo_count(),
        'B/A_max_prob': all_max_params[0],
        'C/A_max_prob': all_max_params[1],
        'sigmaB_max_prob': all_max_params[2],
        'sigmaC_max_prob': all_max_params[3],
        'B/A_mean': np.mean(all_samples[:, 0]),
        'C/A_mean': np.mean(all_samples[:, 1]),
        'sigmaB_mean': np.mean(all_samples[:, 2]),
        'sigmaC_mean': np.mean(all_samples[:, 3])
    })

    # Add results for disky galaxies
    if disky_count > 0:
        comparison_data.append({
            'Category': 'Disky Galaxies',
            'N_Halos': disky_count,
            'B/A_max_prob': disky_max_params[0],
            'C/A_max_prob': disky_max_params[1],
            'sigmaB_max_prob': disky_max_params[2],
            'sigmaC_max_prob': disky_max_params[3],
            'B/A_mean': np.mean(disky_samples[:, 0]),
            'C/A_mean': np.mean(disky_samples[:, 1]),
            'sigmaB_mean': np.mean(disky_samples[:, 2]),
            'sigmaC_mean': np.mean(disky_samples[:, 3])
        })

    # Add results for non-disky galaxies
    if non_disky_count > 0:
        comparison_data.append({
            'Category': 'Non-Disky Galaxies',
            'N_Halos': non_disky_count,
            'B/A_max_prob': non_disky_max_params[0],
            'C/A_max_prob': non_disky_max_params[1],
            'sigmaB_max_prob': non_disky_max_params[2],
            'sigmaC_max_prob': non_disky_max_params[3],
            'B/A_mean': np.mean(non_disky_samples[:, 0]),
            'C/A_mean': np.mean(non_disky_samples[:, 1]),
            'sigmaB_mean': np.mean(non_disky_samples[:, 2]),
            'sigmaC_mean': np.mean(non_disky_samples[:, 3])
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("results/summary/category_comparison.csv", index=False)

    print("\nComparison Summary:")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("Analysis completed successfully!")
    print(f"Results saved in results/summary/")