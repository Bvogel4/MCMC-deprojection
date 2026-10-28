import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
from galaxy_ellipse_collection import GalaxyEllipseCollection, SyntheticEllipseCollection,get_keys
from glob import glob

#load configs
import sys
from config import db_connection, sys_path, results_output_directory, pickle_file
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all_shapes.db'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos

tangos_sims = tangos.all_simulations()

band_key = '' #use none for mass based stellar shape
# image_type = 'stars'
# radius_list = [2,3,4] # in Reff
# indices = [0,1,2]
radius_list = [2] # in Reff
indices = [0]

# Configuration parameters (added from test code)
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# MCMC parameters (added from test code)
N_STEPS = 3000 # Number of MCMC steps
BURN_IN = int(N_STEPS*3/4) # Number of burn-in steps to discard
N_CORES = 32 # Number of CPU cores to use for parallel processing
N_WALKERS = 64  # Number of MCMC walkers
N_ANGLES_PER_HALO = 1000  # Number of angles to sample for each halo
N_ANGLES_PER_HALO_ALL = 200 # Number of angles to sample for each halo when running all combined
N_BOOTSTRAP = 1000

# Set force_rerun to False to use existing results if available
test = False
force_rerun = False
Klein_comparison = False
RUN_CONVARIANCE=True
CONTAIN=False

# Disky/Non-disky classification thresholds
BA_THRESHOLD = 0.65
CA_THRESHOLD = 0.4

MASS_BINS = {
    'KF_low': (7.0, 8.5),
    'KF_med': (8.5, 9.0),
    'KF_high': (9.0, 9.6)
}

# --- Kado-Fong et al. test shapes -------------------------------------------

KADO_FONG_SHAPES = {
    # "disk": {"params": (1.0, 0.9, 0.1), "color": "blue", "label": "Disk"},
    # "spheroid": {"params": (1.0, 0.95, 0.9), "color": "red", "label": "Spheroid"},
    # "prolate": {"params": (1.0, 0.15, 0.1), "color": "green", "label": "Prolate"},
    "Triaxial": {"params": (1.0, 0.8, 0.44), "color": "purple", "label": "Triaxial"},
}


def kado_fong_param_vector(shape_key, sigma_ba=0.13, sigma_ca=0.17,
                           sigma_a=None, rho_ac=None, mu_a=None):
    """
    Convert a (A, B, C) triple into a 4- or 7-parameter vector.

    4-param:  (mu_ba, mu_ca, sigma_ba, sigma_ca)
    7-param:  (mu_a, mu_ba, mu_ca, sigma_a, sigma_ba, sigma_ca, rho_ac)

    The 7-parameter form is returned if sigma_a is given.
    """
    A, B, C = KADO_FONG_SHAPES[shape_key]["params"]
    mu_ba, mu_ca = B / A, C / A

    if sigma_a is None:
        return np.array([mu_ba, mu_ca, sigma_ba, sigma_ca])

    if mu_a is None:
        mu_a = A
    if rho_ac is None:
        rho_ac = 0.0
    return np.array([mu_a, mu_ba, mu_ca, sigma_a, sigma_ba, sigma_ca, rho_ac])


def nan_func(x):
    return np.nan
def smooth_shape(rbins, a, b, c, k=3):
    """
    Smooth and filter data, handling a few NaN values gracefully.

    Parameters:
    rbins, a, b, c: array-like, input data (NaNs are assumed aligned across a, b, c)
    k: int, degree of the smoothing spline (default 3, recommended cubic)

    Returns:
    rbins, a, b, c: filtered arrays
    a_s, b_s, c_s: smoothed spline functions (or nan_func if insufficient data)
    """
    import numpy as np
    from scipy.interpolate import splrep, splev

    s_factor = 1
    d = 5  # outlier threshold in std devs
    min_points = max(k + 1, 3)

    nan_result = (rbins[:0], a[:0], b[:0], c[:0],
                  nan_func, nan_func, nan_func)

    def _fit(rb, arrs):
        """Fit a spline to each array in arrs, sharing the same domain bounds."""
        xb, xe = rb[0], rb[-1]
        tcks = [splrep(rb, arr, k=k, s=s_factor * len(rb), xb=xb, xe=xe) for arr in arrs]
        return tcks, xb, xe

    def _make_func(tck, xb, xe):
        def f(x):
            x = np.asarray(x)
            x_clipped = np.clip(x, xb, xe)
            return splev(x_clipped, tck)
        return f

    # Remove rows where any of a, b, c is NaN
    mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c)
    rbins_f, a_f, b_f, c_f = rbins[mask], a[mask], b[mask], c[mask]

    if len(rbins_f) < min_points:
        return nan_result

    # Initial splines
    tcks, xb, xe = _fit(rbins_f, [a_f, b_f, c_f])

    for i in range(3):
        arr_f = (a_f, b_f, c_f)[i]          # rebuilt from current bindings each pass
        residuals = arr_f - splev(rbins_f, tcks[i])
        std = np.std(residuals)
        if std > 0:
            keep = np.abs(residuals) < d * std
            rbins_f, a_f, b_f, c_f = rbins_f[keep], a_f[keep], b_f[keep], c_f[keep]

        if len(rbins_f) < min_points:
            return rbins_f, a_f, b_f, c_f, nan_func, nan_func, nan_func

        tcks, xb, xe = _fit(rbins_f, [a_f, b_f, c_f])

    # Remove large gaps
    diff = np.diff(rbins_f, prepend=0)
    gap_mask = diff > 1
    rbins_f, a_f, b_f, c_f = rbins_f[~gap_mask], a_f[~gap_mask], b_f[~gap_mask], c_f[~gap_mask]

    if len(rbins_f) < min_points:
        return rbins_f, a_f, b_f, c_f, nan_func, nan_func, nan_func

    # Final spline creation
    tcks, xb, xe = _fit(rbins_f, [a_f, b_f, c_f])
    a_s_func, b_s_func, c_s_func = (_make_func(tck, xb, xe) for tck in tcks)

    return rbins_f, a_f, b_f, c_f, a_s_func, b_s_func, c_s_func

def extract_single_value(value):
    """Extract scalar value from potentially nested data structures"""
    if isinstance(value, list):
        return value[0]
    return float(value)

# Loading function remains the same
def load_and_process_halo_data(pickle_filename, sim_name=None, halo_id=None):
    """Load ellipse data from the pickle file for a specific halo."""
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)
    # 
    # if sim_name is None:
    #     sim_name = list(ellipse_dict.keys())[0]
    # if sim_name not in ellipse_dict:
    #     raise ValueError(f"Simulation '{sim_name}' not found")
    #
    # if halo_id is None:
    #     halo_id = list(ellipse_dict[sim_name].keys())[0]
    # if halo_id not in ellipse_dict[sim_name]:
    #     raise ValueError(f"Halo ID {halo_id} not found")

    return ellipse_dict[sim_name][halo_id]




def run_all_individual_halos(galaxy_collection, n_angles=1000, force_rerun=False, output_suffix=""):
    """Run inference on each halo individually and collect results."""
    print(f"\nRunning inference for each halo separately{' (' + output_suffix + ')' if output_suffix else ''}...")

    results = {}
    summary_data = []
    radius=galaxy_collection.radius

    # Save summary as CSV
    summary_dir = Path("Marvel.DC.results/summary")
    summary_dir.mkdir(exist_ok=True)

    for sim_name, halo_id in galaxy_collection.get_all_halo_keys():
        print(f"Processing halo {halo_id} from simulation {sim_name}...")
        try:
            samples, max_prob_params, sampler, q_obs, a_s, ba_s, ca_s = galaxy_collection.run_inference_single_halo(
                sim_name, halo_id,
                n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN,
                n_cores=N_CORES,
                output_dir=f'Marvel.DC.results/single_halos/{sim_name}_h{halo_id}_r{radius}{("_" + output_suffix) if output_suffix else ""}',
                force_rerun=force_rerun, n_angles=n_angles
            )

            # Store results
            results[(sim_name, halo_id)] = {
                'samples': samples,
                'max_prob_params': max_prob_params,
                'q_obs': q_obs,
                'sampler': sampler,
                'a_s': a_s,
                'ba_s': ba_s,
                'ca_s': ca_s,
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
                "σC Std": std_params[3],
                "True B/A": ba_s,
                "True C/A": ca_s,

            })

            # Save the results to a file
            output_filename = f"all_individual_results_{radius}Reff{('_' + output_suffix) if output_suffix else ''}.pkl"
            with open(summary_dir / output_filename, 'wb') as f:
                pickle.dump(results, f)

        except Exception as e:
            print(f"Error processing halo {halo_id} from {sim_name}: {e}")

    summary_df = pd.DataFrame(summary_data)
    csv_filename = f"all_halos_summary{radius}Reff{('_' + output_suffix) if output_suffix else ''}.csv"
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
# Helper functions
# ======================================

def build_collections(galaxy_collection, halo_metadata, collection_specs):
    """
    Populate multiple GalaxyEllipseCollections in a single pass.

    Parameters
    ----------
    galaxy_collection : GalaxyEllipseCollection
        Already-populated master collection to copy from.
    halo_metadata : dict
        Maps (sim, hid) -> dict with keys like 'ba_s', 'ca_s', 'log_stellar_mass'.
    collection_specs : list of dict, each with keys:
        - 'label'      : str, human-readable name
        - 'collection' : GalaxyEllipseCollection instance
        - 'predicate'  : callable(meta) -> bool, decides membership
        - 'color'      : str (optional, for later plotting)

    Returns
    -------
    dict mapping label -> {'collection': ..., 'color': ..., 'count': int}
    """
    results = {
        spec['label']: {'collection': spec['collection'], 'color': spec.get('color', 'gray'), 'count': 0}
        for spec in collection_specs
    }

    for (sim, hid), meta in halo_metadata.items():
        for spec in collection_specs:
            if spec['predicate'](meta):
                spec['collection'].copy_halo_from(galaxy_collection, sim, hid)
                results[spec['label']]['count'] += 1

    return results


def run_inference_batch(collection_specs, inference_kwargs, results_output_directory, force_rerun):
    """
    Run inference on each non-empty collection and return all results.

    Parameters
    ----------
    collection_specs : dict as returned by build_collections()
        Keys are labels; values have 'collection', 'color', 'count'.
    inference_kwargs : dict
        Shared keyword args passed to run_inference_all_halos()
        (n_steps, n_walkers, burn_in, n_cores, n_angles_per_halo).
    results_output_directory : str
    force_rerun : bool

    Returns
    -------
    dict mapping label -> {'samples', 'max_params', 'q_obs', 'n_halos'}
    """
    inference_results = {}

    for label, spec in collection_specs.items():
        collection = spec['collection']
        n_halos = collection.get_halo_count()

        if n_halos == 0:
            print(f"\nSkipping '{label}' — no halos.")
            continue

        print(f"\n{'=' * 50}")
        print(f"\n{'=' * 50}")
        print(f"Running inference on '{label}' ({n_halos} halos)...")

        subdir = label.lower().replace(' ', '_')
        #samples, max_params, sampler, q_obs =
        radius = collection.radius
        true_params,true_params_cov, max_prob_params, max_prob_params_cov, = collection.run_inference_all_halos(
            **inference_kwargs,
            force_rerun=force_rerun,
            output_dir=f"{results_output_directory}/{subdir}.radius{radius}.contained.{CONTAIN}",
            output_prefix=subdir,
            label=label,
            color=spec['color'],
            contain=CONTAIN
        )
        print("inference_kwargs keys:", sorted(inference_kwargs))
        print("reference_dir in kwargs:   ", inference_kwargs.get('reference_dir'))
        print("reference_prefix in kwargs:", inference_kwargs.get('reference_prefix'))
        collection.run_bootstrap_inference(
            output_dir=f"{results_output_directory}/{subdir}.radius{radius}.contained.{CONTAIN}",
            n_bootstrap = N_BOOTSTRAP,
            **inference_kwargs,
            n_workers=10,
            force_rerun=force_rerun,
            output_prefix=subdir,
            label=label,
            color=spec['color'],
            contain=CONTAIN,
            true_params=true_params,
            true_params_cov=true_params_cov,
            max_prob_params=max_prob_params,
            max_prob_params_cov=max_prob_params_cov,
        )

        # create_summary_table(
        #     {label: {'samples': samples, 'max_prob_params': max_params, 'q_obs': q_obs}},
        #     output_file=f"{results_output_directory}/summary/combined_summary_{subdir}.csv"
        # )
        #
        # inference_results[label] = {
        #     'samples': samples,
        #     'max_params': max_params,
        #     'q_obs': q_obs,
        #     'n_halos': n_halos,
        # }
        # print(f"'{label}' analysis complete.")

    #return inference_results
    return



def build_comparison_df(inference_results, output_path):
    """
    Build and save the category comparison DataFrame from inference results.

    Parameters
    ----------
    inference_results : dict as returned by run_inference_batch()
    output_path : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    param_names = ['B/A', 'C/A', 'sigmaB', 'sigmaC']

    for label, res in inference_results.items():
        row = {'Category': label, 'N_Halos': res['n_halos']}
        for i, name in enumerate(param_names):
            row[f'{name}_max_prob'] = res['max_params'][i]
            row[f'{name}_mean']     = np.mean(res['samples'][:, i])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def plot_klein_comparisons(klein_collections, klein_data, results_output_directory):
    """
    Generate Klein comparison plots for edge-on and random projections.

    Parameters
    ----------
    klein_collections : dict with keys 'low' and 'med', each a GalaxyEllipseCollection
    klein_data : module (Kleindata), expected attributes: x, gama_x, gama_y,
                 firebox_sideon_low_y, firebox_sideon_med_y, firebox_low, firebox_med
    results_output_directory : str
    """
    import shape_plotting

    mass_labels = {
        'low': r'$10^8 < M_{\odot} < 10^9$',
        'med': r'$10^9 < M_{\odot} < 10^{10}$',
    }
    firebox_sideon = {'low': klein_data.firebox_sideon_low_y, 'med': klein_data.firebox_sideon_med_y}
    firebox_random = {'low': klein_data.firebox_low,          'med': klein_data.firebox_med}

    gama_overlay = {
        'x_centers': klein_data.gama_x, 'y_values': klein_data.gama_y,
        'label': 'GAMA r-band', 'color': 'black', 'style': 'line',
        'bin_width': 0.1, 'linewidth': 2,
    }


    for key, collection in klein_collections.items():

        
        feedback = f'.{key}'

        key = 'med'
        halo_data = collection.halo_data
        q_sideon =[]
        orientations_strings = []
        x = 90
        y_l = np.arange(0, 180, 30)
        for y in y_l:
            orientation = f'x{x:03d}y{y:03d}'
            orientations_strings.append(orientation)
        for halo in halo_data:
            q_sideon_candidates = []
            for orientation in orientations_strings:
                q_sideon_candidates.append(1-halo_data[halo][orientation][1])
            q_sideon.append(np.nanmin(q_sideon_candidates))
        #print(q_sideon)


        # q_sideon  = [halo_data[k]['x000y090'][0] for k in halo_data]
        q_random  = collection.generate_q_distribution_all_halos(10)
        title     = mass_labels[key]

        for q_vals, firebox_y, suffix, label in [
            (q_sideon, firebox_sideon[key], f'edge_{key}',   'This work Edge-on V-band'),
            (q_random, firebox_random[key], f'random_{key}', 'This work V-band'),
        ]:
            overlays = [
                {'x_centers': klein_data.x, 'y_values': firebox_y,
                 'label': 'FireBox Edge-on r-band' if 'edge' in suffix else 'FireBox r-band',
                 'color': 'red', 'style': 'step', 'bin_width': 0.1, 'linewidth': 2},
                gama_overlay,
            ]
            shape_plotting.plot_projected_distributions(
                [q_vals], labels=[label], colors=None,
                bin_width=0.1, title=title, kde=False,
                precomputed_overlays=overlays,
                output_file=f"{results_output_directory}summary/klein_comparison_{suffix}{feedback}.png"
            )


"""
Refactored main block: all radius-independent work (tangos queries, pickle load,
shape splines, stellar masses, load_and_process_halo_data) happens ONCE.
The per-radius loop only evaluates the splines and rebuilds the collections,
so the interfaces used downstream are unchanged:

    GalaxyEllipseCollection(reff_index=..., run_covariances=...)
    run_all_individual_halos(collection, radius, force_rerun=...)
    run_inference_batch(built, radius, inference_kwargs, results_output_directory, force_rerun)
"""

import pickle
import traceback

import numpy as np
import tangos


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers: radius-INDEPENDENT
# ──────────────────────────────────────────────────────────────────────────────

def _feedback_type(sim):
    if sim.endswith('sbBH'):
        return 'sbBH'
    if sim.endswith('bwK1BH'):
        return 'bwBH'
    return None


def _select_halo_refs(sim, halo_refs):
    """Apply the per-simulation halo cuts. Independent of radius."""
    # Massive Merians (r* sims): keep only the most massive halo
    if sim.startswith('r') and not sim.startswith('rogue'):
        max_halo_ref, max_stars = None, 0
        for halo_ref in halo_refs:
            try:
                n_stars = tangos.get_halo(halo_ref)['n_star'][0]
            except Exception:
                continue
            if n_stars > max_stars:
                max_stars, max_halo_ref = n_stars, halo_ref
        return [max_halo_ref] if max_halo_ref else []

    # h* sims: skip halo 0
    if sim.startswith('h'):
        kept = [hr for hr in halo_refs
                if tangos.get_halo(hr).basename.split('_')[1] != '0']
        #print(f'Removing halo 0 from sim {sim}')
        return kept

    return halo_refs


def load_halo_records(pickle_file, test=False):
    """
    Do every expensive, radius-independent thing exactly once.

    Returns
    -------
    records : list of dict
        One entry per surviving halo, holding the raw ellipse payload plus the
        smoothed shape splines so that a_s/b_s/c_s can be evaluated at any radius.
    skipped : int
    """
    if test:
        print("testing shapes, will not load galaxies!")
        return [], 0

    with open(pickle_file, 'rb') as f:
        ellipse_dict = pickle.load(f)

    records, skipped = [], 0

    for sim in ellipse_dict:
        halo_refs = _select_halo_refs(sim, list(ellipse_dict[sim].keys()))
        feedback_type = _feedback_type(sim)

        for halo_ref in halo_refs:
            try:
                halo = tangos.get_halo(halo_ref)
                hid = halo.basename.split('_')[1]

                contaim_frac = halo['contamination_fraction']
                print(f'{sim}{hid} contamination fraction {contaim_frac:.2g}!')
                if contaim_frac > 0.02:
                    print(f'{sim}{hid} contamination fraction exceeds threshold {contaim_frac:.2g}!')
                    skipped += 1
                    continue
                else:
                    continue

                    # except Exception as e:
                    #     raise e
                    print(f'could not read contamination fraction for {sim} {hid}')

                try:
                    a, b, c = halo['a_s'], halo['b_s'], halo['c_s']
                    rbins = halo['rbins_s']
                except Exception:
                    print(f'no shape found for {sim} {hid}')
                    continue

                Reff = get_keys(sim, hid, (0.0, 0.0), 'Reff')

                rbins_f, a_f, b_f, c_f, a_s_func, b_s_func, c_s_func = smooth_shape(rbins, a, b, c, k=3)

                try:
                    a_i, b_i, c_i = halo['a_s_i'], halo['b_s_i'], halo['c_s_i']
                    rbins_i = halo['rbins_s_i']
                except Exception:
                    print(f'no shape (i) found for {sim} {hid}')
                    raise Exception

                rbins_i_f, a_i_f, b_i_f, c_i_f, a_i_s_func, b_i_s_func, c_i_s_func = smooth_shape(rbins_i, a_i, b_i, c_i, k=3)
                try:
                    stellar_mass = halo['M_star']
                except KeyError:
                    stellar_mass = halo['finder_star_mass']
                log_stellar_mass = np.log10(extract_single_value(stellar_mass))

                # The heavy ellipse payload — also radius-independent.
                halo_data = load_and_process_halo_data(pickle_file, sim, halo_ref)

            except Exception as e:
                print(f"\nError loading halo from {sim} {halo_ref}: {e}")
                traceback.print_exc()
                skipped += 1
                continue

            records.append({
                'sim': sim,
                'hid': hid,
                'halo_ref': halo_ref,
                'halo_data': halo_data,
                'a_s_func': a_s_func,
                'b_s_func': b_s_func,
                'c_s_func': c_s_func,
                'rbins': rbins,
                'Reff': Reff,
                'a_s_i_func': a_i_s_func,
                'b_s_i_func': b_i_s_func,
                'c_s_i_func': c_i_s_func,
                'rbins_i': rbins_i,

                'max_rbin': float(np.max(rbins)),
                'log_stellar_mass': log_stellar_mass,
                'feedback_type': feedback_type,
            })
    sys.exit()
    return records, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 helper: radius-DEPENDENT (cheap — just spline evaluations)
# ──────────────────────────────────────────────────────────────────────────────

def build_master_collection(records, radius, reff_index, run_covariances,
                            warn_extrapolation=True):
    """Evaluate the shape splines at `radius * Reff` and fill a fresh collection."""
    galaxy_collection = GalaxyEllipseCollection(radius=radius, reff_index=reff_index,
                                                run_covariances=run_covariances,band_key=band_key)
    halo_metadata = {}
    skipped = 0

    for rec in records:
        sim, hid = rec['sim'], rec['hid']
        r = radius * rec['Reff']
        # try:


        if warn_extrapolation and r > 1.5 * rec['max_rbin']:
            print(f'Warning: {radius}*Reff = {r:.3g} vs max rbin '
                  f'{rec["max_rbin"]:.3g} for {sim} {hid}')

        try:
            a_s = rec['a_s_func']
            b_s = rec['b_s_func']
            c_s = rec['c_s_func']
            rbins= rec['rbins']
            Reff = rec['Reff']
            ba_s, ca_s = b_s(Reff*radius) / a_s(Reff*radius) , c_s(Reff*radius)  / a_s(Reff*radius) 

            # if not (0 <= ba_s <= 1 and 0 <= ca_s <= 1):
            #     raise ValueError(f'axis ratios out of range: ba={ba_s}, ca={ca_s}')
        except Exception as e:
            print(f"\nError evaluating shape for {sim} {hid} at {radius}*Reff: {e}")
            skipped += 1
            continue
        try:
            a_s_i = rec['a_s_i_func']
            b_s_i = rec['b_s_i_func']
            c_s_i = rec['c_s_i_func']
            rbins_i= rec['rbins_i']
            ba_s_i, ca_s_i = b_s_i(Reff*radius) / a_s_i(Reff*radius) , c_s_i(Reff*radius)  / a_s_i(Reff*radius)

            # if not (0 <= ba_s <= 1 and 0 <= ca_s <= 1):
            #     raise ValueError(f'axis ratios out of range: ba={ba_s}, ca={ca_s}')
        except Exception as e:
            print(f"\nError evaluating shape for {sim} {hid} at {radius}*Reff: {e}")
            raise Exception

        # Shallow copy so per-radius shape values never leak between iterations.
        halo_data = dict(rec['halo_data'])
        halo_data['a_s'], halo_data['b_s'], halo_data['c_s'], halo_data['rbins'] = a_s, b_s, c_s, rbins
        halo_data['a_s_i'], halo_data['b_s_i'], halo_data['c_s_i'],halo_data['rbins_i'] = a_s_i, b_s_i, c_s_i, rbins_i



        try:
            galaxy_collection.add_halo(
                sim_name=sim, halo_id=hid, halo_data=halo_data
            )
        except AssertionError as e:
            print(f"\nError loading galaxy {sim} {hid}: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"\nError loading galaxy {sim} {hid}: {e}")
            traceback.print_exc()
            skipped += 1
            continue

        halo_metadata[(sim, hid)] = {
            'ba_s': ba_s,
            'ca_s': ca_s,
            'ba_s_i': ba_s_i,
            'ca_s_i': ca_s_i,
            'log_stellar_mass': rec['log_stellar_mass'],
            'feedback_type': rec['feedback_type'],
        }
    return galaxy_collection, halo_metadata, skipped


def make_collection_specs(radius,reff_index,run_covariances):
    """Fresh (empty) collections every call — build_collections fills them."""
    mb = MASS_BINS
    kwargs = dict(radius=radius,reff_index=reff_index,run_covariances=run_covariances,band_key=band_key)
    return [
        {'label': 'All galaxies',
         'collection': GalaxyEllipseCollection(**kwargs),
         'color': 'green',
         'predicate': lambda m: True},

        {'label': r"M$_* < 10^8 \text{M}_{\odot}$",
         'collection': GalaxyEllipseCollection(**kwargs),
         'color': 'red',
         'predicate': lambda m: 4 <= (m['log_stellar_mass'] - np.log10(0.6)) < 8},
        {'label': r"M$_* > 10^8 \text{M}_{\odot}$",
         'collection': GalaxyEllipseCollection(**kwargs),
         'color': 'blue',
         'predicate': lambda m: 8 <= m['log_stellar_mass'] < 12},
        #
        # {'label': 'KF Low Mass',
        #  'collection': GalaxyEllipseCollection(**kwargs),
        #  'color': 'red',
        #  'predicate': lambda m: mb['KF_low'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_low'][1]},
        #
        # {'label': 'KF Medium Mass',
        #  'collection': GalaxyEllipseCollection(**kwargs),
        #  'color': 'blue',
        #  'predicate': lambda m: mb['KF_med'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_med'][1]},
        #
        # {'label': 'KF High Mass',
        #  'collection': GalaxyEllipseCollection(**kwargs),
        #  'color': 'purple',
        #  'predicate': lambda m: mb['KF_high'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_high'][1]},
        #
        # {'label': 'Disky',
        #  'collection': GalaxyEllipseCollection(**kwargs),
        #  'color': 'orange',
        #  'predicate': lambda m: m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD},
        #
        # {'label': 'Non-disky',
        #  'collection': GalaxyEllipseCollection(**kwargs),
        #  'color': 'purple',
        #  'predicate': lambda m: not (m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD)},


        # ── kept for reference ───────────────────────────────────────────────
        # {'label': 'Superbubble', ..., 'predicate': lambda m: m['feedback_type'] == 'sbBH'},
        # {'label': 'Blastwave',   ..., 'predicate': lambda m: m['feedback_type'] == 'bwBH'},
        # {'label': 'Klein_low',   ..., 'predicate': lambda m: 8 <= m['log_stellar_mass'] < 9},
        # {'label': 'Klein_med',   ..., 'predicate': lambda m: 9 <= m['log_stellar_mass'] < 10},
        # {'label': 'Disky',       ..., 'predicate': lambda m: m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD},
        # {'label': 'Non-disky',   ..., 'predicate': lambda m: not (m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD)},
    ]


def make_synthetic_specs():
    return [
        {'label': f"Synthetic: {meta['label']}",
         'collection': SyntheticEllipseCollection(
             kado_fong_param_vector(key, rho_ac=-0.5, sigma_ba=0.13, sigma_ca=0.17),
             n_ellipsoids=200,
             run_covariances=False,
             sim_name=f"synthetic_{key}"),
         'color': meta['color'],
         'predicate': lambda m: False}
        for key, meta in KADO_FONG_SHAPES.items()
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("3D Shape Inference from Galaxy Ellipse Data")
    print("===========================================")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"MCMC: {N_WALKERS} walkers, {N_STEPS} steps, {BURN_IN} burn-in, {N_CORES} cores")

    # ── Step 0: load everything radius-independent, once ─────────────────────
    records, load_skipped = load_halo_records(pickle_file, test=test)
    print(f"\nLoaded {len(records)} halo records, skipped {load_skipped}.")

    # for reff_index, radius in enumerate(radius_list):
    for i in range(len(radius_list)):
        radius = radius_list[i]
        reff_index = indices[i]

        print(f"\n=== radius = {radius} Reff (reff_index={reff_index}) ===")

        # ── Step 1: cheap per-radius rebuild ─────────────────────────────────
        galaxy_collection, halo_metadata, skipped_count = build_master_collection(
            records, radius, reff_index, RUN_CONVARIANCE
        )
        print(f"Collection has {galaxy_collection.get_halo_count()} halos, "
              f"skipped {skipped_count} at this radius.")

        # ── Step 2: build sub-collections ────────────────────────────────────
        if test:
            built = build_collections(galaxy_collection, halo_metadata,
                                      make_synthetic_specs())
        else:
            specs = make_collection_specs(radius,reff_index,RUN_CONVARIANCE)
            if Klein_comparison:
                specs = [s for s in specs if s['label'] not in ('Disky', 'Non-disky')]
            built = build_collections(galaxy_collection, halo_metadata, specs)

        for label, info in built.items():
            print(f"  {label}: {info['count']} halos")



        # ── Step 3 (optional): Klein comparison plots ────────────────────────
        if Klein_comparison:
            import Kleindata
            plot_klein_comparisons(
                klein_collections={'low': built['Klein_low']['collection'],
                                   'med': built['Klein_med']['collection']},
                klein_data=Kleindata,
                results_output_directory=results_output_directory,
            )

        # ── Step 4: run inference on all collections ─────────────────────────
        inference_kwargs = dict(
            n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN,
            n_cores=N_CORES, n_angles_per_halo=N_ANGLES_PER_HALO_ALL,
        )
        run_inference_batch(built, inference_kwargs,
                            results_output_directory, force_rerun)


        # use to run all halos individually
        # for label, spec in built.items():
        #     print('Running individual halos')
        #     run_all_individual_halos(spec['collection'], radius, force_rerun=force_rerun)
        # sys.exit()

        # ── Step 5: comparison summary ───────────────────────────────────────
        # comparison_df = build_comparison_df(
        #     inference_results,
        #     output_path=results_output_directory + '/summary/category_comparison.csv'
        # )
        # print("\nComparison Summary:")
        # print(comparison_df.to_string(index=False))


