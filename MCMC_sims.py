import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
from galaxy_ellipse_collection import GalaxyEllipseCollection, SyntheticEllipseCollection
from glob import glob

#load configs
import sys
from config import db_connection, sys_path, results_output_directory, pickle_file
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all_shapes.db'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos

tangos_sims = tangos.all_simulations()

band = 'v'
image_type = 'stars'
radius = 2.5  # in Reff
reff_index = 1

# Configuration parameters (added from test code)
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# MCMC parameters (added from test code)
N_STEPS = 500  # Number of MCMC steps
BURN_IN = int(N_STEPS*3/4) # Number of burn-in steps to discard
N_CORES = 32 # Number of CPU cores to use for parallel processing
N_WALKERS = 64  # Number of MCMC walkers
N_ANGLES_PER_HALO = 2000  # Number of angles to sample for each halo
N_ANGLES_PER_HALO_ALL = 1000 # Number of angles to sample for each halo when running all combined

# Set force_rerun to False to use existing results if available
test = False
force_rerun = False
Klein_comparison = False

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
    return [d[k] for k in keys]


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
        collection.run_inference_all_halos(
            **inference_kwargs,
            force_rerun=force_rerun,
            output_dir=f"{results_output_directory}/{subdir}",
            output_prefix=subdir,
            label=label,
            color=spec['color'],
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


if __name__ == "__main__":
    print("3D Shape Inference from Galaxy Ellipse Data")
    print("===========================================")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"MCMC: {N_WALKERS} walkers, {N_STEPS} steps, {BURN_IN} burn-in, {N_CORES} cores")



    # ── Step 1: Load master collection ──────────────────────────────────────
    galaxy_collection = GalaxyEllipseCollection(reff_index=reff_index, run_covariances=False)
    halo_metadata = {}   # (sim, hid) -> {ba_s, ca_s, log_stellar_mass}
    skipped_count = 0

    if test:
        print("testing shapes, will not load galaxies!")
        ellipse_dict = {}
    else:
        with open(pickle_file, 'rb') as f:
            ellipse_dict = pickle.load(f)

    for sim in ellipse_dict:
        halo_refs = list(ellipse_dict[sim].keys())
        #get feedback type if possible
        feedback_type = None
        if sim.endswith('sbBH'):
            feedback_type = 'sbBH'
        elif sim.endswith('bwK1BH'):
            feedback_type = 'bwBH'


        # --- Filter: Massive Merians (r* sims) — keep only most massive halo ---
        if sim.startswith('r') and not sim.startswith('rogue'):
            max_halo_ref, max_stars = None, 0
            for halo_ref in halo_refs:
                try:
                    n_stars = tangos.get_halo(halo_ref)['n_star'][0]
                    if n_stars > max_stars:
                        max_stars, max_halo_ref = n_stars, halo_ref
                except Exception:
                    continue
            halo_refs = [max_halo_ref] if max_halo_ref else []
            print(f'Keeping only most massive halo from sim {sim}')

        # --- Filter: h* sims — skip halo 0 ---
        elif sim.startswith('h'):
            halo_refs = [hr for hr in halo_refs
                         if tangos.get_halo(hr).basename.split('_')[1] != '0']
            print(f'Removing halo 0 from sim {sim}')

        for halo_ref in halo_refs:
            try:

                halo  = tangos.get_halo(halo_ref)
                hid   = halo.basename.split('_')[1]
                #reff  = halo[f'image_reffs_{band}'][0] #replace this line
                #reff = get_keys(sim,hid,(0,0),'Reff')
                try:
                    a = halo['a_s']
                    b = halo['b_s']
                    c = halo['c_s']
                    rbins = halo['rbins_s']
                except Exception:
                    print(f'no shape found for {sim} {hid}')
                    continue
                Reff = get_keys(sim, hid, (0.0, 0.0), 'Reff')

                rbins_f, a_f, b_f, c_f, a_s_func, b_s_func, c_s_func = smooth_shape(rbins, a, b, c, k=3)
                # get a,b,c at 2*reff
                a_s = a_s_func(radius * Reff)
                b_s = b_s_func(radius * Reff)
                c_s = c_s_func(radius * Reff)
                ba_s = b_s/a_s
                ca_s = c_s/a_s


                # if Klein_comparison:
                #     ba_s, ca_s = halo['ba_s_v'], halo['ca_s_v']
                # else:
                #     rbins_s = halo['rbins_s']
                #     #show a warning if reff is more than 10% larger than largest rbin
                #     max_r = rbins_s[-1]
                #     if radius * reff > 1.1* max_r:
                #         print(f'Warning: {radius}*reff {radius* reff} vs max rbin {rbins_s[-1]} for halo {halo_ref}')
                #     ba_s  = halo.calculate('ba_s_smoothed()')(radius * reff)
                #     ca_s  = halo.calculate('ca_s_smoothed()')(radius * reff)

                try:
                    stellar_mass = halo['M_star']
                except KeyError:
                    stellar_mass = halo['finder_star_mass']

                stellar_mass    = extract_single_value(stellar_mass)
                log_stellar_mass = np.log10(stellar_mass)

                assert 0 <= ba_s <= 1
                assert 0 <= ca_s <= 1

            except Exception as e:
                print(f"\nError loading halo from {sim} {halo}: {e}")
                traceback.print_exc()
                skipped_count += 1
                continue

            halo_data = load_and_process_halo_data(pickle_file, sim, halo_ref)
            halo_data['a_s'], halo_data['ba_s'], halo_data['ca_s'] = a_s, ba_s, ca_s
            
            try:
                galaxy_collection.add_halo(
                    sim_name=sim, halo_id=hid, halo_data=halo_data,
                    interpolation_method='linear',
                    coordinate_system='angles'
                )
            except Exception as e:
                print(f"\nError loading galaxy {sim} {halo}: {e}")
                skipped_count += 1
                continue

            halo_metadata[(sim, hid)] = {'ba_s': ba_s, 'ca_s': ca_s, 'log_stellar_mass': log_stellar_mass, 'feedback_type': feedback_type}

    print(f"\nLoaded {galaxy_collection.get_halo_count()} halos, skipped {skipped_count}.")

    # ── Step 2: Build sub-collections ───────────────────────────────────────
    mb = MASS_BINS

    synthetic_specs = [
        {'label': f"Synthetic: {meta['label']}",
         'collection': SyntheticEllipseCollection(
             kado_fong_param_vector(key,rho_ac=-0.5,sigma_ba=0.13, sigma_ca=0.17),
             n_ellipsoids=200,
             run_covariances=False,
             sim_name=f"synthetic_{key}"),
         'color': meta['color'],
         'predicate': lambda m: False}
        for key, meta in KADO_FONG_SHAPES.items()
    ]
    
    collection_specs = [
        {'label': 'All galaxies', 'collection': GalaxyEllipseCollection(run_covariances=True), 'color': 'green',
         'predicate': lambda m: True},
        # {'label': 'Superbubble', 'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: m['feedback_type']=='sbBH'},
        # {'label': 'Blastwave', 'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: m['feedback_type'] == 'bwBH'},
        #
        # {'label': 'Klein_low_sb',    'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: (8 <= m['log_stellar_mass'] < 9 and m['feedback_type']=='sbBH')},
        # {'label': 'Klein_med_sb',    'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: (9 <= m['log_stellar_mass'] < 10 and m['feedback_type']=='sbBH')},
        #
        # {'label': 'Klein_low_bw',    'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: (8 <= m['log_stellar_mass'] < 9 and m['feedback_type']=='bwBH')},
        # {'label': 'Klein_med_bw',    'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: (9 <= m['log_stellar_mass'] < 10 and m['feedback_type']=='bwBH')},

        # {'label': 'All galaxies', 'collection': GalaxyEllipseCollection(), 'color': 'green',
        #  'predicate': lambda m: True},
        # {'label': r"M$_* < 10^8 \text{M}_{\odot}$",      'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: 4 <= (m['log_stellar_mass'] - np.log10(0.6))  < 8},

        # {'label': 'Disky',        'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD},
        # {'label': 'Non-disky',    'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: not (m['ba_s'] > BA_THRESHOLD and m['ca_s'] < CA_THRESHOLD)},
        # {'label': 'KF Low Mass',       'collection': GalaxyEllipseCollection(), 'color': 'red',
        #  'predicate': lambda m: mb['KF_low'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_low'][1]},
        # {'label': 'KF Medium Mass',       'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: mb['KF_med'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_med'][1]},
        #
        #run our covariances last:
        # {'label': r"M$_* > 10^8 \text{M}_{\odot}$", 'collection': GalaxyEllipseCollection(run_covariances=False),
        #  'color': 'blue',
        #  'predicate': lambda m: 8 <= m['log_stellar_mass'] < 12},
        # {'label': 'KF High Mass', 'collection': GalaxyEllipseCollection(run_covariances=False), 'color': 'green',
        #  'predicate': lambda m: mb['KF_high'][0] <= (m['log_stellar_mass'] - np.log10(0.6)) < mb['KF_high'][1]},


        # {'label': 'Klein_low',    'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: 8 <= m['log_stellar_mass'] < 9},
        # {'label': 'Klein_med',    'collection': GalaxyEllipseCollection(), 'color': 'blue',
        #  'predicate': lambda m: 9 <= m['log_stellar_mass'] < 10},
    ]

    # Skip disky/non-disky when doing Klein comparison


    if Klein_comparison:
        collection_specs = [s for s in collection_specs if s['label'] not in ('Disky', 'Non-disky')]
    if test:
        built = build_collections(galaxy_collection, halo_metadata, synthetic_specs)
    else:
        built = build_collections(galaxy_collection, halo_metadata, collection_specs)
    # use to run all halos individually
    # for label, spec in built.items():
    #     collection = spec['collection']
    #     print(f'Running individual halos')
    #     run_all_individual_halos(collection,force_rerun=force_rerun)
    # sys.exit()

    for label, info in built.items():
        print(f"  {label}: {info['count']} halos")

    # ── Step 3 (optional): Klein comparison plots ────────────────────────────
    if Klein_comparison:
        import Kleindata
        plot_klein_comparisons(
            klein_collections={'low': built['Klein_low']['collection'],
                                'med': built['Klein_med']['collection']},

            # klein_collections={'Superbubble': built['Superbubble']['collection'], 'Blastwave': built['Blastwave']['collection'],
            #                    },
            klein_data=Kleindata,
            results_output_directory=results_output_directory,
        )

    # ── Step 4: Run inference on all collections ─────────────────────────────
    inference_kwargs = dict(
        n_steps=N_STEPS, n_walkers=N_WALKERS, burn_in=BURN_IN,
        n_cores=N_CORES, n_angles_per_halo=N_ANGLES_PER_HALO_ALL,
    )
    # inference_results = (
    run_inference_batch(built, inference_kwargs, results_output_directory, force_rerun)

    # # ── Step 5: Comparison summary ───────────────────────────────────────────
    # comparison_df = build_comparison_df(
    #     inference_results,
    #     output_path=results_output_directory + '/summary/category_comparison.csv'
    # )
    # print("\nComparison Summary:")
    # print(comparison_df.to_string(index=False))
    # print("\nAnalysis completed successfully!")



