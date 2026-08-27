"""
Galaxy Morphology Analysis: Ellipticity-Surface Brightness Correlation
====================================================================
This code analyzes the correlation between galaxy ellipticity and surface brightness
following Xu & Randall 2020. It processes simulation data and compares with observations
of Local Group dwarf spheroidal galaxies.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import traceback
import dill  # Changed from pickle to dill
import galaxy_ellipse_collection as gec
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
from typing import Optional, Dict, List
from collections import defaultdict

TIMING_DATA = defaultdict(list)
ENABLE_TIMING = True

warnings.filterwarnings("ignore")

# ===========================
# Configuration and Setup
# ===========================

import sys
from config import db_connection, sys_path, xu_output_dir, pickle_file

os.environ['TANGOS_DB_CONNECTION'] =  '/home/bk639/data_base/CDM_all_shapes.db'#db_connection
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos

import mytangosproperty
from galaxy_ellipse_collection import random_viewing_angles, SphereInterpolator, get_keys
from MCMC_sims import smooth_shape

# Physical constants
MASS_TO_LIGHT_THRESHOLD = 85  # M_sun/L_sun threshold for bright/dim classification

# Analysis parameters
N_ITERATIONS = 1000  # Number of Monte Carlo iterations
MIN_STAR_PARTICLES = 10000  # Minimum star particles for analysis

# Output directory
OUTPUT_DIR = xu_output_dir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Progress file paths
PROGRESS_FILE = os.path.join('caches', 'galaxy_data_progress.pkl')
print(PROGRESS_FILE)


# ===========================
# Data Structures
# ===========================

@dataclass
class GalaxyData:
    """Container for all galaxy properties"""
    simulation: str
    halo_id: str
    stellar_mass: float
    total_luminosity: float
    dynamical_mass: float
    mass_to_light: float
    half_light_radius: float
    environment: str  # 'central', 'satellite', 'backsplash'
    ellipticity_interpolator: callable
    surface_brightness_interpolator: callable
    shape_b_over_a: callable  # 3D shape parameter b/a
    shape_c_over_a: callable  # 3D shape parameter c/a
    reff: float


@dataclass
class ObservationData:
    """Container for observational comparison data"""
    name: str
    correlation: float
    error: float
    color: str
    label: str


# ===========================
# Progress Tracking Functions
# ===========================

def save_progress(galaxy_list: List[GalaxyData]):
    """Save current progress to pickle file using dill"""
    try:
        with open(PROGRESS_FILE, 'wb') as f:
            dill.dump(galaxy_list, f)
        print(f"Progress saved: {len(galaxy_list)} galaxies")
    except Exception as e:
        print(f"Warning: Failed to save progress: {str(e)}")
        traceback.print_exc()


def load_progress() -> Tuple[List[GalaxyData], set]:
    """Load existing progress from pickle file and derive completed halos from galaxy list"""
    galaxy_list = []
    completed_halos = set()

    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                galaxy_list = dill.load(f)
            print(f"Loaded {len(galaxy_list)} galaxies from previous run")

            # Reconstruct completed_halos from the galaxy_list
            for galaxy in galaxy_list:
                # Need to reconstruct the halo_name format used in get_halo_key
                halo_name = f"halo_{galaxy.halo_id}"
                halo_key = get_halo_key(galaxy.simulation, halo_name)
                completed_halos.add(halo_key)

            print(f"Derived {len(completed_halos)} completed halos from galaxy list")

        except Exception as e:
            print(f"Warning: Failed to load galaxy list: {str(e)}")
            traceback.print_exc()

    return galaxy_list, completed_halos


def get_halo_key(sim_name: str, halo_name: str) -> str:
    """Create unique key for tracking completed halos"""
    return f"{sim_name}::{halo_name}"


# ===========================
# Utility Functions
# ===========================

def calculate_correlation(ellipticity: np.ndarray, surface_brightness: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between ellipticity and surface brightness.

    Parameters:
    -----------
    ellipticity : array-like
        Galaxy ellipticity values (1 - b/a)
    surface_brightness : array-like
        Surface brightness values

    Returns:
    --------
    float : Correlation coefficient
    """
    if not isinstance(ellipticity, np.ndarray):
        ellipticity = np.array(ellipticity)
    if not isinstance(surface_brightness, np.ndarray):
        surface_brightness = np.array(surface_brightness)

    # Remove NaN values
    mask = np.isfinite(ellipticity) & np.isfinite(surface_brightness)
    if np.sum(mask) < 3:  # Need at least 3 points for meaningful correlation
        return np.nan

    ell_clean = ellipticity[mask]
    sb_clean = surface_brightness[mask]

    numerator = np.sum((ell_clean - np.mean(ell_clean)) * (sb_clean - np.mean(sb_clean)))
    denominator = np.sqrt(np.sum((ell_clean - np.mean(ell_clean)) ** 2) *
                          np.sum((sb_clean - np.mean(sb_clean)) ** 2))

    return numerator / denominator if denominator != 0 else np.nan


def calculate_triaxiality(b_over_a: float, c_over_a: float) -> float:
    """
    Calculate triaxiality parameter T_s.

    T_s = (1 - b²/a²) / (1 - c²/a²)
    T_s < 1/3: oblate
    1/3 < T_s < 2/3: triaxial
    T_s > 2/3: prolate
    """
    return (1 - b_over_a ** 2) / (1 - c_over_a ** 2)


def extract_single_value(value):
    """Extract scalar value from potentially nested data structures"""
    if isinstance(value, list):
        return value[0]
    return float(value)


def load_observational_data(xu_folder: str) -> Tuple[ObservationData, ...]:
    """Load Xu & Randall 2020 observational comparison data"""
    # Define observation categories
    obs_data = [
        ObservationData("All LG Dsphs", -0.287, 0.058, '#7B0B01', 'All LG Dsphs'),
        ObservationData("LG Bright", -0.353, 0.059, '#FFCC99',
                        'LG Dsphs\n$M/L<100M_\\odot/L_\\odot$'),
        ObservationData("LG Dim", 0.509, 0.309, '#339AFE',
                        'LG Dsphs\n$M/L>100M_\\odot/L_\\odot$'),
        ObservationData("FIRE", -0.322, 0.213, '#411B52', 'FIRE Simulations')
    ]

    # Load FIRE simulation data
    fire_data = pd.read_csv(os.path.join(xu_folder, 'Data/BasicData/FIRE_Data.csv'))

    # Load scatter plot data
    xu_dim = pd.read_csv(os.path.join(xu_folder, 'Data/BasicData/Xu_Scatter_Dim.csv'))
    xu_bright = pd.read_csv(os.path.join(xu_folder, 'Data/BasicData/Xu_Scatter_Bright.csv'))

    return obs_data, fire_data, xu_dim, xu_bright


def SB_integrated(r_bins, profile_lum_den_v):
    """
    Integrate luminosity density profile to get total luminosity.

    Parameters:
    -----------
    r_bins : array
        Radial bin edges
    profile_lum_den_v : array
        Luminosity density in each bin

    Returns:
    --------
    total_lum : float
        Total integrated luminosity
    """
    # Add 0 at the beginning of r_bins to include the innermost bin
    r_bins_extended = np.insert(r_bins, 0, 0)

    # Calculate annular area for each bin
    # Area of annulus = π * (r_outer² - r_inner²)
    r_inner = r_bins_extended[:-1]
    r_outer = r_bins_extended[1:]

    areas = np.pi * (r_outer ** 2 - r_inner ** 2)

    # remove nans
    nanmask = np.isnan(profile_lum_den_v)
    profile_lum_den_v = profile_lum_den_v[~nanmask]
    areas = areas[~nanmask]

    # Total luminosity = sum of (luminosity density × area)
    total_lum = np.sum(profile_lum_den_v * areas)

    return total_lum


# ===========================
# Data Loading Functions
# ===========================

def load_galaxy_data() -> List[GalaxyData]:
    """
    Load all galaxy data from Tangos database and ellipse files.
    Automatically resumes from previous progress if available.

    Returns:
    --------
    List[GalaxyData] : List of galaxy data objects
    """
    print("Loading galaxy data from Tangos database...")

    # Load existing progress
    galaxy_list, completed_halos = load_progress()

    # Load ellipse data from pickle file (can use regular pickle for this)
    import pickle
    with open('ellipse_data_v_stars.pickle', 'rb') as f:
        ellipse_dict = pickle.load(f)

    # Get all simulations from Tangos
    sims = tangos.all_simulations()
    print(f"Found {len(sims)} simulations")

    for sim in sims:
        sim_name = str(sim.basename)
        print(f"\nProcessing simulation {sim_name}")

        if len(sim.timesteps)==0:
            print(f'No timesteps available for {sim_name}')
            continue
        elif len(sim.timesteps) > 1:
            timestep = sim.timesteps[-1]
        else:
            timestep = sim.timesteps[0]

        halos = timestep.halos[:20]  # Limit for testing

        for halo in halos:
            halo_name = halo.basename
            halo_ref = f'{sim_name}/%/{halo_name}'
            halo_key = get_halo_key(sim_name, halo_name)

            # Skip if already processed
            if halo_key in completed_halos:
                print(f"Skipping halo {halo_name} - already processed")
                continue

            # Check if halo has readable stellar mass
            if 'finder_star_mass' in halo:
                if halo['finder_star_mass'] <= 0:
                    completed_halos.add(halo_key)  # Mark as processed (skip)
                    continue
            elif 'M_star' in halo:
                if extract_single_value(halo['M_star']) <= 0:
                    completed_halos.add(halo_key)  # Mark as processed (skip)
                    continue

            # Check if halo exists in ellipse_dict
            if not halo_ref in ellipse_dict[sim_name]:
                completed_halos.add(halo_key)  # Mark as processed (skip)
                continue

            print(f"Processing halo {halo.halo_number} ({halo_name})")

            try:
                galaxy = process_halo(halo, sim, ellipse_dict)
                if galaxy:
                    galaxy_list.append(galaxy)
                    print(f"Successfully processed halo {halo_name}")

                # Mark as completed and save progress
                completed_halos.add(halo_key)
                save_progress(galaxy_list)

            except Exception as e:
                print(f"Error processing halo {halo.halo_number}: {str(e)}")
                traceback.print_exc()
                # Mark as completed even on error to avoid infinite retries
                completed_halos.add(halo_key)
                save_progress(galaxy_list)

    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Successfully loaded {len(galaxy_list)} galaxies")
    print(f"Processed {len(completed_halos)} total halos")
    print(f"{'=' * 60}\n")

    return galaxy_list


def process_halo(halo, sim, ellipse_dict: dict) -> Optional[GalaxyData]:
    """Process a single halo and extract all relevant data"""
    # Basic setup
    sim_name = str(sim.basename)
    halo_id = str(halo.halo_number)
    halo_ref = f'{sim.basename}/%/{halo.basename}'

    # Check star particle count
    if 'n_star' in halo.keys() and halo['n_star'][0] < MIN_STAR_PARTICLES:
        return None

    # Load ellipse data
    halo_data = gec.load_and_process_halo_data(sim.basename, halo_ref,pickle_filename=pickle_file)

    # Extract masses
    if 'finder_star_mass' in halo:
        stellar_mass = halo.get('finder_star_mass')
        gas_mass = halo.get('finder_gas_mass')
        dm_mass = halo.get('finder_dm_mass')
        virial_mass = stellar_mass + dm_mass + gas_mass
    elif 'n_star' in halo:
        virial_mass = halo.get('Mvir')
        stellar_mass = halo.get('M_star')

    dynamical_mass = extract_single_value(halo.get('Mdyn', np.nan))



    # Get half-light radius
    #half_light_radius = halo.get('Rhalf_v', np.nan)

    # reff = halo.get('image_reffs_v',np.nan)[0]

    Reff,v_lum,half_light_radius = get_keys(sim.basename, halo_id,(0,0), ['Reff','V_lum_den','Rhalf'])
    profile_rbins,mags_v = get_keys(sim.basename, halo_id,(0,0), ['rbins','mags_V'])
    try:
        a = halo['a_s']
        b = halo['b_s']
        c = halo['c_s']
        rbins_s = halo['rbins_s']
    except Exception:
        print(f'no shape found for {sim} {halo}')
        return None

    rbins_f, a_f, b_f, c_f, a_s_func, b_s_func, c_s_func = smooth_shape(rbins_s, a, b, c, k=3)
    # get a,b,c at 2*reff
    a_s = a_s_func(2 * Reff)
    b_s = b_s_func(2 * Reff)
    c_s = c_s_func(2 * Reff)
    ba_s = b_s / a_s
    ca_s = c_s / a_s

    #halo_data = gec.load_and_process_halo_data(pickle_file, sim, halo_ref)
    ellipticity_interpolator = SphereInterpolator(halo_data)
    sb_dict = {}

    thetas, phis = gec.extract_floats(list(halo_data.keys()))
    thetas = np.asarray(thetas)
    phis = np.asarray(phis)
    orientations = list(halo_data.keys())
    for orientation,theta,phi in zip(orientations,thetas,phis):
        mag,area = get_keys(sim.basename, halo_id, (theta,phi),['mags_V','binarea'])
        mag0 = mag[0]
        area0 = area[0]
        sigma0 = (10 ** (0.4 * (4.8 - mag0))) / area0   # normalize to total lum
        sb_dict[orientation] = [sigma0]
    sb_interpolator = SphereInterpolator(sb_dict)

    # mags = halo['profile_mags_v'][0]
    # rbins = halo['profile_rbins_v'][0]
    # reff = halo['image_reffs_v'][0]



    # Find index closest to effective radius
    ind_eff = np.argmin(np.abs(Reff - profile_rbins))

    # Calculate luminosity within effective radius
    luminosities = 10 ** (0.4 * (4.8 - mags_v))
    lum_eff = np.sum(luminosities[:ind_eff + 1])

    #return
    mass_to_light = dynamical_mass / lum_eff
    # Calculate mass-to-light ratio (using face-on orientation)
    # mass_to_light = calculate_mass_to_light_ratio(halo, dynamical_mass)

    # # Create interpolation functions for ellipticity and surface brightness
    # ellipticity_interpolator, sb_interpolator = create_interpolators(
    #     halo, halo_ellipse_data
    # )

    #total_lum = SB_integrated(halo['profile_rbins_v'][0], halo['profile_v_lum_den'][0])
    total_lum = SB_integrated(profile_rbins, v_lum)

    # Get 3D shape functions
    # shape_b_over_a = halo.calculate('ba_s_smoothed()')
    # shape_c_over_a = halo.calculate('ca_s_smoothed()')

    # Determine environment (will be updated later from external file)
    environment = 'unknown'

    # Create galaxy data object
    galaxy_data = GalaxyData(
        simulation=sim_name,
        halo_id=halo_id,
        stellar_mass=stellar_mass,
        total_luminosity=total_lum,
        dynamical_mass=dynamical_mass,
        mass_to_light=mass_to_light,
        half_light_radius=half_light_radius,
        environment=environment,
        ellipticity_interpolator=ellipticity_interpolator,
        surface_brightness_interpolator=sb_interpolator,
        shape_b_over_a=ba_s,
        shape_c_over_a=ca_s,
        reff=Reff
    )

    return galaxy_data


def calculate_mass_to_light_ratio(halo, dynamical_mass: float) -> float:
    """Calculate M/L ratio for face-on orientation"""

    # if 'profile_mags_v' not in halo.keys() or 'profile_rbins_v' not in halo.keys():
    #     return np.nan

    # Get face-on data (index 0)
    # mags = halo['profile_mags_v'][0]
    # rbins = halo['profile_rbins_v'][0]
    # reff = halo['image_reffs_v'][0]
    # mags,rbins,reff = get_keys(sim_name,hid)

    # Find index closest to effective radius
    ind_eff = np.argmin(np.abs(reff - rbins))

    # Calculate luminosity within effective radius
    luminosities = 10 ** (0.4 * (4.8 - mags))
    lum_eff = np.sum(luminosities[:ind_eff + 1])

    return dynamical_mass / lum_eff if lum_eff > 0 else np.nan


# def create_interpolators(halo, halo_ellipse_data) -> Tuple[callable, callable]:
#     """Create interpolation functions for ellipticity and surface brightness"""
#     orientations = halo['image_orientations_v']
#
#     # Create ellipticity data dict
#     ellipse_data_dict = {}
#     for i, orientation in enumerate(orientations):
#         # Use 2Reff data (index 0)
#         ellipse_data_dict[orientation] = [halo_ellipse_data[orientation][0]]
#
#     # Create surface brightness data dict
#     sb_data_dict = {}
#     # total luminosity at face-on orientation (not significantly different from any other orientation)
#
#     for i, orientation in enumerate(orientations):
#         # Calculate central surface brightness
#         if 'profile_mags_v' in halo.keys() and 'profile_binarea_v' in halo.keys():
#             mag0 = halo['profile_mags_v'][i][0]
#             area0 = halo['profile_binarea_v'][i][0]
#             sigma0 = (10 ** (0.4 * (4.8 - mag0))) / area0   # normalize to total lum
#             sb_data_dict[orientation] = [sigma0]
#
#     sim_name = halo.path.split('.')[0]
#     # Create interpolation functions
#     ellipticity_func, _ = gec._build_interpolators_cached(sim_name, halo.id,
#         ellipse_data_dict, 0, 'linear', 'angles'
#     )
#     sb_func, _ = gec._build_interpolators_cached(sim_name, halo.id,
#         sb_data_dict, 0,'linear', 'angles'
#     )
#
#     return ellipticity_func, sb_func


def load_environment_classifications(galaxy_list: List[GalaxyData]) -> None:
    """Load environment classifications from external files"""
    # Create mapping for quick lookup
    galaxy_dict = {(g.simulation, g.halo_id): g for g in galaxy_list}

    # Read environment files
    env_files = [
        '/home/bk639/MorphologyMeasurements/Data/BasicData/HaloTypes.BWMDC.txt',
        # '/home/bk639/MorphologyMeasurements/Data/BasicData/HaloTypes.MerianCDM.txt'
    ]

    for file_path in env_files:
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            sim_short = parts[0]
            halo_id = parts[1]
            env_type = parts[-2]

            # Find matching galaxy
            for galaxy in galaxy_list:
                if sim_short in galaxy.simulation and galaxy.halo_id == halo_id:
                    if env_type == 'Central':
                        galaxy.environment = 'central'
                    elif env_type == 'Satellite':
                        galaxy.environment = 'satellite'
                    break

    # Mark all 'r' simulations as centrals
    for galaxy in galaxy_list:
        if galaxy.simulation.startswith('r') and not galaxy.simulation.startswith('rogue'):
            galaxy.environment = 'central'


# ===========================
# Helper function for Jupyter notebooks
# ===========================



def load_galaxy_data_from_file(filepath: str = None) -> List[GalaxyData]:
    """
    Load galaxy data from pickle file for use in Jupyter notebooks.

    Parameters:
    -----------
    filepath : str, optional
        Path to the galaxy data pickle file. If None, uses default location.

    Returns:
    --------
    List[GalaxyData] : List of galaxy data objects

    Usage in Jupyter notebook:
    --------------------------
    import dill
    from your_script_name import load_galaxy_data_from_file
    galaxy_list = load_galaxy_data_from_file()
    """
    if filepath is None:
        filepath = PROGRESS_FILE

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Galaxy data file not found: {filepath}")

    with open(filepath, 'rb') as f:
        galaxy_list = dill.load(f)

    print(f"Loaded {len(galaxy_list)} galaxies from {filepath}")



    return galaxy_list


# ===========================
# Main execution
# ===========================

if __name__ == "__main__":
    # Load galaxy data (will resume from previous progress if available)
    galaxy_list = load_galaxy_data()

    # Load environment classifications
    load_environment_classifications(galaxy_list)


    # Load observational comparison data
    xu_folder = '/home/bk639/MorphologyMeasurements'
    obs_data, fire_data, xu_dim, xu_bright = load_observational_data(xu_folder)

    print("\nData loading complete! Galaxy data saved to:")
    print(f"  {PROGRESS_FILE}")