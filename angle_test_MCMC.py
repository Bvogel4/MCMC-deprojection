import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
import re
import galaxy_ellipse_collection as gec

#load configs
import sys
from config import db_connection, sys_path, results_output_directory, pickle_file
os.environ['TANGOS_DB_CONNECTION'] = db_connection
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos

tangos_sims = tangos.all_simulations()

# Configuration parameters (added from test code)
RANDOM_SEED = 14
np.random.seed(RANDOM_SEED)

# Set force_rerun to False to use existing results if available
force_rerun = True

def extract_floats(data):
    """Parse '(np.float64(t), np.float64(p))' style keys -> (thetas, phis)."""
    pattern = r'np\.float64\(([^)]+)\)\s*,\s*np\.float64\(([^)]+)\)'
    if isinstance(data, str):
        data = [data]
    pairs = [m for s in data for m in re.findall(pattern, s)]
    thetas = [float(t) for t, p in pairs]
    phis   = [float(p) for t, p in pairs]
    return thetas, phis

# def parse_orientation(orientation):
#     try:
#         x_angle = int(orientation[1:4])
#         y_angle = int(orientation[5:8])
#         return x_angle, y_angle
#     except (ValueError, IndexError):
#         print(f"Invalid orientation format: {orientation}")
#         return None, None






def interpolator_test_MCMC(data,results_output_directory,subdir,color):

    data['ba_s'], data['ca_s'] = ba_s,ca_s
    collection = gec.GalaxyEllipseCollection(2)
    collection.add_halo(
                sim_name='r431', halo_id=1, halo_data=data,
                reff_index=0, interpolation_method='linear',
                coordinate_system='angles'
            )

    #print(collection,'\n',collection.halos.keys(),'\n',collection.interpolators)

    samples, max_params, sampler, q_obs = collection.run_inference_single_halo(
        sim_name='r431',
        halo_id=1,
        n_angles=3000,
        n_walkers=32,
        n_steps=20000,
        burn_in=1000,
        n_cores=32,
        output_prefix=subdir,
        output_dir=f"{results_output_directory}/{subdir}",
        force_rerun=force_rerun,
        color=color
    )

    
halo = tangos.get_halo('r431.romulus25.3072g1HsbBH/%/halo_1')
reff  = halo[f'image_reffs_v'][0]
ba_s = halo.calculate('ba_s_smoothed()')(2 * reff)
ca_s = halo.calculate('ca_s_smoothed()')(2 * reff)



#load ellipse_data
pickle_filename = 'angle_test_tangos/angle_test_ellipses_data_500'
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)
    print(f"Loaded data from {pickle_filename}??!!")
else:
    raise FileNotFoundError(f"{pickle_filename} not found")



fine_data = ellipse_dict['r431']['/r431.romulus25.3072g1HsbBH/%/halo_1']

def keys_to_xyz(keys):
    """Map (theta, phi) keys to unit vectors on the sphere."""
    thetas, phis = extract_floats(keys)
    thetas, phis = np.asarray(thetas), np.asarray(phis)
    return np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas),
    ])

def farthest_point_order(xyz, start=0):
    """Return an ordering of points; any prefix is ~evenly spread."""
    n = len(xyz)
    order = np.empty(n, dtype=int)
    order[0] = start
    d2 = np.sum((xyz - xyz[start])**2, axis=1)   # squared chordal dist
    for i in range(1, n):
        idx = int(np.argmax(d2))
        order[i] = idx
        d2 = np.minimum(d2, np.sum((xyz - xyz[idx])**2, axis=1))
    return order

def downsample(data, n_keep, order, keys):
    keep = order[:n_keep]
    return {keys[i]: data[keys[i]] for i in keep}

keys  = list(fine_data.keys())
order = farthest_point_order(keys_to_xyz(keys))
coarse_data = downsample(fine_data, 100, order, keys)




if __name__ == "__main__":
    results_output_directory = 'angle_test_tangos/MCMC/'
    interpolator_test_MCMC(fine_data,results_output_directory, 'fine',None)
    interpolator_test_MCMC(coarse_data,results_output_directory, 'coarse',None)