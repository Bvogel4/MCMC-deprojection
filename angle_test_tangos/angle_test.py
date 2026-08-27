import os
import numpy as np
import time
from pathlib import Path
import pickle
import pandas as pd
import traceback
from scipy import stats
import matplotlib.pyplot as plt
import pynbody
import warnings
import traceback
import logging
#from angle_test_tangos import mytangosproperty as mtp
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pynbody
import pymp
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pynbody
import pymp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator


# ---------------------------------------------------------------------- #
# Load
# ---------------------------------------------------------------------- #
pickle_filename = 'angle_test_ellipses_data'
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)
    print(f"Loaded existing data from {pickle_filename}")
else:
    raise FileNotFoundError(f"{pickle_filename} not found")


def parse_orientation(orientation):
    try:
        x_angle = int(orientation[1:4])
        y_angle = int(orientation[5:8])
        return x_angle, y_angle
    except (ValueError, IndexError):
        print(f"Invalid orientation format: {orientation}")
        return None, None


def _build_interpolators_cached(halo_data, reff_index,
                                 interpolation_method, coordinate_system):


    original_points = []
    original_values = []

    for orientation, ellipticities in halo_data.items():
        if not (orientation.startswith('x') and 'y' in orientation):
            continue
        x_angle, y_angle = parse_orientation(orientation)
        original_points.append([x_angle, y_angle])
        if reff_index < len(ellipticities):
            original_values.append(float(ellipticities[reff_index]))

    extended_points = original_points.copy()
    extended_values = original_values.copy()

    if coordinate_system == 'angles':
        for idx, point in enumerate(original_points):
            x_angle, y_angle = point
            if x_angle == 0:
                extended_points.append([180, y_angle])
                extended_values.append(original_values[idx])
            elif x_angle == 180:
                extended_points.append([0, y_angle])
                extended_values.append(original_values[idx])
            if y_angle == 0:
                extended_points.append([x_angle, 360])
                extended_values.append(original_values[idx])
            elif y_angle in (360, 359):
                extended_points.append([x_angle, 0])
                extended_values.append(original_values[idx])
            if (x_angle == 0 and y_angle == 0):
                extended_points.append([180, 360])
                extended_values.append(original_values[idx])
            elif (x_angle == 180 and y_angle == 0):
                extended_points.append([0, 360])
                extended_values.append(original_values[idx])
            elif (x_angle == 0 and y_angle in (360, 359)):
                extended_points.append([180, 0])
                extended_values.append(original_values[idx])
            elif (x_angle == 180 and y_angle in (360, 359)):
                extended_points.append([0, 0])
                extended_values.append(original_values[idx])

    points_array = np.array(extended_points)
    values_array = np.array(extended_values)
    valid_mask = ~np.isnan(values_array)
    points_array = points_array[valid_mask]
    values_array = values_array[valid_mask]

    interpolator = None
    fallback_interpolator = None

    if len(values_array) > 0:
        if interpolation_method == 'rbf':
            if coordinate_system == 'vectors':
                interpolator = Rbf(points_array[:, 0], points_array[:, 1],
                                   points_array[:, 2], values_array, function='multiquadric')
            else:
                interpolator = Rbf(points_array[:, 0], points_array[:, 1],
                                   values_array, function='multiquadric')
        elif interpolation_method == 'nearest':
            interpolator = NearestNDInterpolator(points_array, values_array)
        else:
            interpolator = LinearNDInterpolator(points_array, values_array)
            fallback_interpolator = NearestNDInterpolator(points_array, values_array)

    return interpolator, fallback_interpolator


def evaluate(interp, fallback, X, Y, method):
    """Evaluate an interpolator on a grid, filling NaNs from the fallback."""
    if method == 'rbf':
        Z = interp(X, Y)
    else:
        Z = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        if fallback is not None:
            bad = np.isnan(Z)
            if np.any(bad):
                Z[bad] = fallback(np.column_stack([X[bad], Y[bad]]))
    return np.asarray(Z, dtype=float)


# ---------------------------------------------------------------------- #
# Compare fine (10°) vs coarse (30°)
# ---------------------------------------------------------------------- #
REFF_INDEX  = 0          # which target radius (0 = 2×Reff, 1 = 3×, 2 = 4×)
METHOD      = 'linear'   # 'linear' | 'rbf' | 'nearest'
COORD_SYS   = 'angles'

halo_data = ellipse_dict   # adjust if your pickle nests by halo id


fine_data = ellipse_dict['r431']['/r431.romulus25.3072g1HsbBH/%/halo_1']
print(fine_data.keys())
orientations = fine_data.keys()
course_data = {}
indices = range
for i in len(orientations):

    orientation = orientations[i]





interp_fine, fb_fine = _build_interpolators_cached(
    halo_data, REFF_INDEX, METHOD, COORD_SYS)
interp_coarse, fb_coarse = _build_interpolators_cached(
    halo_data, REFF_INDEX, METHOD, COORD_SYS,)

# Evaluation grid: the full 10° sampling
x_vals = np.arange(0, 180, FINE)
y_vals = np.arange(0, 360, FINE)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

Z_fine   = evaluate(interp_fine,   fb_fine,   X, Y, METHOD)
Z_coarse = evaluate(interp_coarse, fb_coarse, X, Y, METHOD)

residual = Z_fine - Z_coarse

# Exclude the shared training nodes (both angles multiples of 30) — the two
# interpolators agree there by construction, so they'd bias the statistics.
shared_node = (X % COARSE == 0) & (Y % COARSE == 0)
eval_mask = ~shared_node & ~np.isnan(residual)

res = residual[eval_mask]

print(f"\nResidual statistics (fine {FINE}° − coarse {COARSE}°)")
print(f"  excluded shared nodes : {shared_node.sum()} of {X.size} grid points")
print(f"  evaluated points      : {res.size}")
print(f"  mean                  : {np.mean(res):+.5f}")
print(f"  median                : {np.median(res):+.5f}")
print(f"  std                   : {np.std(res):.5f}")
print(f"  mean |residual|       : {np.mean(np.abs(res)):.5f}")
print(f"  median |residual|     : {np.median(np.abs(res)):.5f}")
print(f"  RMSE                  : {np.sqrt(np.mean(res**2)):.5f}")
print(f"  max |residual|        : {np.max(np.abs(res)):.5f}")
print(f"  16th / 84th pct       : {np.percentile(res, 16):+.5f} / "
      f"{np.percentile(res, 84):+.5f}")

# Relative error, guarding against near-zero ellipticity
den = np.abs(Z_fine[eval_mask])
ok = den > 1e-6
if np.any(ok):
    rel = np.abs(res[ok]) / den[ok] * 100
    print(f"  mean rel. error       : {np.mean(rel):.2f}%")
    print(f"  median rel. error     : {np.median(rel):.2f}%")

# ---------------------------------------------------------------------- #
# Plot
# ---------------------------------------------------------------------- #
plot_res = np.where(eval_mask, residual, np.nan)
vmax = np.nanmax(np.abs(plot_res))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

im0 = axes[0].pcolormesh(Y, X, Z_fine, shading='auto', cmap='viridis')
axes[0].set_title(f'Fine interpolator ({FINE}°)')
fig.colorbar(im0, ax=axes[0], label='ellipticity')

im1 = axes[1].pcolormesh(Y, X, Z_coarse, shading='auto', cmap='viridis')
axes[1].set_title(f'Coarse interpolator ({COARSE}°)')
fig.colorbar(im1, ax=axes[1], label='ellipticity')

im2 = axes[2].pcolormesh(Y, X, plot_res, shading='auto',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_title(f'Residual (fine − coarse)\nshared {COARSE}° nodes masked')
fig.colorbar(im2, ax=axes[2], label='Δ ellipticity')

# Mark the coarse training nodes on the residual panel
axes[2].scatter(Y[shared_node], X[shared_node], s=18, facecolors='none',
                edgecolors='k', linewidths=0.8, label=f'{COARSE}° nodes')
axes[2].legend(loc='upper right', fontsize=8)

for ax in axes:
    ax.set_xlabel('y angle (deg)')
    ax.set_ylabel('x angle (deg)')
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(0, 181, 30))

plt.tight_layout()
plt.savefig('interpolator_comparison.png', dpi=150)
plt.show()

