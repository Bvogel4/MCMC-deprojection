import os
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import argparse
import shutil
from datetime import datetime
from matplotlib.patches import Ellipse

# Set environment variables
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all.db'
os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/bk639/data/CDM_z0'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
# Add python path /home/bk639/MorphologyMeasurements/Code/tangos
sys.path.append('/home/bk639/mytangosproperty')
import tangos
import mytangosproperty


def plot_isophotes(images, isophote_params, orientations, reffs, rhalf, filename, save_plots=True):
    """
    Function to plot isophotes on images

    Parameters:
    -----------
    images : list
        List of image arrays
    isophote_params : list
        List of isophote parameters
    orientations : list
        List of orientation strings
    reffs : list
        List of effective radii
    rhalf : float
        Half-mass radius
    filename : str
        Path to save the figure
    save_plots : bool
        Whether to save the plot to disk

    Returns:
    --------
    ellipse_dict : dict
        Dictionary containing ellipticity measurements
    """
    # 72 images, 72 isophote_params, 72 orientations
    extent = 6 * rhalf  # in kpc
    kpc_per_pixel = extent / images[0].shape[0]
    images_3d = np.array(images)

    fig, axs = plt.subplots(6, 12, figsize=(30, 15))
    fig.patch.set_facecolor('black')  # Set the figure background color

    axs = axs.flatten()
    ellipse_dict = {}
    center = (images[0].shape[0] // 2, images[0].shape[1] // 2)
    for i in range(len(images)):
        vmin = np.min(images[i])
        axs[i].imshow(np.log10(images[i]), cmap='magma', origin='lower')
        reff = reffs[i]
        # convert reff to pixels
        reff = reff / kpc_per_pixel

        # plot isophotes
        iso_params = isophote_params[i]

        smas, epss, pas, grad_errs, x0s, y0s = [], [], [], [], [], []

        for j in range(len(iso_params)):
            sma, eps, pa, grad_err, x0, y0 = iso_params[j]
            smas.append(sma)
            epss.append(eps)
            pas.append(pa)
            grad_errs.append(grad_err)
            x0s.append(x0)
            y0s.append(y0)

        ellipses = np.ones(3) * np.nan
        #check if smas is empty
        if len(smas) == 0:
            print(f"No isophotes found for image {i}, skipping")
            continue
        for j in [2, 3, 4]:
            # find index of sma closest to j*reff
            idx = (np.abs(np.array(smas) - j * reff)).argmin()

            sma = smas[idx]
            eps = epss[idx]
            pa = pas[idx]
            grad_err = grad_errs[idx]
            x0 = x0s[idx]
            y0 = y0s[idx]

            center_offset = np.sqrt((images[i].shape[0] // 2 - x0) ** 2 + (images[i].shape[1] // 2 - y0) ** 2)

            # get ellipse parameters
            # color by gradient error
            vmin = 0
            vmax = 0.5
            # create colormap
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.winter

            # distance in pixels from 0.1kpc
            d = 10 / kpc_per_pixel * reff
            if (center_offset < d) and (grad_err < 1):
                ellipse = Ellipse((x0, y0), 2 * sma, 2 * sma * (1 - eps), angle=np.degrees(pa),
                                  edgecolor=cmap(norm(grad_err)), facecolor='none')
                axs[i].add_patch(ellipse)
                ellipses[j - 2] = eps
            else:
                ellipses[j - 2] = np.nan

        # save ellipses to dict
        ellipse_dict[orientations[i]] = ellipses

        axs[i].axis('off')
        axs[i].set_aspect('equal')

        axs[i].set_title(f'{orientations[i]}', color='white', y=0.85)
    # reduce white space
    plt.subplots_adjust(wspace=0, hspace=0)

    if save_plots:
        # Create figures directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=150)

    plt.close(fig)
    return ellipse_dict


def create_backup(pickle_filename):
    """Create a backup of the pickle file with timestamp"""
    if os.path.exists(pickle_filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.splitext(pickle_filename)[0]}_{timestamp}.pickle"
        shutil.copy2(pickle_filename, backup_filename)
        print(f"Created backup at {backup_filename}")
        return backup_filename
    return None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process isophote data from simulations')
    parser.add_argument('--pickle_file', default='ellipse_data.pickle', help='Filename for the pickle data')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing pickle file')
    parser.add_argument('--backup', action='store_true',
                        help='Create backup of existing pickle file before overwriting')
    parser.add_argument('--no_plots', action='store_true', help='Disable plot creation and saving')
    parser.add_argument('--max_halos', type=int, default=100, help='Maximum number of halos to process')
    parser.add_argument('--min_stars', type=int, default=4000, help='Minimum number of stars required')
    args = parser.parse_args()

    pickle_filename = args.pickle_file
    save_plots = not args.no_plots

    # Initialize ellipse_dict
    if os.path.exists(pickle_filename) and not args.overwrite:
        with open(pickle_filename, 'rb') as f:
            ellipse_dict = pickle.load(f)
        print(f"Loaded existing data from {pickle_filename}")
    else:
        if os.path.exists(pickle_filename) and args.backup:
            create_backup(pickle_filename)
        ellipse_dict = {}
        print(f"Starting with fresh data")

    # Get all simulations
    sims = tangos.all_simulations()

    for sim in sims:
        # Initialize dictionary for this sim if it doesn't exist
        if sim.basename not in ellipse_dict:
            ellipse_dict[sim.basename] = {}
        print(f"Processing simulation: {sim.basename}")

        if len(sim.timesteps) > 1:
            timestep = sim.timesteps[-1]
        elif len(sim.timesteps) == 1:
            timestep = sim.timesteps[0]
        else:
            print(f"No timesteps found for {sim.basename}, skipping")
            continue

        halos = timestep.halos[:args.max_halos]

        for hid in range(len(halos)):
            # Skip if we've already processed this halo and not overwriting
            if hid in ellipse_dict[sim.basename] and not args.overwrite:
                print(f"Skipping halo {hid} - already processed")
                continue

            try:
                halo = halos[hid]
                if halo['n_star'][0] < args.min_stars:
                    #print(f"Skipping halo {hid} - only {halo['n_star'][0]} stars (minimum {args.min_stars})")
                    continue

                print(f"Processing halo {hid} with {halo['n_star'][0]} stars")

                # Get images and isophote
                halo_images = halo['halo_images']
                image_reffs = halo['image_reffs']
                image_orientations = halo['image_orientations']
                Rhalf = halo['Rhalf']
                isophote_params = halo['isophote_parameters']
                reffs = np.array(image_reffs)

                filename = os.path.join('figures', f"{sim.basename}.{hid}.isophotes.png")

                halo_dict = plot_isophotes(
                    halo_images,
                    isophote_params,
                    image_orientations,
                    reffs,
                    Rhalf,
                    filename,
                    save_plots=save_plots
                )

                # Save to dictionary
                ellipse_dict[sim.basename][hid] = halo_dict

                # Save to pickle file after each halo is processed
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(ellipse_dict, f)
                print(f"Saved data to {pickle_filename} after processing halo {hid}")

            except KeyError as e:
                print(f"KeyError for halo {hid}: {e}")
                continue


if __name__ == "__main__":
    main()