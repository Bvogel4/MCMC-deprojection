#this code pull the isophote fits from tangos, and get's only the ones near a desired reff radius.
#plots these ellipses on top of v-band images in folder 'figures'
#lastly, saves these ellipses in a dict, for easy collection in MCMC codes. 


import os
import importlib
import os
from logging import exception
import matplotlib.colors as mcolors
import colorsys
import sys

import matplotlib.pyplot as plt
import numpy as np


# Set environment variables
from config import db_connection, sys_path
os.environ['TANGOS_DB_CONNECTION'] = db_connection
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos
sims = tangos.all_simulations()
import mytangosproperty

#function to plot isophotes on images
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


import pickle
import os
import numpy as np
band = 'v'
image_type = 'stars'
# Define pickle filename using a consistent pattern
pickle_filename = f'ellipse_data_{band}_{image_type}.pickle'

# Check if pickle file exists and load it if it does
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)
    print(f"Loaded existing data from {pickle_filename}")
else:
    ellipse_dict = {}
    print(f"Starting with fresh data")
    
ellipse_dict = {}


def plot_isophotes_testing(images, isophote_params, orientations, reffs, rhalf, filename,
                   max_grad_err=10, max_center_dist_factor=0.5, sma_tolerance=0.3,
                   relaxed_grad_err=1000):
    """
    Plot isophotes with intelligent selection based on gradient error within a radius range.

    Parameters:
    -----------
    images : list of 2D arrays
        The galaxy images
    isophote_params : list of lists
        Isophote parameters for each image
    orientations : list of str
        Orientation labels for each image
    reffs : list of float
        Effective radius for each image in kpc
    rhalf : float
        Half-light radius in kpc
    filename : str
        Output filename
    max_grad_err : float, optional
        Maximum acceptable gradient error (strict criteria)
    max_center_dist_factor : float, optional
        Factor to multiply with reff for maximum center distance
    sma_tolerance : float, optional
        Tolerance for semi-major axis match as fraction of radius (0.5 = ±50%)
    relaxed_grad_err : float, optional
        Relaxed gradient error threshold when no isophotes meet strict criteria
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # Calculate extent and pixel scale
    extent = 9 * rhalf  # in kpc
    kpc_per_pixel = extent / images[0].shape[0]

    # Create figure and axes
    fig, axs = plt.subplots(6, 12, figsize=(30, 15))
    fig.patch.set_facecolor('black')
    axs = axs.flatten()

    # Add selection criteria information at the top of the figure
    fig.suptitle(
        f"Selection Criteria: Primary Max Gradient Error = {max_grad_err:.2f}, "
        f"Relaxed Gradient Error = {relaxed_grad_err:.2f}, "
        f"Center Distance < {max_center_dist_factor}×Reff, "
        f"SMA Tolerance = ±{sma_tolerance}×Reff",
        color='white', fontsize=16, y=0.98
    )

    ellipse_dict = {}
    missing_isophote_count = {2: 0, 3: 0, 4: 0}  # Track missing isophotes for each radius

    # Process each image
    for i in range(len(images)):
        # Set up image display
        vmin = max(-1, np.min(np.log10(images[i])))
        axs[i].imshow(np.log10(images[i]), cmap='magma', origin='lower', vmin=vmin)

        # Convert effective radius to pixels
        reff_px = reffs[i] / kpc_per_pixel

        # Extract isophote parameters
        iso_params = isophote_params[i]

        # Initialize arrays to store ellipse parameters at 2, 3, 4 × reff
        ellipses = np.ones(3) * np.nan

        # Process isophotes at different radii (2, 3, 4) * reff
        for k_idx, k in enumerate([2, 3, 4]):
            target_radius = k * reff_px
            target_min = target_radius - sma_tolerance
            target_max = target_radius  + sma_tolerance

            # Create lists to store candidate isophotes within the radius range
            strict_candidates = []
            relaxed_candidates = []

            # Find all valid isophotes within the radius range
            for j in range(len(iso_params)):
                sma, eps, pa, grad_err, x0, y0, intens, rms = iso_params[j]

                # Calculate center offset
                center_offset = np.sqrt((images[i].shape[0] // 2 - x0) ** 2 +
                                        (images[i].shape[1] // 2 - y0) ** 2)

                # Check signal-to-noise ratio
                sn_ratio = intens / rms if rms > 0 else 0

                # Maximum allowed center distance
                max_center_dist = max_center_dist_factor * reff_px

                # Check if isophote is within radius range and meets center distance criterion
                if (target_min <= sma <= target_max and
                        center_offset < max_center_dist):

                    # Separate into strict and relaxed candidates
                    if grad_err < max_grad_err:
                        strict_candidates.append((sma, eps, pa, grad_err, x0, y0, intens, rms))
                    elif grad_err < relaxed_grad_err:
                        relaxed_candidates.append((sma, eps, pa, grad_err, x0, y0, intens, rms))

            # Select the best candidate
            best_candidate = None
            is_relaxed = False

            if strict_candidates:
                # Use the strict candidate with lowest gradient error
                strict_candidates.sort(key=lambda x: x[3])  # Sort by gradient error (index 3)
                best_candidate = strict_candidates[0]  # Pick the one with lowest error
            elif relaxed_candidates:
                # Use relaxed criteria if no strict candidates are found
                relaxed_candidates.sort(key=lambda x: x[3])  # Sort by gradient error (index 3)
                best_candidate = relaxed_candidates[0]  # Pick the one with lowest error
                is_relaxed = True

            # If no candidates found at all, mark as missing and continue
            if best_candidate is None:
                missing_isophote_count[k] += 1

                # Draw a marker indicating missing isophote
                center_x, center_y = images[i].shape[0] // 2, images[i].shape[1] // 2
                axs[i].scatter(center_x, center_y + target_radius, marker='x', s=50,
                               color='red', alpha=0.7)

                # Add text explaining the issue
                axs[i].text(center_x, center_y + target_radius + 10,
                            f"No valid {k}R", color='red', fontsize=8, ha='center')

                continue

            # Extract parameters for the best isophote
            sma, eps, pa, grad_err, x0, y0, intens, rms = best_candidate

            # Set line style based on whether it's a relaxed criterion and signal-to-noise
            sn_ratio = intens / rms if rms > 0 else 0

            if is_relaxed:
                linestyle = 'dotted'
                linewidth = 2.0
                color_scale = 0.5  # Desaturate color for relaxed criteria
            else:
                linestyle = '-'
                linewidth = 1.5
                color_scale = 1.0

            # Create colormap for gradient error
            norm = plt.Normalize(vmin=0, vmax=relaxed_grad_err)
            cmap = plt.cm.winter
            color = cmap(norm(grad_err))

            # Apply desaturation for relaxed criteria
            if is_relaxed:
                # Convert to desaturated color
                color_rgb = mcolors.to_rgb(color)
                h, l, s = colorsys.rgb_to_hls(*color_rgb)
                desaturated_rgb = colorsys.hls_to_rgb(h, l, s * color_scale)
                color = desaturated_rgb

            # Add ellipse
            ellipse = Ellipse(
                (x0, y0), 2 * sma, 2 * sma * (1 - eps),
                angle=np.degrees(pa),
                edgecolor=color,
                facecolor='none',
                linestyle=linestyle,
                linewidth=linewidth
            )
            axs[i].add_patch(ellipse)
            ellipses[k_idx] = eps

            theta = k  # Position the label at 45 degrees
            # Adjust for the ellipse's rotation angle
            adjusted_theta = theta - np.radians(np.degrees(pa))
            # Calculate coordinates on the ellipse
            label_x = x0 + sma * np.cos(adjusted_theta)
            label_y = y0 + sma * (1 - eps) * np.sin(adjusted_theta)

            # Add the radius label on the ellipse
            axs[i].text(label_x, label_y, f"{k}R", color='white', fontsize=8,
                        ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=1),
                        zorder=10)  # zorder ensures the text appears on top

            # Add small text showing radius, error, and status
            # status_text = "Relaxed" if is_relaxed else "Primary"
            # if k == 2:
            #     axs[i].text(
            #         x0, y0,
            #         f"{k}R, ε:{grad_err:.2f}\n{status_text}",
            #         color='white', fontsize=6,
            #         ha='center', va='center',
            #         bbox=dict(facecolor='black', alpha=0.5, pad=1)
            #     )

        # Save ellipses to dictionary
        ellipse_dict[orientations[i]] = ellipses

        # Format plot
        axs[i].axis('off')
        axs[i].set_aspect('equal')
        axs[i].set_title(f'{orientations[i]}', color='white', y=0.85)

    # Add a legend for isophote types
    legend_elements = [
        plt.Line2D([0], [0], color='turquoise', lw=1.5, linestyle='-', label='Primary Criteria'),
        plt.Line2D([0], [0], color='turquoise', lw=2.0, linestyle='dotted', alpha=0.7, label='Relaxed Criteria'),
        plt.Line2D([0], [0], color='red', marker='x', lw=0, label='No Valid Isophote')
    ]
    leg = fig.legend(handles=legend_elements, loc='upper right',
                     bbox_to_anchor=(0.91, 0.96), frameon=True, facecolor='black',
                     edgecolor='white', framealpha=0.7)
    for text in leg.get_texts():
        text.set_color('white')

    # Add a color bar to show gradient error scale with two regions
    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    bounds = np.linspace(0, relaxed_grad_err, 100)
    norm = plt.Normalize(vmin=0, vmax=relaxed_grad_err)

    # Create a custom colormap with a threshold indicator
    colors = plt.cm.winter(np.linspace(0, 1, 100))
    # Add a subtle visual cue for the threshold
    threshold_idx = int(max_grad_err / relaxed_grad_err * 100)
    colors[threshold_idx:, :] = colors[threshold_idx:, :] * 0.8  # Slightly desaturate colors above threshold
    custom_cmap = mcolors.ListedColormap(colors)

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, boundaries=bounds, ticks=[0, max_grad_err, relaxed_grad_err])
    cbar.set_label('Gradient Error', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

    # Add threshold line and labels to colorbar
    cbar.ax.axhline(y=max_grad_err, color='white', linestyle='--', linewidth=0.8)
    cbar.ax.text(1.5, max_grad_err, 'Primary\nThreshold', color='white',
                 fontsize=8, ha='left', va='center')

    # Print summary of missing isophotes
    total_images = len(images)
    missing_percent = {k: (count / total_images * 100) for k, count in missing_isophote_count.items()}

    fig.text(0.5, 0.01,
             f"Missing Isophotes: {missing_percent[2]:.1f}% at 2R, "
             f"{missing_percent[3]:.1f}% at 3R, "
             f"{missing_percent[4]:.1f}% at 4R",
             color='white', fontsize=12, ha='center')

    # Adjust layout and save figure
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=150)
    plt.close(fig)

    return ellipse_dict


def plot_isophotes(images, isophote_params, orientations, reffs, rhalf, filename):
    # 72 images, 72 isophote_params, 72 orientations
    extent = 9 * rhalf  # in kpc
    kpc_per_pixel = extent / images[0].shape[0]
    images_3d = np.array(images)

    fig, axs = plt.subplots(6, 12, figsize=(30, 15))
    fig.patch.set_facecolor('black')  # Set the figure background color

    axs = axs.flatten()
    ellipse_dict = {}
    center = (images[0].shape[0] // 2, images[0].shape[1] // 2)
    for i in range(len(images)):
        # print(orientations[i])

        vmin = np.min(np.log10(images[i]))
        if vmin < -1:
            vmin = -1
        axs[i].imshow(np.log10(images[i]), cmap='magma', origin='lower', vmin=vmin)
        reff = reffs[i]
        # convert reff to pixels
        reff = reff / kpc_per_pixel

        # plot isophotes
        iso_params = isophote_params[i]

        smas, epss, pas, grad_errs, x0s, y0s, intenss, rmss = [], [], [], [], [], [], [], []

        for j in range(len(iso_params)):
            sma, eps, pa, grad_err, x0, y0, intens, rms = iso_params[j]
            # print(f'sma: {sma}, eps: {eps}, pa: {pa}, grad_err: {grad_err}, x0: {x0}, y0: {y0}')
            # print(sma,eps,pa,grad_err,x0,y0)
            if grad_err < 0.15:
                smas.append(sma)
                epss.append(eps)
                pas.append(pa)
                grad_errs.append(grad_err)
                x0s.append(x0)  # remove center addition later
                y0s.append(y0)
                intenss.append(intens)
                rmss.append(rms)

        ellipses = np.ones(3) * np.nan
        for k in [2, 3, 4]:
            # boolean filter for grad_err <0.1

            # find index of sma closest to j*reff
            try:
                idx = (np.abs(np.array(smas) - k * reff)).argmin()
            except:
                # print(f"Available smas: {smas}")
                continue

            sma = smas[idx]

            # if sma is far, print
            # if np.abs(sma - k*reff) > 0.3*reff:
            #     print(f'smas: {sma:.2f}, reff: {k*reff:.2f}')
            eps = epss[idx]
            pa = pas[idx]
            grad_err = grad_errs[idx]
            x0 = x0s[idx]
            y0 = y0s[idx]
            intens = intenss[idx]
            rms = rmss[idx]

            # print(idx,sma,eps,pa,grad_err,x0,y0)
            center_offset = np.sqrt((images[i].shape[0] // 2 - x0) ** 2 + (images[i].shape[1] // 2 - y0) ** 2)
            # plot ellipse

            # get ellipse parameters
            # color by gradient error
            vmin = 0
            vmax = 0.15
            # create colormap
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.winter
            # if center_offset < 100:
            # print(sma,eps,pa,grad_err)
            # distance in pixels from 0.1kpc
            d = 0.5 / kpc_per_pixel * reff
            # set linestyle based on intensity
            if rms == 0:
                rms = 1e-3
            if intens / rms > 0.5 and intens > 1:
                linestyle = '-'
            else:
                linestyle = '--'

            if (center_offset < d) and (grad_err < 0.3):
                ellipse = Ellipse((x0, y0), 2 * sma, 2 * sma * (1 - eps), angle=np.degrees(pa),
                                  edgecolor=cmap(norm(grad_err)), facecolor='none',
                                  linestyle=linestyle, linewidth=1.5)
                axs[i].add_patch(ellipse)
                ellipses[k - 2] = eps
            else:
                ellipses[k - 2] = np.nan

        # save ellipses to dict
        ellipse_dict[orientations[i]] = ellipses

        axs[i].axis('off')
        axs[i].set_aspect('equal')
        axs[i].set_title(f'{orientations[i]}', color='white', y=0.85)
    # reduce white space
    plt.subplots_adjust(wspace=0, hspace=0)
    #make sure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=150)
    plt.close(fig)
    return ellipse_dict


for sim in sims:
    sim_name = str(sim.basename)
    print(f"Simulation {sim_name}")

    # Initialize dictionary for this sim if it doesn't exist
    if sim_name not in ellipse_dict:
        ellipse_dict[sim_name] = {}


    if len(sim.timesteps) > 1:
        timestep = sim.timesteps[-1]
    elif len(sim.timesteps) == 1:
        timestep = sim.timesteps[0]
    halos = timestep.halos[:100]

    #for Massive merians, only process largest halo
    if sim_name.startswith('r') and not sim_name.startswith('rogue'):
        #print(sim)
        max_halo = None
        max_stars = 0
        for i,halo in enumerate(halos):
            n_stars = halo['n_star'][0]
            if n_stars > max_stars:
                max_stars = n_stars
                max_halo = i
        halos = [halos[max_halo]]
        print('removing all halos except {} from sim {}'.format(halos[0].basename.split('_')[1],sim_name))
    elif sim_name.startswith('h'):
        print(f'removing halo 0 from sim {sim_name}')
        halos = halos[1:]

    for _, halo in enumerate(halos):
        # try:
            halo_name = halo.basename
            halo_ref = f'{sim_name}/%/{halo_name}'
            hid = halo_name.split('_')[1]

            # Skip if we've already processed this halo
            if halo_ref in ellipse_dict[sim_name].keys():
                print(f"Skipping halo {hid} - already processed")
                continue
                
            #print(halo['n_star'][0])
            if halo.calculate('NStar()') < 4000:
                continue
            
            print(f'Processing halo {hid} with {halo.calculate('NStar()')} stars')
            #get images and isophote

            halo_images = halo[f'halo_images_{band}']

            image_reffs = halo[f'image_reffs_{band}']
            image_orientations = halo[f'image_orientations_{band}']
            Rhalf = halo[f'Rhalf_{band}']
            isophote_params = halo[f'isophote_parameters_{band}_stars']

            reffs = np.array(image_reffs)
            #print(np.min(reffs),np.max(reffs),np.mean(reffs),np.std(reffs))
            filename = ('figures/' + str(sim.basename) +'.'+ str(hid)+ '.isophotes.png')

            halo_dict = plot_isophotes(halo_images, isophote_params, image_orientations, reffs, Rhalf, filename)
            #save to folder figures

            ellipse_dict[sim_name][halo_ref] = halo_dict
            #extract values out of halo_dict, list of length len(orientaions) containg a list of ellipse values of length 3
            ellipses = []
            for orientation in image_orientations:
                ellipses.append(halo_dict[orientation])
            #print(ellipses)

            #halo['ellipses'] = ellipses

            # Save to pickle file after each halo is processed
            with open(pickle_filename, 'wb') as f:
                pickle.dump(ellipse_dict, f)
            print(f"Saved data to {pickle_filename} after processing halo {hid}")
            
            #print(halo['isophote_parameters'])

        # except KeyError:
        #     #continue
        #     print('No isophote parameters')


#print all sims and hids
n_total = 0
for sim in ellipse_dict:
    halos_str = '\t'
    print(f'halos processed in sim {sim}:')
    for halo_ref in ellipse_dict[sim]:
        hid = halo_ref.split('_')[-1]
        halos_str = halos_str + f'{hid},'
        n_total = n_total + 1
    print(halos_str)
print(f'total numnber of halos {n_total}')
        
with open('ellipse_data.pickle', 'wb') as f:
    pickle.dump(ellipse_dict, f)


    
