#this code pull the isophote fits from tangos, and get's only the ones near a desired reff radius.
#plots these ellipses on top of v-band images in folder 'figures'
#lastly, saves these ellipses in a dict, for easy collection in MCMC codes. 


import os
import importlib
import os
from logging import exception

import matplotlib.pyplot as plt
import numpy as np


# Set environment variables
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all.db'
os.environ['TANGOS_SIMULATION_FOLDER'] = '/home/bk639/data/CDM_z0'
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
#add python path /home/bk639/MorphologyMeasurements/Code/tangos
import sys
sys.path.append('/home/bk639/mytangosproperty')
import tangos
sims = tangos.all_simulations()
import mytangosproperty

#function to plot isophotes on images
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


import pickle
import os
import numpy as np

# Define pickle filename using a consistent pattern
pickle_filename = 'ellipse_data.pickle'

# Check if pickle file exists and load it if it does
if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        ellipse_dict = pickle.load(f)
    print(f"Loaded existing data from {pickle_filename}")
else:
    ellipse_dict = {}
    print(f"Starting with fresh data")
    
#ellipse_dict = {}




def plot_isophotes(images,isophote_params,orientations,reffs,rhalf,filename):
    #72 images, 72 isophote_params, 72 orientations
    extent = 9*rhalf #in kpc
    kpc_per_pixel = extent / images[0].shape[0]
    images_3d = np.array(images)

    
    fig, axs = plt.subplots(6, 12, figsize=(30, 15))
    fig.patch.set_facecolor('black')  # Set the figure background color
    
    axs = axs.flatten()
    ellipse_dict = {}
    center = (images[0].shape[0]//2,images[0].shape[1]//2)
    for i in range(len(images)):
        #print(orientations[i])
        
        vmin = np.min(np.log10(images[i]))
        if vmin < -1:
            vmin = -1
        axs[i].imshow(np.log10(images[i]), cmap='magma', origin='lower',vmin=vmin)
        reff = reffs[i]
        #convert reff to pixels 
        reff = reff / kpc_per_pixel
        
        #plot isophotes
        iso_params = isophote_params[i]
        
        smas,epss,pas,grad_errs,x0s,y0s,intenss,rmss = [],[],[],[],[],[],[],[]
        
        for j in range(len(iso_params)):
            sma,eps,pa,grad_err,x0,y0,intens,rms = iso_params[j]
            #print(f'sma: {sma}, eps: {eps}, pa: {pa}, grad_err: {grad_err}, x0: {x0}, y0: {y0}')
            #print(sma,eps,pa,grad_err,x0,y0)
            if grad_err < 0.15:
                smas.append(sma)
                epss.append(eps)
                pas.append(pa)
                grad_errs.append(grad_err)
                x0s.append(x0) #remove center addition later
                y0s.append(y0)
                intenss.append(intens)
                rmss.append(rms)
                
                
            
            
        ellipses = np.ones(3)*np.nan    
        for k in [2,3,4]:
            #boolean filter for grad_err <0.1


            #find index of sma closest to j*reff      
            try:
                idx = (np.abs(np.array(smas) - k*reff)).argmin()
            except:
                #print(f"Available smas: {smas}")
                continue

            sma = smas[idx]

            #if sma is far, print
            # if np.abs(sma - k*reff) > 0.3*reff:
            #     print(f'smas: {sma:.2f}, reff: {k*reff:.2f}')
            eps = epss[idx]
            pa = pas[idx]
            grad_err = grad_errs[idx]
            x0 = x0s[idx]
            y0 = y0s[idx]
            intens = intenss[idx]
            rms = rmss[idx]
            
            #print(idx,sma,eps,pa,grad_err,x0,y0)
            center_offset = np.sqrt((images[i].shape[0]//2 - x0)**2 + (images[i].shape[1]//2 - y0)**2)
            #plot ellipse
                        
            #get ellipse parameters
            #color by gradient error
            vmin = 0
            vmax = 0.15
            #create colormap
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.winter
            #if center_offset < 100:
            #print(sma,eps,pa,grad_err)
            #distance in pixels from 0.1kpc
            d = 0.5/kpc_per_pixel * reff
            #set linestyle based on intensity
            if intens/rms > 0.5 and intens > 1:
                linestyle = '-'
            else:
                linestyle = '--'

            if (center_offset < d) and (grad_err <0.3):
                ellipse = Ellipse((x0, y0), 2 * sma, 2 * sma * (1 - eps), angle=np.degrees(pa), edgecolor=cmap(norm(grad_err)), facecolor='none',
                                  linestyle=linestyle, linewidth=1.5)
                axs[i].add_patch(ellipse)
                ellipses[k-2] = eps
            else: 
                ellipses[k-2] = np.nan

        #save ellipses to dict
        ellipse_dict[orientations[i]] = ellipses


        axs[i].axis('off')
        axs[i].set_aspect('equal')
        axs[i].set_title(f'{orientations[i]}', color='white', y=0.85)
    #reduce white space
    plt.subplots_adjust(wspace=0, hspace=0)
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
        try:
            halo_name = halo.basename
            halo_ref = f'{sim_name}/%/{halo_name}'
            hid = halo_name.split('_')[1]

            # Skip if we've already processed this halo
            if halo_ref in ellipse_dict[sim_name].keys():
                print(f"Skipping halo {hid} - already processed")
                continue
                
            #print(halo['n_star'][0])
            if halo['n_star'][0] < 4000:
                continue
            
            print(f'Processing halo {hid} with {halo["n_star"][0]} stars')
            #get images and isophote
            halo_images = halo['halo_images']

            image_reffs = halo['image_reffs']
            image_orientations = halo['image_orientations']
            Rhalf = halo['Rhalf']
            isophote_params = halo['isophote_parameters']

            reffs = np.array(image_reffs)
            #print(np.min(reffs),np.max(reffs),np.mean(reffs),np.std(reffs))
            filename = ('figures/' + str(sim.basename) +'.'+ str(hid)+ '.isophotes.png')
            halo_dict = plot_isophotes(halo_images,isophote_params,image_orientations,reffs,Rhalf,filename)
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

        except KeyError:
            #continue
            print('No isophote parameters')


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


    
