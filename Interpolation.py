import pynbody
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TANGOS_DB_CONNECTION'] = '/home/bk639/data_base/CDM_all.db'
#TANGOS_PROPERTY_MODULES=shapesproperty
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
#add python path /home/bk639/MorphologyMeasurements/Code/tangos
import sys
sys.path.append('/home/bk639/mytangosproperty')
import tangos
sims = tangos.all_simulations()

#gather all halos

halos = []
for sim in sims:
    for halo in sim.timesteps[-1].halos:
        halos.append(halo)

