from pythonjsonlogger import exception
from scipy import stats
import pynbody
import warnings
import traceback
import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable

from scipy.optimize import curve_fit
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator
import pynbody
import pymp
import pynbody
import pymp
import re
import os
import sys
import gc
import pickle
import traceback
import numpy as np
from config import (db_connection, sys_path, pickle_file, ba_s_key, ca_s_key,
                    results_output_directory)

os.environ['TANGOS_DB_CONNECTION'] =db_connection
os.environ['TANGOS_PROPERTY_MODULES'] = 'mytangosproperty'
sys.path.append(sys_path)
import tangos
file = 'ellipse_data_n_100.pickle'
with open(file,'rb') as f:
    ellipse_dict = pickle.load(f)

print(ellipse_dict.keys())
