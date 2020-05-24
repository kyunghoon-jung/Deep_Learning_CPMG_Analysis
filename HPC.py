import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from imports.utils import *
from imports.models import *
from imports.adabound import AdaBound 
import time
import argparse
np.set_printoptions(suppress=True)

from multiprocessing import Pool 
POOL_PROCESS = 23  
FILE_GEN_INDEX = 2 
pool = Pool(processes=POOL_PROCESS)  

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import itertools

class HPC_model():

    def __init__(self, *agrs):

        self.exp_data = np.load('./data/exp_data_{}.npy'.format(N_PULSE)).flatten()  # the experimental data to be evalutated
        self.exp_data_deno = np.load('./data/exp_data_{}_deno.npy'.format(N_PULSE))  # the denoised experimental data to be evalutated
        self.time_data = np.load('./data/time_data_{}.npy'.format(N_PULSE))          # the time data for the experimental data to be evalutated
        self.spin_bath = np.load('./data/spin_bath_M_value_N{}.npy'.format(N_PULSE)) # the spin bath data for the experimental N_PULSE (it is not pre-requisite so one can just ignore this line.)
        self.total_indices = np.load('./data/total_indices_v4_N{}.npy'.format(N_PULSE)).item() 