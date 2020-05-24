import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import glob
import sys
from imports.utils import *
from imports.models import *
from HPC import *
from imports.adabound import AdaBound 
import time
import argparse

parser = argparse.ArgumentParser(description='parameter assign') 

parser.add_argument('-cuda', required=True, type=int, help='choice of cuda device. type: int')
parser.add_argument('-pulse', required=True, type=int, help='CPMG pulse (N). type: int')
parser.add_argument('-width', required=True, type=int, help='image width. type: int')
parser.add_argument('-time', required=True, type=int, help='number of data points used. type: int')
parser.add_argument('-bmin', required=True, type=int, help='minimum boundary of B (Hz). type: int')
parser.add_argument('-bmax', required=True, type=int, help='maximum boundary of B (Hz). type: int')
parser.add_argument('-aint', required=True, type=int, help='initial value of A (Hz) range in the whole model. type: int')
parser.add_argument('-afinal', required=True, type=int, help='final value of A (Hz) range in the whole model. type: int')
parser.add_argument('-arange', required=True, type=int, help='coverage range of a model. type: int')
parser.add_argument('-astep', required=True, type=int, help='distance between each model. type: int')
parser.add_argument('-noise', required=True, type=float, help='maxmum noise value (scale: M value). type: float')
parser.add_argument('-path', required=True, type=str, help='name of save directory for prediction files. type: float')
parser.add_argument('-existspin', required=True, type=int, help='if there is a list of the existing spins = 1, if not = 0 type: int')
'''
Excution Example) 
python -cuda 1 -pulse 32 -width 10 -time 7000 -bmin 20000 -bmax 80000 -aint 10000 -afinal 10500 -arange 200 -astep 250 -noise 0.05 -path temp_dir
'''

args = parser.parse_args()

CUDA_DEVICE = args.cuda
N_PULSE = args.pulse
IMAGE_WIDTH = args.width
TIME_RANGE  = args.time
EXISTING_SPINS = args.existspin

A_init  = args.aint   
A_final = args.afinal    
A_step  = args.astep      
A_range = args.arange    
B_init  = args.bmin       
B_final = args.bmax       
noise_scale = args.noise  
SAVE_DIR_NAME = str(args.path)

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE, EXISTING_SPINS, A_init, A_final, A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists)

hpc_model = HPC_Model(*args)
total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
