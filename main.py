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
parser.add_argument('-time32', required=True, type=int, help='number of data points used for evaluating N32. type: int')
parser.add_argument('-time256', required=True, type=int, help='number of data points used for evaluating N256. If no experimental data of N256: set as 0. type: int') 
parser.add_argument('-bmin', required=True, type=int, help='minimum boundary of B (Hz). type: int')
parser.add_argument('-bmax', required=True, type=int, help='maximum boundary of B (Hz). type: int')
parser.add_argument('-aint', required=True, type=int, help='initial value of A (Hz) range in the whole model. type: int')
parser.add_argument('-afinal', required=True, type=int, help='final value of A (Hz) range in the whole model. type: int')
parser.add_argument('-arange', required=True, type=int, help='coverage range of a model. type: int')
parser.add_argument('-astep', required=True, type=int, help='distance between each model. type: int')
parser.add_argument('-noise', required=True, type=float, help='maxmum noise value (scale: M value). type: float')
parser.add_argument('-path', required=True, type=str, help='name of save directory for prediction files. type: float')
parser.add_argument('-existspin', required=True, type=int, help='if there is a list of the existing spins: yes = 1, no = 0 type: int')
parser.add_argument('-evaluateall', required=True, type=int, help='if want to evaluate N32 and N256 at once: yes = 1, no = 0 type: int')
parser.add_argument('-distance', required=True, type=int, help='it determines the distance between the target spin and side spin (Hz): int')
parser.add_argument('-is_CNN', required=True, type=int, help='it determines whether the model is trained by CNN_structure: yes = 1 or no = 0: int')
parser.add_argument('-is_rm_index', required=True, type=int, help='it determines whether the varible "model_index" is reduced w.r.t. the existing spins: yes = 1 or no = 0: int')
'''
Excution Example) 
python3 -cuda 1 -pulse 32 -width 10 -time 7000 -bmin 20000 -bmax 80000 -aint 10000 \
        -afinal 10500 -arange 200 -astep 250 -noise 0.05 -path temp_dir -existspin 0 -evaluateall 0
'''

pars_args = parser.parse_args()

CUDA_DEVICE = pars_args.cuda
N_PULSE = pars_args.pulse
IMAGE_WIDTH = pars_args.width
TIME_RANGE_32  = pars_args.time32
TIME_RANGE_256  = pars_args.time256
EXISTING_SPINS = pars_args.existspin
EVALUATION_ALL = pars_args.evaluateall

target_side_distance = pars_args.distance
A_init  = pars_args.aint   
A_final = pars_args.afinal    
A_step  = pars_args.astep      
A_range = pars_args.arange    
B_init  = pars_args.bmin       
B_final = pars_args.bmax       
noise_scale = pars_args.noise  
SAVE_DIR_NAME = str(pars_args.path)
is_CNN = pars_args.is_CNN
is_remove_model_index = pars_args.is_rm_index

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
tic = time.time()
if pars_args.evaluateall:
    EXISTING_SPINS = 0
    A_init, A_final, B_init, B_final, target_side_distance = -50000, 50000, 12000, 80000, 50
    args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, 
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)

    hpc_model = HPC_Model(*args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    predicted_periods = return_filtered_A_lists_wrt_pred(total_deno_pred_list[1,:])

    regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final, 
                        A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    regression_model = Regression_Model(*regression_args)
    regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    # total_raw_pred_list, total_deno_pred_list 이 결과를 가지고 개수를 파악. 
    # regression_model = Regression_Model(*args)
    # regression_model.estimate_the_number_of_spins(A_lists)
    # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)
    # 여기서 얻은 것 중에, B=15000보다 큰 리스트를 저장해놓음. --> 아래 경로로. 
    # np.load('./data/predicted_results_N32_B15000above.npy') 

    EXISTING_SPINS = 1
    B_init, B_final = 6000, 12000
    model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, 
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    # regression_model = Regression_Model(*args)
    # regression_model.estimate_the_number_of_spins(A_lists)
    # regression_model.estimate_specific_AB_values(A_lists_with_the_number_of_spins)

    N_PULSE = 256
    EXISTING_SPINS = 1
    B_init, B_final = 2000, 15000 #### *** 단 여기는 N256에서 A가10보다 작을 때는 B_init = 1500으로 해야함!!
    model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)
    args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final, 
            A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    # total_raw_pred_list, total_deno_pred_list 이결과를 가지고 개수를 파악
    # regression_model = Regression_Model(*args)
    # regression_model.estimate_the_number_of_spins(total_raw_pred_list, total_deno_pred_list)
    # regression_model.estimate_specific_AB_values()
    print('Total computational time:', time.time() - tic)
    
else:
    if N_PULSE==32:
        args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, EXISTING_SPINS, A_init, A_final, 
                A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    elif N_PULSE==256:
        args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final, 
                A_step, A_range, B_init, B_final, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN, is_remove_model_index)
    hpc_model = HPC_Model(*args)
    total_A_lists, total_raw_pred_list, total_deno_pred_list = hpc_model.binary_classification_train()
    # print('Total computational time:', time.time() - tic)

    predicted_periods = return_filtered_A_lists_wrt_pred(np.array(total_deno_pred_list[1,:]), np.array(total_A_lists), 0.8)
    # predicted_periods = np.load('/home/sonic/Coding/Git/Paper_git_repo/Deep_Learning_CPMG_Analysis/data/models/predicted_periods.npy')
    # zero_scale = 0.
    # regression_args = (CUDA_DEVICE, N_PULSE, IMAGE_WIDTH, TIME_RANGE_32, TIME_RANGE_256, EXISTING_SPINS, A_init, A_final, 
    #                     A_step, A_range, B_init, B_final, zero_scale, noise_scale, SAVE_DIR_NAME, model_lists, target_side_distance, is_CNN)
    # regression_model = Regression_Model(*regression_args)
    # regression_results = regression_model.estimate_specific_AB_values(predicted_periods)
    # np.save('/home/sonic/Coding/Git/Paper_git_repo/Deep_Learning_CPMG_Analysis/data/models/regression_results.npy', regression_results)
