import glob, sys, time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from imports.utils import *
from imports.models import *
from imports.adabound import AdaBound 

from multiprocessing import Pool 
POOL_PROCESS = 23  
FILE_GEN_INDEX = 2 
pool = Pool(processes=POOL_PROCESS)  

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import itertools
import argparse

print("Generation of datasets excuted.", time.asctime())
tic = time.time()
exp_data_256 = np.load('./data/exp_data_256.npy').flatten()
exp_data_256_deno = np.load('./data/exp_data_256_deno.npy') 
time_data_256 = np.load('./data/time_data_256.npy') 
spin_bath_256 = np.load('./data/spin_bath_M_value_N32.npy')
AB_lists_dic = np.load('./data/AB_target_dic_v4.npy').item()

PRE_PROCESS = False
PRE_SCALE = 1

MAGNETIC_FIELD = 403.553                        # The external magnetic field strength. Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.0705*1000               # Unit: Herts 
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi
parser = argparse.ArgumentParser(description='parameter assign')

parser.add_argument('-cuda', required=True, type=int, help='choice of cuda device. type: int')
parser.add_argument('-pulse', required=True, type=int, help='CPMG pulse (N). type: int')
parser.add_argument('-width', required=True, type=int, help='image width. type: int')
parser.add_argument('-time', required=True, type=int, help='number of data points used. type: int')
parser.add_argument('-bmin', required=True, type=int, help='minimum boundary of B (Hz). type: int')
parser.add_argument('-bmax', required=True, type=int, help='maximum boundary of B (Hz). type: int')
parser.add_argument('-aint', required=True, type=int, help='initial value of A (Hz) range in the whole model. type: int')
parser.add_argument('-afinal', required=True, type=int, help='final value of A (Hz) range in the whole model. type: int')
parser.add_argument('-astep', required=True, type=int, help='distance between models. type: int')
parser.add_argument('-arange', required=True, type=int, help='coverage range of a model. type: int')
parser.add_argument('-noise', required=True, type=float, help='maxmum noise value (scale: M value). type: float')
parser.add_argument('-path', required=True, type=str, help='name of save directory for prediction files. type: float')
'''
Excution Example) 
python -cuda 1 -pulse 256 -width 10 -time 7000 -bmin 20000 -bmax 80000 -aint 10000 -afinal 10500 -arange 250 -astep 200 -noise 0.05 -path temp_dir
'''
args = parser.parse_args()
CUDA_DEVICE = args.cuda
N_PULSE = args.pulse
total_indices = np.load('./data/total_indices_v4_N{}.npy'.format(N_PULSE)).item() 
IMAGE_WIDTH = args.width
TIME_RANGE  = args.time

A_init  = args.aint   
A_final = args.afinal    
A_step  = args.astep      
A_range = args.arange    
B_init  = args.bmin       
B_final = args.bmax       
noise_scale = args.noise  
SAVE_DIR_NAME = str(args.path)

model_lists = get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final)

A_existing_margin = 150
B_existing_margin = 2500
# the variable 'deno_pred_N32_B12000_above' is a list of spins predicted from N32 data
# intentionally included to make the HPC models more accurate for classification. 
deno_pred_N32_B12000_above = np.array([
    [-20738.524397906887, 40421.56414091587],
    [-8043.729442048509, 19196.62602543831],
    [36020.12688619586, 26785.71864962578],
    [11463.297802180363, 57308.602420240],
    [-24492.32775241693, 23001.877063512802]
])

tic = time.time()
for file_num in range(1):
    total_raw_pred_list = []
    total_deno_pred_list = []
    total_A_lists = []

    for model_idx, [A_first, A_end, B_first, B_end] in enumerate(model_lists):
        print("========================================================================")
        print('A_first:{}, A_end:{}, B_first:{}, B_end:{}'.format(A_first, A_end, B_first, B_end))
        print("========================================================================")
        A_num = 1
        B_num = 1
        A_resol, B_resol = 50, B_end-B_first+500

        A_idx_list = np.arange(A_first, A_end+A_resol, A_num*A_resol)
        if (B_end-B_first)%B_resol==0:
            B_idx_list = np.arange(B_first, B_end+B_resol, B_num*B_resol)
        else:
            B_idx_list = np.arange(B_first, B_end, B_num*B_resol)
        AB_idx_set = [[A_idx, B_idx] for A_idx, B_idx in itertools.product(A_idx_list, B_idx_list)]
        
        A_side_num = 8
        A_side_resol = 600
        B_side_min, B_side_max = 1500, 25000
        B_side_gap = 100
        B_target_gap = 0
        A_target_margin = 25
        A_side_margin = 300
        A_far_side_margin = 5000
        distance_btw_target_side = 375
        class_num = A_num*B_num + 1
        side_candi_num = 5             # the number of "how many times" to generate 'AB_side_candidate'

        image_width = IMAGE_WIDTH 
        time_range = TIME_RANGE
        class_num = A_num*B_num + 1
        cpu_num_for_multi = 20
        batch_for_multi = 256
        class_batch = cpu_num_for_multi*batch_for_multi

        spin_zero_scale = {'same':0.70, 'side':0.25, 'mid':0.05, 'far':0.05}

        torch.cuda.set_device(device=CUDA_DEVICE) 
        epochs = 15
        valid_batch = 4096
        valid_mini_batch = 1024

        args = (AB_lists_dic, N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                    B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                    class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num) 

        TPk_AB_candi, Y_train_arr  = gen_TPk_AB_candidates(AB_idx_set, False, *args)

        model_index1 = get_model_index(total_indices, AB_idx_set[0][0], time_thres_idx=time_range-20, image_width=image_width) 
        model_index2 = get_model_index(total_indices, AB_idx_set[-1][0], time_thres_idx=time_range-20, image_width=image_width) 
        cut_idx = min(model_index1.shape[0], model_index2.shape[0])

        X_train_arr = np.zeros((class_num, len(AB_idx_set)*class_batch, cut_idx, 2*image_width+1))
        X_train_TPk_arr = np.zeros((class_num, len(AB_idx_set)*class_batch, 1))

        for idx1, [A_idx, B_idx] in enumerate(AB_idx_set):
            model_index = get_model_index(total_indices, A_idx, time_thres_idx=time_range-20, image_width=image_width)
            model_index = model_index[:cut_idx, :]
            for class_idx in range(class_num):
                for idx2 in range(cpu_num_for_multi):
                    AB_lists_batch = TPk_AB_candi[class_idx, idx1*class_batch+idx2*batch_for_multi:idx1*class_batch+(idx2+1)*batch_for_multi]
                    globals()["pool_{}".format(idx2)] = pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index, time_data_256[:TIME_RANGE], 
                                                                                            WL_VALUE, N_PULSE, PRE_PROCESS, PRE_SCALE, 
                                                                                            noise_scale, spin_bath_256[:TIME_RANGE]])

                for idx3 in range(cpu_num_for_multi):  
                    X_train_arr[class_idx, idx1*class_batch+idx3*batch_for_multi:idx1*class_batch+(idx3+1)*batch_for_multi] = globals()["pool_{}".format(idx3)].get(timeout=None) 
                    X_train_TPk_arr[class_idx, idx1*class_batch+idx3*batch_for_multi:idx1*class_batch+(idx3+1)*batch_for_multi, 0] = A_idx
                print("_", end=' ') 

        X_train_arr = X_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, model_index.flatten().shape[0]) 
        Y_train_arr = Y_train_arr.reshape(class_num*len(AB_idx_set)*class_batch, class_num) 

        X_train_arr, Y_train_arr = shuffle(X_train_arr, Y_train_arr)
        model = HPC(X_train_arr.shape[1], Y_train_arr.shape[1]).cuda()
        try:
            model(torch.Tensor(X_train_arr[:5]).cuda()) 
        except:
            raise NameError("The input shape should be revised")

        total_parameter = sum(p.numel() for p in model.parameters()) 
        print('total_parameter: ', total_parameter / 1000000, 'M')

        MODEL_PATH = './data/models/' + SAVE_DIR_NAME + '/'
        if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)

        mini_batch_list = [2048]  
        learning_rate_list = [5e-6] 
        op_list = [['Adabound', [30,15,7,1]]] 
        criterion = nn.BCELoss().cuda()
        hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for mini_batch, learning_rate, selected_optim_name in itertools.product(mini_batch_list, learning_rate_list, op_list)]
        print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

        total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion,
                                                                    epochs, valid_batch, valid_mini_batch, exp_data_256, is_pred=False, is_print_results=False, is_preprocess=PRE_PROCESS, PRE_SCALE=PRE_SCALE,
                                                                    model_index=model_index, exp_data_deno=exp_data_256_deno)
        min_A = np.min(np.array(AB_idx_set)[:,0])
        max_A = np.max(np.array(AB_idx_set)[:,0])

        model.load_state_dict(torch.load(trained_model[0][0])) 
        model.eval()
        print("Model loaded as evalutation mode. Model path:", trained_model[0][0])

        raw_pred = []
        deno_pred = []
        A_pred_lists = []
        for idx1, [A_idx, B_idx] in enumerate(AB_idx_set):
            model_index = get_model_index(total_indices, A_idx, time_thres_idx=time_range-20, image_width=image_width)
            model_index = model_index[:cut_idx, :]
            exp_data_256_test = exp_data_256[model_index.flatten()]

            exp_data_256_test = 1-(2*exp_data_256_test - 1)
            exp_data_256_test = exp_data_256_test.reshape(1, -1)
            exp_data_256_test = torch.Tensor(exp_data_256_test).cuda()

            pred = model(exp_data_256_test)
            pred = pred.detach().cpu().numpy()

            A_pred_lists.append(A_idx)
            raw_pred.append(pred[0])

            total_A_lists.append(A_idx)
            total_raw_pred_list.append(pred[0])

            print(A_idx, np.argmax(pred), np.max(pred), pred)
            exp_data_256_test = exp_data_256_deno[model_index.flatten()]

            exp_data_256_test = 1-(2*exp_data_256_test - 1)
            exp_data_256_test = exp_data_256_test.reshape(1, -1)
            exp_data_256_test = torch.Tensor(exp_data_256_test).cuda()

            pred = model(exp_data_256_test)
            pred = pred.detach().cpu().numpy()
            deno_pred.append(pred[0])
            print(A_idx, np.argmax(pred), np.max(pred), pred)
            print() 

            total_deno_pred_list.append(pred[0])
        raw_pred = np.array(raw_pred).T
        deno_pred = np.array(deno_pred).T

        np.save(MODEL_PATH+'A_idx_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), A_pred_lists)
        np.save(MODEL_PATH+'raw_pred_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), raw_pred)
        np.save(MODEL_PATH+'deno_pred_{}_A{}-{}_B{}-{}'.format(model_idx, A_first, A_end, B_first, B_end), deno_pred)
        print('================================================================')
        print('================================================================')

    total_raw_pred_list  = np.array(total_raw_pred_list).T
    total_deno_pred_list = np.array(total_deno_pred_list).T

    np.save(MODEL_PATH+'total_N{}_A_idx_{}.npy'.format(N_PULSE, file_num), total_A_lists)
    np.save(MODEL_PATH+'total_N{}_raw_pred_{}.npy'.format(N_PULSE, file_num), total_raw_pred_list)
    np.save(MODEL_PATH+'total_N{}_deno_pred_{}.npy'.format(N_PULSE, file_num), total_deno_pred_list)
    print('================================================================')
    print('Training Completed. Parsing parameters as follows.')
    print('N:{}, A_init:{}, A_final:{}, A_range:{}, A_step:{}, B_init:{}, B_final:{}, Image Width:{}, Time range:{}, noise:{}'.format(N_PULSE, 
                                                        A_init, A_final, A_range, A_step, B_init, B_final, IMAGE_WIDTH, TIME_RANGE, noise_scale))
    print('================================================================')
print('total_time_consumed', time.time() - tic)