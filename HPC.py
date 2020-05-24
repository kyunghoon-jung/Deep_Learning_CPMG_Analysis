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

AB_lists_dic = np.load('./data/AB_target_dic_v4.npy').item() # a pre-calculated grouped list of nuclear spins with the same target period.

PRE_PROCESS = False
PRE_SCALE = 1

MAGNETIC_FIELD = 403.553                        # # The external magnetic field strength. Unit: Gauss
GYRO_MAGNETIC_RATIO = 1.0705*1000               # Unit: Herts 
WL_VALUE = MAGNETIC_FIELD*GYRO_MAGNETIC_RATIO*2*np.pi 

class HPC_Model():

    def __init__(self, *args):
        
        self.CUDA_DEVICE, self.N_PULSE, self.IMAGE_WIDTH, self.TIME_RANGE, self.EXISTING_SPINS, \
            self.A_init, self.A_final, self.A_step, self.A_range, self.B_init, self.B_final, \
                self.noise_scale, self.SAVE_DIR_NAME, self.model_lists = args

        self.exp_data = np.load('./data/exp_data_{}.npy'.format(self.N_PULSE)).flatten()  # the experimental data to be evalutated
        self.exp_data_deno = np.load('./data/exp_data_{}_deno.npy'.format(self.N_PULSE))  # the denoised experimental data to be evalutated
        self.time_data = np.load('./data/time_data_{}.npy'.format(self.N_PULSE))          # the time data for the experimental data to be evalutated
        self.spin_bath = np.load('./data/spin_bath_M_value_N{}.npy'.format(self.N_PULSE)) # the spin bath data for the experimental N_PULSE (it is not pre-requisite so one can just ignore this line.)
        self.total_indices = np.load('./data/total_indices_v4_N{}.npy'.format(N_PULSE)).item() 

    def binary_classification_train(self):

        if self.SAVE_DIR_NAME:
            deno_pred_N32_B15000_above = np.load('./data/predicted_results_N32_B15000above.npy') 
        total_raw_pred_list = []
        total_deno_pred_list = []
        total_A_lists = []

        for model_idx, [A_first, A_end, B_first, B_end] in enumerate(self.model_lists):
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
            B_target_gap = 0
            A_target_margin = 25
            A_side_margin = 300
            A_far_side_margin = 5000
            class_num = A_num*B_num + 1
            side_candi_num = 5             # the number of "how many times" to generate 'AB_side_candidate'

            image_width = self.IMAGE_WIDTH 
            class_num = A_num*B_num + 1
            cpu_num_for_multi = 20
            batch_for_multi = 256
            class_batch = cpu_num_for_multi*batch_for_multi

            spin_zero_scale = {'same':0.5, 'side':0.20, 'mid':0.05, 'far':0.05}

            torch.cuda.set_device(device=self.CUDA_DEVICE) 
            epochs = 15
            valid_batch = 4096
            valid_mini_batch = 1024

            if self.N_PULSE==32:
                B_side_min, B_side_max = 6000, 70000
                B_side_gap = 5000
                B_target_gap = 1000
                distance_btw_target_side = 1000

            elif self.N_PULSE==256:
                B_side_min, B_side_max = 1500, 25000
                B_side_gap = 100  # distance between target and side (applied for both side_same and side)
                B_target_gap = 0  # distance between targets only valid when B_num >= 2.
                distance_btw_target_side = 375

            if ((self.N_PULSE == 32) & (B_first<12000)):
                PRE_PROCESS = True
                PRE_SCALE = 8  
                print("==================== PRE_PROCESSING:True =====================")

            args = (AB_lists_dic, self.N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
                        B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
                        class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num) 

            TPk_AB_candi, Y_train_arr, _  = gen_TPk_AB_candidates(AB_idx_set, False, *args)
            if self.EXISTING_SPINS:
                TPk_AB_candi = return_existing_spins_wrt_margins(deno_pred_N32_B15000_above, TPk_AB_candi, A_existing_margin, B_existing_margin)

            model_index1 = get_model_index(total_indices, AB_idx_set[0][0], time_thres_idx=self.TIME_RANGE-20, image_width=self.IMAGE_WIDTH) 
            model_index2 = get_model_index(total_indices, AB_idx_set[-1][0], time_thres_idx=self.TIME_RANGE-20, image_width=self.IMAGE_WIDTH) 
            cut_idx = min(model_index1.shape[0], model_index2.shape[0])

            X_train_arr = np.zeros((class_num, len(AB_idx_set)*class_batch, cut_idx, 2*self.IMAGE_WIDTH+1))

            for idx1, [A_idx, B_idx] in enumerate(AB_idx_set):
                model_index = get_model_index(total_indices, A_idx, time_thres_idx=self.TIME_RANGE-20, image_width=self.IMAGE_WIDTH)
                model_index = model_index[:cut_idx, :]
                for class_idx in range(class_num):
                    for idx2 in range(cpu_num_for_multi):
                        AB_lists_batch = TPk_AB_candi[class_idx, idx1*class_batch+idx2*batch_for_multi:idx1*class_batch+(idx2+1)*batch_for_multi]
                        globals()["pool_{}".format(idx2)] = pool.apply_async(gen_M_arr_batch, [AB_lists_batch, model_index, time_data[:self.TIME_RANGE], 
                                                                                                WL_VALUE, self.N_PULSE, PRE_PROCESS, PRE_SCALE, 
                                                                                                noise_scale, spin_bath[:self.TIME_RANGE]])

                    for idx3 in range(cpu_num_for_multi):  
                        X_train_arr[class_idx, idx1*class_batch+idx3*batch_for_multi:idx1*class_batch+(idx3+1)*batch_for_multi] = globals()["pool_{}".format(idx3)].get(timeout=None) 
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

            MODEL_PATH = './data/models/' + self.SAVE_DIR_NAME + '/'
            if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)

            mini_batch_list = [2048]  
            learning_rate_list = [5e-6] 
            op_list = [['Adabound', [30,15,7,1]]] 
            criterion = nn.BCELoss().cuda()
            hyperparameter_set = [[mini_batch, learning_rate, selected_optim_name] for mini_batch, learning_rate, selected_optim_name in itertools.product(mini_batch_list, learning_rate_list, op_list)]
            print("==================== A_idx: {}, B_idx: {} ======================".format(A_first, B_first))

            total_loss, total_val_loss, total_acc, trained_model = train(MODEL_PATH, self.N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion,
                                                                        epochs, valid_batch, valid_mini_batch, self.exp_data, is_pred=False, is_print_results=False, is_preprocess=PRE_PROCESS, PRE_SCALE=PRE_SCALE,
                                                                        model_index=model_index, exp_data_deno=self.exp_data_deno)
            min_A = np.min(np.array(AB_idx_set)[:,0])
            max_A = np.max(np.array(AB_idx_set)[:,0])

            model.load_state_dict(torch.load(trained_model[0][0])) 

            total_A_lists, total_raw_pred_list, total_deno_pred_list = HPC_prediction(model, AB_idx_set, total_indices, self.TIME_RANGE, self.IMAGE_WIDTH, cut_idx, self.exp_data, self.exp_data_deno, 
                                                                                        total_A_lists, total_raw_pred_list, total_deno_pred_list, save_to_file=False)

        total_raw_pred_list  = np.array(total_raw_pred_list).T
        total_deno_pred_list = np.array(total_deno_pred_list).T

        np.save(MODEL_PATH+'total_N{}_A_idx.npy'.format(self.N_PULSE), total_A_lists)
        np.save(MODEL_PATH+'total_N{}_raw_pred.npy'.format(self.N_PULSE), total_raw_pred_list)
        np.save(MODEL_PATH+'total_N{}_deno_pred.npy'.format(self.N_PULSE), total_deno_pred_list)

        print('================================================================')
        print('Training Completed. Parsing parameters as follows.')
        print('N:{}, A_init:{}, A_final:{}, A_range:{}, A_step:{}, B_init:{}, B_final:{}, Image Width:{}, Time range:{}, noise:{}'.format(self.N_PULSE, 
                                                            self.A_init, self.A_final, self.A_range, self.A_step, self.B_init, self.B_final, self.IMAGE_WIDTH, self.TIME_RANGE, self.noise_scale))
        print('================================================================')
        print('total_time_consumed', time.time() - tic)

        return total_A_lists, total_raw_pred_list, total_deno_pred_list