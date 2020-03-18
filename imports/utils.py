import pandas as pd
import numpy as np
import os, sys, glob
import gc
import time
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# calculate M values using a single DD interaction model.
def M_list_return(time_table, wL_value, AB_list, n_pulse):  

    AB_list = np.array(AB_list)
    A = AB_list[:,0].reshape(len(AB_list), 1)     # (annotated output dimensions in following lines)
    B = AB_list[:,1].reshape(len(AB_list), 1)     # a,1     a = len(AB_list)
    
    w_tilda = pow(pow(A+wL_value, 2) + B*B, 1/2)  # a,1
    mz = (A + wL_value) / w_tilda                 # a,1
    mx = B / w_tilda                              # a,1

    alpha = w_tilda * time_table.reshape(1, len(time_table))    # a,b    b=len(time_table)
    beta = wL_value * time_table.reshape(1, len(time_table))    # 1,b

    phi = np.arccos(np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta))  # a,b
    K1 = (1 - np.cos(alpha)) * (1 - np.cos(beta))               # a,b
    K2 = 1 + np.cos(phi)                                        # a,b
    K = pow(mx,2) * (K1 / K2)                                   # a,b
    M_list_temp = 1 - K * pow(np.sin(n_pulse * phi/2), 2)       # a,b
    M_list = np.prod(M_list_temp, axis=0)

    return M_list

# return modified data by adding marginal_values to the original data
def get_marginal_arr(arr_data, margin_value, random_type='uni', mean=0, std=1):
    if random_type=='uni':
        return arr_data + np.random.uniform(low=-margin_value, high=margin_value, size=arr_data.shape) 
    elif random_type=='nor':
        return arr_data + (std*np.random.randn(size=arr_data.shape) - mean) 

# generate target (A,B) candidates with repect to B range from the target AB list(variable: target_AB). 
def gen_target_wrt_B(target_AB, class_batch, B_resol, B_start, B_num, B_target_gap): 
    target_lists = np.zeros((B_num, class_batch, 2)) 
    B_value_lists = np.arange(B_start, B_start+B_resol*B_num, B_resol) 
    for idx, B_temp in enumerate(B_value_lists): 
        indices = ((target_AB[:, 1] >= B_temp+B_target_gap) & (target_AB[:, 1] < (B_temp+B_resol-B_target_gap)))  
        target_temp = target_AB[indices]   
        indices = np.random.randint(target_temp.shape[0], size=class_batch)  
        target_lists[idx] = target_temp[indices] 
    return target_lists 

# generate side (A,B) candidates of the target period but of different B values. 
def gen_side_same_ABlist(target_AB, side_same_num, B_resol, B_start, B_num, B_side_gap, B_side_max): 
    indices = ((target_AB[:, 1] < B_start-B_side_gap) | ((target_AB[:, 1] > B_start+(B_resol*B_num)+B_side_gap) & (target_AB[:, 1]<B_side_max)))
    side_temp = target_AB[indices] 
    indices = np.random.randint(side_temp.shape[0], size=side_same_num) 
    side_same_AB_candi = side_temp[indices] 
    return side_same_AB_candi 

# generate side (A,B) candidates of the A range: range(25000, 55000, 10000) of B values with (B_side_min, B_side_max)
def gen_mid_AB_candi(A_start, AB_lists_dic, B_side_min, B_side_max, side_num): 
    idx_lists = np.arange(25000, 55000, 10000) 
    AB_mid_candi = np.zeros((2*idx_lists.shape[0], side_num, 2)) 
    for idx, temp_idx in enumerate(idx_lists): 
        try:
            temp_list = AB_lists_dic[A_start+temp_idx] 
            indices = ((temp_list[:,1]>B_side_min) & (temp_list[:,1]<B_side_max)) 
            AB_list_temp = temp_list[indices] 
            indices = np.random.randint(AB_list_temp.shape[0], size=side_num) 
            AB_mid_candi[2*idx] = AB_list_temp[indices] 
            temp_list = AB_lists_dic[A_start-temp_idx] 
            indices = ((temp_list[:,1]>B_side_min) & (temp_list[:,1]<B_side_max)) 
            AB_list_temp = temp_list[indices]
            indices = np.random.randint(AB_list_temp.shape[0], size=side_num)
            AB_mid_candi[2*idx+1] = AB_list_temp[indices]
            
        except IndexError:
            print("AB_mid_candi_IndexError: {}".format(temp_idx))
            return AB_mid_candi[:2*idx]
    return AB_mid_candi

# generate side (A,B) candidates of the A range: range(55000, 105000, 10000) of B values with (B_side_min, B_side_max)
def gen_far_AB_candi(A_start, AB_lists_dic, B_side_min, B_side_max, side_num): 
    idx_lists = np.arange(55000, 105000, 10000) 
    AB_far_candi = np.zeros((2*idx_lists.shape[0], side_num, 2)) 
    for idx, temp_idx in enumerate(idx_lists): 
        try:
            temp_list = AB_lists_dic[A_start+temp_idx] 
            indices = ((temp_list[:,1]>B_side_min) & (temp_list[:,1]<B_side_max)) 
            AB_list_temp = temp_list[indices] 
            indices = np.random.randint(AB_list_temp.shape[0], size=side_num) 
            AB_far_candi[2*idx] = AB_list_temp[indices] 

            temp_list = AB_lists_dic[A_start-temp_idx] 
            indices = ((temp_list[:,1]>B_side_min) & (temp_list[:,1]<B_side_max)) 
            AB_list_temp = temp_list[indices] 
            indices = np.random.randint(AB_list_temp.shape[0], size=side_num) 
            AB_far_candi[2*idx+1] = AB_list_temp[indices]
            
        except IndexError:
            print("AB_far_candi_IndexError: {}".format(temp_idx))
            return AB_far_candi[:2*idx]
    return AB_far_candi

# generate random (A, B) lists with batch size. (allowing for picking the same (A, B) candidates in the batch) 
def gen_random_lists_wrt_batch(AB_lists, batch):
    indices = np.random.randint(AB_lists.shape[0], size=batch)
    return AB_lists[indices] 

# generate (A, B) candidates with respect to the ratio for each B ranges.
def gen_divided_wrt_B_bounds(AB_lists, bound_list, side_num):
    total_AB_lists = []
    first_scale = 10 - np.sum(bound_list[:,1])
    for idx, [bound, scale] in enumerate(bound_list):
        if idx==0:
            indices, = np.where(AB_lists[:, 1] <= bound)
            if len(indices)==0:pass
            else:
                total_AB_lists.append(gen_random_lists_wrt_batch(AB_lists[indices], int(side_num*first_scale)))
            indices, = np.where((AB_lists[:, 1] > bound) & (AB_lists[:, 1] <= bound_list[idx+1][0]))
            if len(indices)==0:pass
            else:
                total_AB_lists.append(gen_random_lists_wrt_batch(AB_lists[indices], int(side_num*scale)))
        elif idx==(len(bound_list)-1):
            indices, = np.where(AB_lists[:, 1] > bound)
            if len(indices)==0:pass
            else:
                total_AB_lists.append(gen_random_lists_wrt_batch(AB_lists[indices], int(side_num*scale)))
        else:
            indices, = np.where((AB_lists[:, 1] > bound) & (AB_lists[:, 1] <= bound_list[idx+1][0]))
            if len(indices)==0:pass
            else:
                total_AB_lists.append(gen_random_lists_wrt_batch(AB_lists[indices], int(side_num*scale)))

    for idx, temp_AB in enumerate(total_AB_lists):
        if idx==0:
            total_flatten = temp_AB.copy()
        else:
            total_flatten = np.concatenate((total_flatten, temp_AB), axis=0)
    np.random.shuffle(total_flatten)
    return total_flatten

def gen_side_AB_candidates(*args):

    AB_lists_dic, A_side_idx_lists, side_num, B_side_min, B_side_max, bound_list, spin_zero_scale, class_num, class_batch, A_side_margin, side_candi_num = args
    
    for chosen_num in range(side_candi_num):
        side_AB_candi = np.zeros((A_side_idx_lists.shape[0], side_num, 2))
        for idx, A_side_idx in enumerate(A_side_idx_lists):
            indices = ((AB_lists_dic[A_side_idx][:, 1] > B_side_min) & (AB_lists_dic[A_side_idx][:, 1] < B_side_max))
            side_AB_temp = AB_lists_dic[A_side_idx][indices]
            indices = np.random.randint(side_AB_temp.shape[0], size=side_num)
            side_AB_candi[idx] = side_AB_temp[indices]
        side_AB_candi = side_AB_candi.reshape(-1, side_AB_candi.shape[-1])
        side_AB_candi[:,0] = get_marginal_arr(side_AB_candi[:,0], A_side_margin)
        side_AB_candi_divided = gen_divided_wrt_B_bounds(side_AB_candi, bound_list, side_num)

        scaled_batch = int(class_batch*(1-spin_zero_scale['side']))
        indices = np.random.randint(side_AB_candi_divided.shape[0], size=(class_num, scaled_batch)) 
        AB_candidates_side = side_AB_candi_divided[indices] 
        zero_candi = np.zeros((class_num, class_batch-scaled_batch, 2))
        AB_candidates_side = np.concatenate((AB_candidates_side, zero_candi), axis=1) 
        indices = np.random.randint(AB_candidates_side.shape[1], size=AB_candidates_side.shape[1])
        AB_candidates_side = AB_candidates_side[:, indices, :]
        
        if chosen_num == 0:
            total_AB_candidates_side = np.expand_dims(AB_candidates_side, -2)
        else:
            total_AB_candidates_side = np.concatenate((
                                            total_AB_candidates_side,
                                            np.expand_dims(AB_candidates_side, -2),
                                        ), axis=-2) 

    return total_AB_candidates_side

# generate (A, B) candidates with respect to the above configurations
def gen_AB_candidates(*args):
    AB_lists_dic, A_num, B_num, A_start, B_start, A_resol, B_resol, A_side_num, A_side_resol, B_side_min, \
    B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin, \
    class_batch, class_num, bound_list, spin_zero_scale,  \
    A_side_start_margin, A_side_start, A_side_end, side_same_num, side_num, side_candi_num = args

    A_idx_list = np.arange(A_start, A_start+A_resol*A_num, A_resol)  # generate A lists for each class
    target_AB_candi = np.zeros((class_num-1, class_batch, 2))

    side_same_target_candi = np.zeros((A_idx_list.shape[0], side_same_num, 2))
    for idx, A_idx in enumerate(A_idx_list):
        target_AB = AB_lists_dic[A_idx]
        target_AB_candi[B_num*idx:B_num*(idx+1)] = gen_target_wrt_B(target_AB, class_batch, B_resol, B_start, B_num, B_target_gap)
        if spin_zero_scale['same'] != 1.:
            side_same_target_candi[idx] = gen_side_same_ABlist(target_AB, side_same_num, B_resol, B_start, B_num, B_side_gap, B_side_max)
    side_same_target_candi = side_same_target_candi.reshape(-1, side_same_target_candi.shape[-1])
    side_same_target_candi[:,0] = get_marginal_arr(side_same_target_candi[:,0], A_side_margin)
    target_AB_candi[:,:,0] = get_marginal_arr(target_AB_candi[:,:,0], A_target_margin)

    A_side_idx_lists = np.hstack((np.arange(A_side_start, A_side_start-A_side_resol*A_side_num, -A_side_resol), 
                                  np.arange(A_side_end, A_side_end+A_side_resol*A_side_num, A_side_resol)))
    args = AB_lists_dic, A_side_idx_lists, side_num, B_side_min, B_side_max, bound_list, spin_zero_scale, class_num, class_batch, A_side_margin, side_candi_num
    AB_candidates_side = gen_side_AB_candidates(*args)

    AB_far_candi = gen_far_AB_candi(A_start+A_resol*(A_num//2), AB_lists_dic, B_side_min, B_side_max, side_num)
    AB_far_candi = AB_far_candi.reshape(-1, AB_far_candi.shape[2])
    AB_far_candi[:,0] = get_marginal_arr(AB_far_candi[:,0], A_far_side_margin)

    AB_mid_candi = gen_mid_AB_candi(A_start+A_resol*(A_num//2), AB_lists_dic, B_side_min, B_side_max, side_num)
    AB_mid_candi = AB_mid_candi.reshape(-1, AB_mid_candi.shape[2])
    AB_mid_candi[:,0] = get_marginal_arr(AB_mid_candi[:,0], A_far_side_margin)

    side_same_target_candi_divided = gen_divided_wrt_B_bounds(side_same_target_candi, bound_list, side_num) 
    AB_mid_candi_divided = gen_divided_wrt_B_bounds(AB_mid_candi, bound_list, side_num)
    AB_far_candi_divided = gen_divided_wrt_B_bounds(AB_far_candi, bound_list, side_num)

    scaled_batch = int(class_batch*(1-spin_zero_scale['mid']))
    indices = np.random.randint(AB_mid_candi_divided.shape[0], size=(class_num, scaled_batch)) 
    AB_candidates_mid = AB_mid_candi_divided[indices] 
    zero_candi = np.zeros((class_num, class_batch-scaled_batch, 2))
    AB_candidates_mid = np.concatenate((AB_candidates_mid, zero_candi), axis=1) 
    indices = np.random.randint(AB_candidates_mid.shape[1], size=AB_candidates_mid.shape[1])
    AB_candidates_mid = AB_candidates_mid[:, indices, :]

    scaled_batch = int(class_batch*(1-spin_zero_scale['far']))
    indices = np.random.randint(AB_far_candi_divided.shape[0], size=(class_num, scaled_batch)) 
    AB_candidates_far = AB_far_candi_divided[indices] 
    zero_candi = np.zeros((class_num, class_batch-scaled_batch, 2))
    AB_candidates_far = np.concatenate((AB_candidates_far, zero_candi), axis=1) 
    indices = np.random.randint(AB_candidates_far.shape[1], size=AB_candidates_far.shape[1])
    AB_candidates_far = AB_candidates_far[:, indices, :]

    scaled_batch = int(class_batch*(1-spin_zero_scale['same']))
    indices = np.random.randint(side_same_target_candi_divided.shape[0], size=(class_num, scaled_batch)) 
    AB_candidates_side_same = side_same_target_candi_divided[indices] 
    zero_candi = np.zeros((class_num, class_batch-scaled_batch, 2))
    AB_candidates_side_same = np.concatenate((AB_candidates_side_same, zero_candi), axis=1) 
    indices = np.random.randint(AB_candidates_side_same.shape[1], size=AB_candidates_side_same.shape[1])
    AB_candidates_side_same = AB_candidates_side_same[:, indices, :]

    indices = np.random.randint(AB_candidates_side.reshape(-1, 2).shape[0], size=(1, class_batch)) 
    AB_pseudo_target = AB_candidates_side.reshape(-1, 2)[indices]
    target_AB_candi = np.concatenate((target_AB_candi, AB_pseudo_target), axis=0)
    
    AB_candidates = np.concatenate((
        np.expand_dims(target_AB_candi, -2),
        np.expand_dims(AB_candidates_mid, -2),
        np.expand_dims(AB_candidates_far, -2),
        AB_candidates_side,
        np.expand_dims(AB_candidates_side_same, -2),
    ), axis=-2) 

    return AB_candidates

# generate model index with respect to the image_width and time index threshold
def get_model_index(total_indices, A_idx, *, time_thres_idx, image_width):
    for idx_cut, time_index in enumerate(total_indices[A_idx][1]):
        if time_index > time_thres_idx:
            break
    temp_index = total_indices[A_idx][1][:idx_cut]
    model_index = np.array([range(k-image_width, k+image_width+1) for k in temp_index])
    return model_index

# generate (A,B) candidates merging each TPk in 'AB_target_set' variable  
def gen_TPk_AB_candidates(AB_target_set: "[[A1,B1],[A2,B2],..]", is_hierarchical=False, *args) -> "total_AB_candi, total_Y_train_arr":

    AB_lists_dic, N_PULSE, A_num, B_num, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,\
    B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,\
    class_batch, class_num, spin_zero_scale, distance_btw_target_side, side_candi_num = args

    total_AB_candi = np.zeros((class_num, len(AB_target_set)*class_batch, 4+side_candi_num, 2)) 
                    # CAUTION: '3rd dimension' of 'total_AB_candi' is determined by the output of 'gen_AB_candidates' function
    total_Y_train_arr = np.zeros((class_num, len(AB_target_set)*class_batch, class_num))
    
    for target_index, [A_start, B_start] in enumerate(AB_target_set):

        if N_PULSE==32:
            if B_start<15000:
                bound_list = np.array([[5000, 4], [11000, 2], [18000, 1], [35000, 1]]) # the first ratio = 10 - all the rest of rates
                                                                                          # example) 10 - (4+2+1+1) = 2 -> 20%
            else:
                bound_list = np.array([[8000, 3], [15000, 3], [27000, 2.5], [45000, 1]]) 
        elif N_PULSE==64:
            bound_list = np.array([[7000, 3], [13500, 3], [27000, 2.5], [45000, 1]]) 
        elif N_PULSE==96:
            bound_list = np.array([[6000, 3], [12000, 3], [27000, 2.5], [45000, 1]]) 
        elif N_PULSE==256:
            bound_list = np.array([[1500, 4], [7500, 2], [17000, 1], [35000, 1]]) 
        
        if is_hierarchical==True:
            A_side_start_margin = A_target_margin + distance_btw_target_side  # 'distance' between target boundary and side boundary  
            A_side_start = AB_target_set[0][0] - A_side_start_margin                      # 'left' starting point of A side
            A_side_end = AB_target_set[-1][0] + A_resol*(A_num-1) + A_side_start_margin   # 'right' starting point of A side. 'A_start' means the center value of A_class
            num_of_targets = 3
        else:
            A_side_start_margin = A_target_margin + distance_btw_target_side  
            A_side_start = A_start - A_side_start_margin                      
            A_side_end = A_start + A_resol*(A_num-1) + A_side_start_margin    
        side_same_num = AB_lists_dic[A_start].shape[0]*10                 #  set 'how many side spins to be chosen' purposely to 10 times larger than AB_list
        side_num = AB_lists_dic[A_start].shape[0]*10                      #       because we use np.random.randint, which includes 'the same' number. 

        args_AB_candi = (AB_lists_dic, A_num, B_num, A_start, B_start, A_resol, B_resol, A_side_num, A_side_resol, B_side_min,
            B_side_max, B_target_gap, B_side_gap, A_target_margin, A_side_margin, A_far_side_margin,
            class_batch, class_num, bound_list, spin_zero_scale, 
            A_side_start_margin, A_side_start, A_side_end, side_same_num, side_num, side_candi_num)
        
        AB_candidates = gen_AB_candidates(*args_AB_candi)
        
        for total_idx in range(len(AB_candidates)):
            total_AB_candi[total_idx, target_index*class_batch:(target_index+1)*class_batch] = AB_candidates[total_idx]
            total_Y_train_arr[total_idx, target_index*class_batch:(target_index+1)*class_batch, total_idx] = 1

    return total_AB_candi, total_Y_train_arr 

# pre-processing of Px value 
def pre_processing(data: 'Px value', power=4):
    return 1-data**power

# generate M value array batch
def gen_M_arr_batch(AB_lists_batch, indexing, time_data, WL, PULSE, is_pre_processing=False, pre_process_scale=4, noise_scale=0., spin_bath=0., existing_spins_M=0):
    index_flat = indexing.flatten()
    X_train_arr_batch = np.zeros((len(AB_lists_batch), indexing.shape[0], indexing.shape[1]))
    if type(spin_bath) == float:
        for idx1, AB_list_temp in enumerate(AB_lists_batch):
            X_train_arr_batch[idx1,:] = M_list_return(time_data[index_flat]*1e-6, WL, AB_list_temp*2*np.pi, PULSE).reshape(indexing.shape)
    else:
        for idx1, AB_list_temp in enumerate(AB_lists_batch):
            X_train_arr_batch[idx1,:] = (spin_bath[index_flat]*M_list_return(time_data[index_flat]*1e-6, WL, AB_list_temp*2*np.pi, PULSE)).reshape(indexing.shape)
    X_train_arr_batch = X_train_arr_batch.reshape(len(AB_lists_batch), len(indexing), len(indexing[2]))
    if noise_scale>0:
        X_train_arr_batch += np.random.uniform(size=X_train_arr_batch.shape)*noise_scale - np.random.uniform(size=X_train_arr_batch.shape)*noise_scale
    
    if is_pre_processing==True:
        if type(existing_spins_M)==int:
            return pre_processing((1+X_train_arr_batch)/2, power=pre_process_scale)
        else:
            return pre_processing((1+X_train_arr_batch*existing_spins_M)/2, power=pre_process_scale)
    else:
        if type(existing_spins_M)==int:
            return 1-X_train_arr_batch
        else:
            return 1-X_train_arr_batch*existing_spins_M

# return A value of TPk from the input (A, B) candidate
def cal_TPk_from_AB(A: 'Hz', B: 'Hz', WL, k=10) -> "A(Hz)":

    A_temp = int(round(A*0.01)*100)
    A_list = np.arange(A_temp-15000, A_temp+15000, 50)*2*np.pi
    A *= 2*np.pi
    B *= 2*np.pi
    w_tilda = np.sqrt((A+WL)**2 + B**2)
    tau = np.pi*(2*k - 1) / (w_tilda + WL)
    B_ref = 10000*2*np.pi
    w_tilda_list = np.sqrt((A_list+WL)**2 + B_ref**2)
    tau_list = np.pi*(2*k - 1) / (w_tilda_list + WL)
    min_idx = np.argmin(np.abs(tau_list - tau))
    
    return int(round(A_list[min_idx]/2/np.pi, 0))

# return [A_start, A_end, B_start, B_end] lists for HPC models
def get_AB_model_lists(A_init, A_final, A_step, A_range, B_init, B_final):
    A_list1 = np.arange(A_init, A_final+A_step, A_step)
    A_list2 = A_list1 + A_range
    B_list1 = np.full(A_list1.shape, B_init)
    B_list2 = np.full(A_list1.shape, B_final)
    return np.stack((A_list1, A_list2, B_list1, B_list2)).T

# return merged spin lists by merging the existing spins with the reference spins (added margins)     
def return_existing_spins_wrt_margins(existing_spins, reference_spins, A_existing_margin, B_existing_margin):

    existing_spins = np.repeat(np.expand_dims(existing_spins, axis=0), reference_spins.shape[1], axis=0)
    existing_spins = np.repeat(np.expand_dims(existing_spins, axis=0), reference_spins.shape[0], axis=0)

    A_margin_arr = np.random.uniform(low=-A_existing_margin, high=A_existing_margin, size=(existing_spins.shape))
    B_margin_arr = np.random.uniform(low=-B_existing_margin, high=B_existing_margin, size=(existing_spins.shape))

    existing_spins[:,:,:,0] += A_margin_arr[:,:,:,0]
    existing_spins[:,:,:,1] += B_margin_arr[:,:,:,1]

    return np.concatenate((reference_spins, existing_spins), axis=-2)

# return reorganized HPC prediction results for the confirmation modelregarding the threshold 
def return_filtered_A_lists_wrt_pred(pred_result, A_idx_list, threshold=0.8):
    (indices, ) = np.where(pred_result>threshold) 
    grouped_indices = [] 
    temp = [] 
    for idx, index in enumerate(indices): 
        if idx==0: 
            if (indices[idx+1]-index)==1:
                temp.append(index) 
        else:
            if (index-indices[idx-1]==1) | (index-indices[idx-1]==2): 
                temp.append(index) 
            elif (((index-indices[idx-1]) > 2) & (len(temp)>=2)):
                grouped_indices.append(temp) 
                temp_index = index
                temp = []
                if idx!=(len(indices)-1):
                    if (indices[idx+1] - temp_index == 1) | (indices[idx+1] - temp_index == 2):
                        temp.append(temp_index)
            if (idx==(len(indices)-1)) & (len(temp)>2):
                grouped_indices.append(temp) 
    grouped_A_lists = [list(A_idx_list[temp_indices]) for temp_indices in grouped_indices if len(temp_indices) > 2]
    return grouped_A_lists 

# return prediction lists
def return_pred_list(path):
    deno_pred_list = glob.glob(path+'total_N*_deno*')
    raw_pred_list = glob.glob(path+'total_N*_raw*')
    A_idx_list = glob.glob(path+'total_N*A_idx*')
    A_idx_list = np.load(A_idx_list[0])
    for idx in range(len(deno_pred_list)):
        if idx==0: 
            avg_raw_pred = np.load(raw_pred_list[idx])
            avg_deno_pred = np.load(deno_pred_list[idx])
        else: 
            avg_raw_pred += np.load(raw_pred_list[idx])
            avg_deno_pred += np.load(deno_pred_list[idx])
    avg_raw_pred /= len(raw_pred_list)
    avg_deno_pred /= len(deno_pred_list)
    return A_idx_list, avg_raw_pred, avg_deno_pred

# calculate power exponential decay rates for time range 
def gaussian_slope_px(M_lists: "data of M values", time_table: "time data", time_index: "time index of calculated point", px_mean_value_at_time: "px value at time index"):

    m_value_at_time = (px_mean_value_at_time * 2) - 1
    Gaussian_co = -time_table[time_index] / np.log(m_value_at_time)

    slope = np.exp(-(time_table / Gaussian_co)**2)
    slope = slope.reshape(1, len(time_table))
    M_lists_slope = M_lists * slope
    px_lists_slope = (M_lists_slope + 1) / 2
    return px_lists_slope.squeeze(), slope.squeeze()