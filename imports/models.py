import torch
import torch.nn as nn
import time
import numpy as np
from adabound import AdaBound
import sys

def pre_processing(data: 'Px value', power=4):
    return 1-data**power

class HPC(nn.Module):
    def __init__(self, input_features, out_features): 
        super(HPC, self).__init__()
        self.linear1 = nn.Linear(input_features, 2048)   # 1st layer: 2048
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, out_features)
        self.bn1 = nn.BatchNorm1d(2048)   # 1st layer: 2048
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, n_input): 
        out = self.leaky(self.bn1(self.linear1(n_input)))
        out = self.leaky(self.bn2(self.linear2(out)))
        out = self.leaky(self.bn3(self.linear3(out))) 
        out = self.sigmoid(self.linear4(out))
        return out

class HPC_hierarchical(nn.Module):
    def __init__(self, input_features, out_features): 
        super(HPC_hierarchical, self).__init__()
        self.linear1 = nn.Linear(input_features, 2048)  
        self.linear2 = nn.Linear(2048, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 512)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.final = nn.Linear(512, out_features)
        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x = self.leaky(self.bn1(self.linear1(x)))
        x = self.leaky(self.bn2(self.linear2(x)))
        x = self.leaky(self.bn3(self.linear3(x)))
        x = self.leaky(self.bn4(self.linear4(x)))
        x = self.sigmoid(self.final(x))
        return x

class Fitting_model(nn.Module):
    def __init__(self, input_features, out_features): 
        super(Fitting_model, self).__init__()
        self.linear1 = nn.Linear(input_features, 2048)  
        self.linear2 = nn.Linear(2048, 2048)
        self.linear3 = nn.Linear(2048, 1024)
        self.linear4 = nn.Linear(1024, 512)
        
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.final = nn.Linear(512, out_features)
        self.leaky = nn.LeakyReLU()

    def forward(self, x): 
        x = self.leaky(self.bn1(self.linear1(x)))
        x = self.leaky(self.bn2(self.linear2(x)))
        x = self.leaky(self.bn3(self.linear3(x)))
        x = self.leaky(self.bn4(self.linear4(x)))
        x = self.final(x)
        return x

class Denoise_Model(nn.Module):
    def __init__(self):
        super(Denoise_Model, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, 64, 4, stride=1, padding=2)
        self.maxpooling = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1d_2 = nn.Conv1d(64, 64, 4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.convTrans1d_3 = nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.convTrans1d_4 = nn.ConvTranspose1d(64, 2, 4, stride=2, padding=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], 2, -1)
        x = self.maxpooling(self.leakyrelu(self.bn1(self.conv1d_1(x))))
        x = self.maxpooling(self.leakyrelu(self.bn2(self.conv1d_2(x))))
        x = self.leakyrelu(self.bn3(self.convTrans1d_3(x)))
        x = self.leakyrelu(self.convTrans1d_4(x))
        return x   

class HPC_CNN(nn.Module):
    def __init__(self, input_features: "torch.Size", output_features: int):
        super(HPC_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)      # Height = (H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)    
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)    
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)    
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)     
        self.conv6_bn = nn.BatchNorm2d(128) 
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)     
        self.conv7_bn = nn.BatchNorm2d(128) 

        self.leaky = nn.LeakyReLU() 
        self.elu = nn.ELU() 
        self.relu = nn.ReLU() 
        self.pooling = nn.MaxPool2d(2)
        self.num_pooling = 2
        self.linear_num = 128*(input_features[0]//(2**2))*(input_features[1]//(2**2)) # number of pooling
        self.fc1 = nn.Linear(self.linear_num, 256)  
        self.fc1_bn = nn.BatchNorm1d(256) 
        self.fc_final = nn.Linear(256, output_features) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
        out = self.leaky(self.conv2_bn(self.conv2(self.leaky(self.conv1_bn(self.conv1(x)))))) 
        out = self.leaky(self.conv4_bn(self.conv4(self.leaky(self.conv3_bn(self.conv3(out)))))) 
        out = self.pooling(self.leaky(self.conv7_bn(self.conv7(self.leaky(self.conv6_bn(self.conv6(self.leaky(self.conv5_bn(self.conv5(out)))))))))) 
        out = self.pooling(self.leaky(self.conv7_bn(self.conv7(out))))

        out = out.flatten().view(out.shape[0], -1)  
        out = self.leaky(self.fc3_bn(self.fc1(out))) 
        out = self.sigmoid(self.fc_final(out)) 
        return out 

def select_optimizer(model, optim_name, learning_rate):
    if optim_name=='SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate) 
    elif optim_name=='Adabound':
        return AdaBound(model.parameters(), lr=learning_rate) 
    elif optim_name=='Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate) 
    else:
        raise NameError("Please choose the optimizer in SGD, Adabound and Adam") 

# prediction with the trained model of the target data. model_index is required
def model_prediction(model, file_name, target_data, model_index, is_preprocess=False, PRE_SCALE=4):

    model.load_state_dict(torch.load(file_name))
    index_flat = model_index.flatten()
    target_data_flat = target_data[index_flat]
    target_data_flat = target_data_flat.reshape(len(model_index), len(model_index[2]))

    if is_preprocess==True:
        target_data_flat = pre_processing(2*target_data_flat-1, power=PRE_SCALE)
    else:
        target_data_flat = 2*target_data_flat-1
        target_data_flat = 1 - target_data_flat
    target_data_flat_test = torch.Tensor(target_data_flat).cuda().flatten()
    target_data_flat_test = target_data_flat_test.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        
        prediction = model(target_data_flat_test)  
        pred = prediction.detach().cpu().numpy()[0]

    return [pred, np.argmax(pred), np.max(pred)] 

def train(MODEL_PATH, N_PULSE, X_train_arr, Y_train_arr, model, hyperparameter_set, criterion, 
          epochs, valid_batch, valid_mini_batch, exp_data=0, is_pred=False, is_print_results=False, is_preprocess=False, PRE_SCALE=4,
          model_index=False, exp_data_deno=False):
    start_time = time.time()
    
    train_batch = X_train_arr.shape[0] - valid_batch 
    print("train_batch: ", train_batch, "valid_batch: ", valid_batch)
    
    total_pred = []
    trained_model_list = []
    for mini_batch, learning_rate, selected_optim_name in hyperparameter_set:
        file_name = MODEL_PATH+'_N{}_batch{}_lr{}_{}.pt'.format(N_PULSE, mini_batch, learning_rate, selected_optim_name[0])

        print("\n\n========================================================================================================\n Training Start: ", time.asctime()) 
        print(' mini_batch:', mini_batch, ' | learning_rate: ', learning_rate, ' | selected_optim_name: ', selected_optim_name, ' |')
        print("========================================================================================================")
        
        start = time.time()
        total_loss = []
        total_val_loss = [] 
        total_acc = []
        
        for epoch in range(epochs):
            model.train() 
            avg_cost = 0
            tic = time.time() 
            
            if epoch<4:
                optimizer = select_optimizer(model, selected_optim_name[0], learning_rate=learning_rate*selected_optim_name[1][epoch])
            
            for i in range(train_batch // mini_batch):
                train_indices = np.random.choice(train_batch, size=mini_batch)
                x_train_temp = torch.Tensor(X_train_arr[train_indices]).cuda()
                y_train_temp = torch.Tensor(Y_train_arr[train_indices]).cuda()

                optimizer.zero_grad() 
                hypothesis = model(x_train_temp) 
                cost = criterion(hypothesis, y_train_temp) 
                cost.backward() 
                optimizer.step() 

                avg_cost += cost
            
            loss_temp = avg_cost / (train_batch // mini_batch)
            total_loss.append(loss_temp.cpu().detach().item())
            print("Epoch:", '%4d' % (epoch + 1), ' | Loss =', '{:.5f}'.format(loss_temp), end=' | ')
            with torch.no_grad():
                model.eval()
                valid_indices = torch.randperm(valid_batch)[:valid_mini_batch] + train_batch
                X_valid = torch.Tensor(X_train_arr[valid_indices]).cuda()
                Y_valid = torch.Tensor(Y_train_arr[valid_indices]).cuda()

                prediction = model(X_valid)
                val_loss = criterion(prediction, Y_valid)

                bool_pred = torch.argmax(prediction, dim=1, keepdim=True)
                bool_y = torch.argmax(Y_valid, dim=1, keepdim=True)
                accuracy = torch.sum(bool_pred == bool_y).float() / len(Y_valid) * 100
                print('Val_loss: {:.5f} | Accuracy: {:.2f} %'.format(val_loss.item(), accuracy.item()), end=' | ')
                total_val_loss.append(val_loss.cpu().detach().item())
                total_acc.append(accuracy.cpu().detach().item())
                
            print("time: {}(s)".format(np.round(time.time() - tic,3)), end=' | ')
            total_val_loss = np.array(total_val_loss)
            if total_val_loss.min() >= total_val_loss[-1]:
                torch.save(model.state_dict(), file_name)
            else:
                if np.min(total_val_loss[-3:-1]) < total_val_loss[-1]:
                    optimizer = select_optimizer(model, selected_optim_name[0], learning_rate=learning_rate*0.35)
            print("lr: ", learning_rate)

            total_val_loss = list(total_val_loss) 
            try:
                if np.min(total_val_loss[-8:-4]) < np.min(total_val_loss[-4:]):
                    break
            except:pass
            if (epoch>4) & (np.min(total_val_loss) < 0.00300):break

        if is_pred==True:
            model.load_state_dict(torch.load(file_name))
            model.eval()
            if type(exp_data_deno)!=np.ndarray:
                [pred, pred_argmax, pred_max] = model_prediction(model, file_name, exp_data, model_index, is_preprocess=is_preprocess, PRE_SCALE=PRE_SCALE)
                total_pred.append([pred_exp, pred_deno, pred_exp_argmax, pred_exp_max, pred_deno_argmax, pred_deno_max])
            else:
                [pred, pred_argmax, pred_max] = model_prediction(model, file_name, exp_data, model_index, is_preprocess=is_preprocess, PRE_SCALE=PRE_SCALE)
                [pred_exp, pred_exp_argmax, pred_exp_max, pred_deno, pred_deno_argmax, pred_deno_max] = [pred, pred_argmax, pred_max] + model_prediction(model, file_name, exp_data_deno, model_index, is_preprocess=is_preprocess, PRE_SCALE=PRE_SCALE)
                total_pred.append([[mini_batch, learning_rate, selected_optim_name], pred_exp, pred_deno, pred_exp_argmax, pred_exp_max, pred_deno_argmax, pred_deno_max])
        if is_print_results==True:
            print([pred_exp, pred_deno])
            print([pred_exp_argmax, pred_exp_max, pred_deno_argmax, pred_deno_max])
        print("Done. Training Time: ", time.asctime()) 
    if is_pred==True:
        trained_model_list.append([file_name, total_pred])
    else:
        trained_model_list.append([file_name])
    
    print("Total consumed time: {}".format(time.time() - start_time))
    return total_loss, total_val_loss, total_acc, trained_model_list

# nomalization function for regression model
def Y_train_normalization(arr):
    A_min = np.min(arr[:,0])
    A_max = np.max(arr[:,0])
    B_min = np.min(arr[:,1])
    B_max = np.max(arr[:,1])
    arr[:, 0] = (A_max - arr[:, 0]) / (A_max - A_min)
    arr[:, 1] = (B_max - arr)[:, 1] / (B_max - B_min)
    return arr, A_min, A_max, B_min, B_max

# reversed normalization process for prediction results 
def reverse_normalization(A_pred, B_pred, A_min, A_max, B_min, B_max):
    A_reversed = A_max-A_pred*(A_max - A_min)
    B_reversed = B_max-B_pred*(B_max - B_min)
    return A_reversed, B_reversed 

def read_results(result1, result2, A_idx_list, B_idx_list, A_num, B_num, A_resol, B_resol, *, No_spin_threshold=0.2, optim='first'): 
    if optim=='first': optim_idx=0
    elif optim=='second': optim_idx=1

    heatmap_exp = np.zeros((len(A_idx_list)*A_num, len(B_idx_list)*B_num, 3))
    heatmap_deno = np.zeros((len(A_idx_list)*A_num, len(B_idx_list)*B_num, 3))

    for idxA, A_idx in enumerate(A_idx_list): 
        for idxB, B_idx in enumerate(B_idx_list): 

            if (idxA*len(B_idx_list) + idxB)%2 == 0:
                pred_exp = result1['A{}_B{}'.format(A_idx, B_idx)][3][0][1][optim_idx][1]
                pred_deno = result1['A{}_B{}'.format(A_idx, B_idx)][3][0][1][optim_idx][2]
            else:
                pred_exp = result2['A{}_B{}'.format(A_idx, B_idx)][3][0][1][optim_idx][1]
                pred_deno = result2['A{}_B{}'.format(A_idx, B_idx)][3][0][1][optim_idx][2]

            for idx_1 in range(A_num):
                for idx_2 in range(B_num):
                    heatmap_exp[idxA*A_num + idx_1, idxB*B_num + idx_2, :2] = A_idx+A_resol*idx_1, B_idx+B_resol*idx_2
                    if pred_exp[-1] > No_spin_threshold:
                        heatmap_exp[idxA*A_num + idx_1, idxB*B_num + idx_2, 2] = 0 
                    else: 
                        heatmap_exp[idxA*A_num + idx_1, idxB*B_num + idx_2, 2] = pred_exp[idx_1*B_num + idx_2]  

                    if pred_deno[-1] > No_spin_threshold:
                        heatmap_deno[idxA*A_num + idx_1, idxB*B_num + idx_2, 2] = 0 
                    else: 
                        heatmap_deno[idxA*A_num + idx_1, idxB*B_num + idx_2, 2] = pred_deno[idx_1*B_num + idx_2]  
    return heatmap_exp, heatmap_deno