import pickle

import torch
import numpy as np
import os
from .utils import print_log, Standardscaler, vrange, choose_model
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from datetime import datetime


import warnings
warnings.filterwarnings('ignore')

def preparation_ETT(dataset, seq_len, pred_len, day_step):
    ett = dataset
    if day_step == 'h':
        month_length = 30 * 24
    if day_step == 'm':
        month_length = 30 * 24 * 4
    ett = ett[:20*month_length]
    
    times = ett['date']
    times = times.to_numpy(dtype='datetime64[ns]')
    # print(len(times))

    num = ett[ett.columns[1:]].values  # (14400, 7) 
    num = num.reshape(num.shape[0], num.shape[1], 1)  # (14400, 7, 1)

    tod = np.zeros(num.shape)  # time of day
    dow = np.zeros(num.shape)  # day of week
    dom = np.zeros(num.shape)  # day of month
    doy = np.zeros(num.shape)  # day of year

    for t in range(len(times)):
        time = times[t]
        time = pd.to_datetime(time)
        tod[t, :, 0] = tod[t, :, 0] + (time.hour + time.minute / 60) / (24 - 1) - 0.5
        dow[t, :, 0] = dow[t, :, 0] + time.weekday() / (7 - 1) - 0.5
        dom[t, :, 0] = dom[t, :, 0] + (time.day - 1) / (31 - 1) - 0.5
        doy[t, :, 0] = doy[t, :, 0] + (time - datetime(time.year,1,1)).days / (366 - 1) - 0.5
        

    data = np.concatenate((num, tod, dow, dom, doy), axis=2)  # (14400, 7, 5)

    # index setting,  96 -> 336, train:val:test = 12m: 4m: 4m
    border1s = [0, 12 * month_length - seq_len, 16 * month_length - seq_len]
    border2s = [12 * month_length, 16 * month_length, 20 * month_length]

    length = [border2s[i] - border1s[i] - seq_len - pred_len + 1 for i in range(3)]
    train_len, val_len, test_len = length
    train, val, test = np.zeros((train_len, 3)), np.zeros((val_len, 3)), np.zeros((test_len, 3))

    for i in range(train_len):
        train[i][0] = border1s[0] + i
        train[i][1] = border1s[0] + i + seq_len
        train[i][2] = border1s[0] + i + seq_len + pred_len

    for i in range(val_len):
        val[i][0] = border1s[1] + i
        val[i][1] = border1s[1] + i + seq_len
        val[i][2] = border1s[1] + i + seq_len + pred_len

    for i in range(test_len):
        test[i][0] = border1s[2] + i
        test[i][1] = border1s[2] + i + seq_len
        test[i][2] = border1s[2] + i + seq_len + pred_len

    train, val, test = train.astype(int), val.astype(int), test.astype(int)
    index = dict()
    index['train'] = train
    index['val'] = val
    index['test'] = test
    return data, index


def preparation_solar(dataset, seq_len, pred_len):
    num = dataset.values  
    num = num.reshape(num.shape[0], num.shape[1], 1)  

    tod = np.zeros(num.shape)  # zeros for solar
    data = np.concatenate((num, tod), axis=2)  

    # train:val:test = 7: 1: 2
    data_len = data.shape[0]
    num_train = int(data_len * 0.7)
    num_test = int(data_len * 0.2)
    num_val = data_len - num_train - num_test
    
    border1s = [0, num_train - seq_len, data_len - num_test - seq_len]
    border2s = [num_train, num_train + num_val, data_len]

    length = [border2s[i] - border1s[i] - seq_len - pred_len + 1 for i in range(3)]
    train_len, val_len, test_len = length
    train, val, test = np.zeros((train_len, 3)), np.zeros((val_len, 3)), np.zeros((test_len, 3))

    for i in range(train_len):
        train[i][0] = border1s[0] + i
        train[i][1] = border1s[0] + i + seq_len
        train[i][2] = border1s[0] + i + seq_len + pred_len

    for i in range(val_len):
        val[i][0] = border1s[1] + i
        val[i][1] = border1s[1] + i + seq_len
        val[i][2] = border1s[1] + i + seq_len + pred_len

    for i in range(test_len):
        test[i][0] = border1s[2] + i
        test[i][1] = border1s[2] + i + seq_len
        test[i][2] = border1s[2] + i + seq_len + pred_len

    train, val, test = train.astype(int), val.astype(int), test.astype(int)
    index = dict()
    index['train'] = train
    index['val'] = val
    index['test'] = test
    return data, index


def preparation_custom(dataset, seq_len, pred_len):
    times = dataset['date']
    times = pd.to_datetime(times.values)
    times = times.to_numpy(dtype='datetime64[ns]')
    # print(len(times))

    num = dataset[dataset.columns[1:]].values  
    num = num.reshape(num.shape[0], num.shape[1], 1)  

    tod = np.zeros(num.shape)  # time of day
    dow = np.zeros(num.shape)  # day of week
    dom = np.zeros(num.shape)  # day of month
    doy = np.zeros(num.shape)  # day of year

    for t in range(len(times)):
        time = times[t]
        time = pd.to_datetime(time)
        tod[t, :, 0] = tod[t, :, 0] + (time.hour + time.minute / 60) / (24 - 1) - 0.5
        dow[t, :, 0] = dow[t, :, 0] + time.weekday() / (7 - 1) - 0.5
        dom[t, :, 0] = dom[t, :, 0] + (time.day - 1) / (31 - 1) - 0.5
        doy[t, :, 0] = doy[t, :, 0] + (time - datetime(time.year,1,1)).days / (366 - 1) - 0.5
        

    data = np.concatenate((num, tod, dow, dom, doy), axis=2)  

    # train:val:test = 7: 1: 2
    data_len = data.shape[0]
    num_train = int(data_len * 0.7)
    num_test = int(data_len * 0.2)
    num_val = data_len - num_train - num_test
    
    border1s = [0, num_train - seq_len, data_len - num_test - seq_len]
    border2s = [num_train, num_train + num_val, data_len]

    length = [border2s[i] - border1s[i] - seq_len - pred_len + 1 for i in range(3)]
    train_len, val_len, test_len = length
    train, val, test = np.zeros((train_len, 3)), np.zeros((val_len, 3)), np.zeros((test_len, 3))

    for i in range(train_len):
        train[i][0] = border1s[0] + i
        train[i][1] = border1s[0] + i + seq_len
        train[i][2] = border1s[0] + i + seq_len + pred_len

    for i in range(val_len):
        val[i][0] = border1s[1] + i
        val[i][1] = border1s[1] + i + seq_len
        val[i][2] = border1s[1] + i + seq_len + pred_len

    for i in range(test_len):
        test[i][0] = border1s[2] + i
        test[i][1] = border1s[2] + i + seq_len
        test[i][2] = border1s[2] + i + seq_len + pred_len

    train, val, test = train.astype(int), val.astype(int), test.astype(int)
    index = dict()
    index['train'] = train
    index['val'] = val
    index['test'] = test
    return data, index
    


def get_dataloaders_from_index_data(
        data_dir, seq_len=96, pred_len=336, tod=True, dow=True, dom=True, doy=True, batch_size=32, log=None, norm_each_node=True
):
    # ----- import data but from .csv and preliminary processing -----
    # print(data_dir)
    data_name = data_dir.split('/')[-1]
    
    if data_name[:3] == 'ETT': 
        day_step = data_name[3].lower()  
        dataset = pd.read_csv(os.path.join(data_dir, "%s.csv" % data_name))
        data, index = preparation_ETT(dataset=dataset, seq_len=seq_len, pred_len=pred_len, day_step=day_step)
    elif data_name == 'SOLAR':
        day_step = None
        df_raw = []
        with open(os.path.join(data_dir, "%s.txt" % data_name), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        data, index = preparation_solar(dataset=df_raw, seq_len=seq_len, pred_len=pred_len)                       
    else:
        day_step = None
        dataset = pd.read_csv(os.path.join(data_dir, "%s.csv" % data_name))
        data, index = preparation_custom(dataset=dataset, seq_len=seq_len, pred_len=pred_len)

   
    # ----- select models -----
    ### 这里要再加一个通过数据集判断模型参数的流程, 然后传回去
    model_select_args = choose_model(data, data_name)
    
    # ----- select features -----
    features = [0]
    if tod:  # time of day
        features.append(1)
    if dow:  # day of week
        features.append(2)
    if dom:  # day of month
        features.append(3)
    if doy:  # day of year
        features.append(4)
    data = data[..., features]
    print("Data shape: ", data.shape)
    
    # ----- scaler -----
    train_index = index["train"]
    val_index = index["val"]
    test_index = index["test"]
    
    if day_step == 'h':
        train_end = 12 * 30 * 24
    if day_step == 'm':
        train_end = 12 * 30 * 24 * 4
    if day_step is None:
        train_end = int(data.shape[0] * 0.7)
    
    data_train = data[:train_end, :, 0]  # [train_set_length, Nodes]
    
    if norm_each_node:
        mean, std = data_train.mean(axis=0, keepdims=True), data_train.std(axis=0, keepdims=True)
        scaler = Standardscaler(mean=mean, std=std)
        data_scaled = deepcopy(data[..., 0])
        data_scaled = scaler.transform(data_scaled)
        print("Data_scaled shape: ", data_scaled.shape)     
    else:    
        scaler = Standardscaler()
        scaler.fit(data_train)
        data_scaled = deepcopy(data[..., 0])
        data_scaled = scaler.transform(data_scaled)
        print("Data_scaled shape: ", data_scaled.shape)
    
    data_mark = data[:, 0, 1:]
    print("Data_mark shape: ", data_mark.shape)
    
    # ----- dataloader -----

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    # 在long term task 中，也需要y的time encoding
    x_train = data_scaled[x_train_index]
    x_train_mark = data_mark[x_train_index]
    y_train = data_scaled[y_train_index]
    y_train_mark = data_mark[y_train_index]
    x_val = data_scaled[x_val_index]
    x_val_mark = data_mark[x_val_index]
    y_val = data_scaled[y_val_index]
    y_val_mark = data_mark[y_val_index]
    x_test = data_scaled[x_test_index]
    x_test_mark = data_mark[x_test_index]
    y_test = data_scaled[y_test_index]
    y_test_mark = data_mark[y_test_index]

    # print(x_train.shape, x_val.shape, x_test.shape)
    print_log(f"Trainset:\tx-{x_train.shape}\tx_mark-{x_train_mark.shape}\ty-{y_train.shape}\ty_mark-{y_train_mark.shape}", log=log)
    print_log(f"Valset:\tx-{x_val.shape}\tx_mark-{x_val_mark.shape}\ty-{y_val.shape}\ty_mark-{y_val_mark.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\tx_mark-{x_test_mark.shape}\ty-{y_test.shape}\ty_mark-{y_test_mark.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(x_train_mark), torch.FloatTensor(y_train), torch.FloatTensor(y_train_mark)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(x_val_mark), torch.FloatTensor(y_val), torch.FloatTensor(y_val_mark)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(x_test_mark), torch.FloatTensor(y_test), torch.FloatTensor(y_test_mark)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler, model_select_args


if __name__ == "__main__":
    get_dataloaders_from_index_data('../data/ETTH1')