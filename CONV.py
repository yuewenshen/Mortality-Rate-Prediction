import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import math
import random
import matplotlib.pyplot as plt
from numpy import genfromtxt
#from web import predict
import torch
import torch.nn as nn
import time
window_size = 8

def MSE(target, prediction):
    error = []
    squaredError = []
    absError = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
        mse= math.pow(sum(squaredError) / len(squaredError), 0.5)
    #print("MSE = ", sum(squaredError) / len(squaredError))
   # print("RMSE = ", math.pow(sum(squaredError) / len(squaredError), 0.5))
    return mse
def MAE(targrt, prediction):
    n = len(targrt)
    absError = []
    for i in range(n):
        absError.append(np.abs(targrt[i] - prediction[i]))
    mae = sum(absError) / n
    #print("MAE=", mae)
    return  mae


def MAPE(targrt, prediction):
    n = len(targrt)
    absError = []
    for i in range(n):
        absError.append(np.abs((targrt[i] - prediction[i]) / targrt[i]))
    mape = sum(absError) / n * 100
   # print("MAPE=", mape, "%")
    return  mape



class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=101*16,out_channels=16,kernel_size=3)
        self.maxpooling = nn.MaxPool1d(2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(48,50)
        self.linear2 = nn.Linear(50,101*16)



    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x




    # 数据集加载
my_data = genfromtxt('F:/csv文件大全/完整国家/法国/法国T-ln.csv', delimiter=',')
my_data = my_data[:,1:102]
test_size = 16
train_set = my_data[:-test_size]
test_set = my_data[-test_size-8:]
ture= my_data[-test_size:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
# train_norm = scaler.fit_transform(train_set)  #归一化处理
train_norm = train_set
train_norm = torch.FloatTensor(train_norm)   #转为tensor
# test_norm = scaler.fit_transform(test_set)  #归一化处理
test_norm = test_set
test_norm  = torch.FloatTensor(test_norm )   #转为tensor

def input_data(seq,ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out
train_data = input_data(train_norm,window_size)
test_data = input_data(test_norm,window_size)



torch.manual_seed(101)
model =CNNnetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 500
# model.train()
start_time = time.time()
for epoch in range(epochs):
    for seq, y_train in train_data:
        # 每次更新参数前都梯度归零和初始化
        optimizer.zero_grad()
        # 注意这里要对样本进行reshape，
        # 转换成conv1d的input size（batch size, channel, series length）
        seq = seq.unsqueeze(0).permute(0,2,1)
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')
print(f'\nDuration: {time.time() - start_time:.0f} seconds')





model.eval()
pre = []
# 循环的每一步表示向时间序列向后滑动一格
for seq, y_test in test_data:
    seq = seq.unsqueeze(0).permute(0,2,1)
    with torch.no_grad():
        pre.append(model(seq))
pred = pre[0]
for i in range(1, 16):
    pred = torch.cat([pred, pre[i]], dim=1)

y_test = pred.reshape(16*101*16,).numpy()
yhat = ture.reshape(16*101*16,)

MSE1=MSE(y_test,yhat)
MAE1=MAE(y_test,yhat)
MAPE1=MAPE(y_test,yhat)
print("rmse",MSE1)
print("MAE",MAE1)
print("MAPE",MAPE1)

pre = yhat.reshape(16,1616)
ture = y_test.reshape(16,1616)
# def MAE(y_true, y_pre):
#     y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
#     y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
#     T = y_true
#     re = np.abs(y_true - y_pre).mean()
#     return re
#
#
# def RMSE(y_true, y_pre):
#     y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
#     y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
#     re = math.sqrt(((y_true - y_pre) ** 2).mean())
#     return re
#
#
# def MAPE(y_true, y_pre):
#     y_true = (y_true).detach().numpy().copy().reshape((-1, 1))
#     y_pre = (y_pre).detach().numpy().copy().reshape((-1, 1))
#     e = (y_true + y_pre) / 2 + 1e-2
#     re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()*100
#     return re

# y_test=y_test.reshape(16*1616,)
# yhat=yhat.reshape(16*1616,)
#
# pre = yhat.reshape(16,1616)
# ture = y_test.reshape(16,1616)
all_predict_value1 = pre
all_y_true1 = ture

pre_uk = all_predict_value1[:, 808:909]
pre_uk= pre_uk.reshape(16*101,)
ture_uk = all_y_true1[:, 808:909].reshape(16*101,)

pre_franch = all_predict_value1[:, 101:202].reshape(16*101,)
ture_franch = all_y_true1[:, 101:202].reshape(16*101,)

pre_italy = all_predict_value1[:, 1010:1111].reshape(16*101,)
ture_italy = all_y_true1[:, 1010:1111].reshape(16*101,)

pre_spain = all_predict_value1[:, 606:707].reshape(16*101,)
ture_spain = all_y_true1[:, 606:707].reshape(16*101,)

mae1 = MAE(ture_uk, pre_uk)
rmse1 = MSE(ture_uk, pre_uk)
mape1 = MAPE(ture_uk, pre_uk)

mae2 = MAE(ture_franch, pre_franch)
rmse2 = MSE(ture_franch, pre_franch)
mape2 = MAPE(ture_franch, pre_franch)

mae3 = MAE(ture_spain, pre_spain)
rmse3 = MSE(ture_spain, pre_spain)
mape3 = MAPE(ture_spain, pre_spain)

mae4 = MAE(ture_italy, pre_italy)
rmse4 = MSE(ture_italy, pre_italy)
mape4 = MAPE(ture_italy, pre_italy)

print("英国 rmes mae mape ", rmse1, mae1, mape1)
print("法国指标：rmse,mae,mape", rmse2, mae2, mape2)
print("意大利指标：rmse,mae,mape", rmse3, mae3, mape3)
print("西班牙指标：rmse,mae,mape", rmse4, mae4, mape4)



np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

pre1 = np.around(pre,  # numpy数组或列表
        decimals=5  # 保留几位小数
    )
ture1 = np.around(ture,decimals=5)
# np.savetxt('F:/csv文件大全/最终版/Dture-conv1d.txt', ture1, fmt=['%.06f,'] * ture1.shape[1], newline='\r\n')
# np.savetxt('F:/csv文件大全/最终版/Dpre-conv1d.txt', pre1, fmt=['%.06f,'] * pre1.shape[1], newline='\r\n')
#
#
#
#
# print(y_test)
# print(yhat)
# import matplotlib.pyplot as plt
# from numpy import genfromtxt
# import numpy as np
# my_data = genfromtxt('F:/csv文件大全/dd.csv', delimiter=',')
# fig, ax = plt.subplots()
# x = my_data
# y1 = y_test
# y2 = yhat
# ax.plot(x, y1, label='l',linewidth=0.5,color='b')
# ax.plot(x, y2, label='d',linewidth=0.5,color='r')
# ax.set_xlabel('x label') #设置x轴名称 x label
# ax.set_ylabel('y label') #设置y轴名称 y label
# ax.set_title('Simple Plot') #设置图名为Simple Plot
# ax.legend() #自动检测要在图例中显示的元素，并且显示
# plt.legend(bbox_to_anchor=(0.9, 0.5), loc=3, borderaxespad=0)
# plt.show() #图形可视化



#
# MSE1=MSE(y_test,yhat)
# MAE1=MAE(y_test,yhat)
# MAPE1=MAPE(y_test,yhat)
# print("rmse",MSE1)
# print("MAE",MAE1)
# print("MAPE",MAPE1)
