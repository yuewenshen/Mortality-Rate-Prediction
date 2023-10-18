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

def MAE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    T = y_true
    re = np.abs(y_true - y_pre).mean()
    return re


def RMSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    e = (y_true + y_pre) / 2 + 1e-2
    re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()*100
    return re
input_size = 101*16
hidden_size = 50
num_layers = 1
output_size = 101*16
batch_size =1
epochs = 100
h1= 60
h2 = 70

class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size,h1,h2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.h1 = h1
        self.h2 = h2
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.output_size,self.h1)
        self.linear2 = nn.Linear(self.h1,self.h2)
        self.linear3 = nn.Linear(self.h2,101)



    def forward(self, x):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size)
        x, _ = self.lstm(x,(h_0,c_0))
        x = self.fc(x)

        # x = self.linear1(x)
        #
        # x = self.linear2(x)
        #
        # x = self.linear3(x)
        x = x[:, -1, :]

        return x





    # 数据集加载
my_data = genfromtxt('F:/csv文件大全/完整国家/npz文件/z-ln-16-0-100-67.csv', delimiter=',')

test_size = 16
train_set = my_data[:-test_size]
test_set = my_data[-test_size-8:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(train_set)  #归一化处理
# train_norm = train_set
train_norm = torch.FloatTensor(train_norm)   #转为tensor
test_norm = scaler.fit_transform(test_set)  #归一化处理
# test_norm = test_set
test_norm  = torch.FloatTensor(test_norm )   #转为tensor\

ture= my_data[-test_size:]
ture_norm = scaler.fit_transform(ture)

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
model = lstm(input_size ,
hidden_size ,
num_layers ,
output_size ,
batch_size,h1,h2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model.train()
start_time = time.time()
for epoch in range(epochs):
    for seq, y_train in train_data:
        # 每次更新参数前都梯度归零和初始化
        optimizer.zero_grad()
        # 注意这里要对样本进行reshape，
        # 转换成conv1d的input size（batch size, channel, series length）
        seq = seq.unsqueeze(0)
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
    seq = seq.unsqueeze(0)
    with torch.no_grad():
        pre.append(model(seq))
pred = pre[0]
for i in range(1, 16):
    pred = torch.cat([pred, pre[i]], dim=1)

yhat = pred.reshape(16,101*16).numpy()
yhat = scaler.inverse_transform(yhat).reshape(16*101*16,)
my_data = genfromtxt('F:/csv文件大全/完整国家/npz文件/z-ln-16-0-100-67.csv', delimiter=',')



y_test = ture.reshape(16*101*16,)


yhat2 = yhat.reshape(16,101*16)
print(yhat2)
MSE1=RMSE(y_test,yhat)
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
rmse1 = RMSE(ture_uk, pre_uk)
mape1 = MAPE(ture_uk, pre_uk)

mae2 = MAE(ture_franch, pre_franch)
rmse2 = RMSE(ture_franch, pre_franch)
mape2 = MAPE(ture_franch, pre_franch)

mae3 = MAE(ture_spain, pre_spain)
rmse3 = RMSE(ture_spain, pre_spain)
mape3 = MAPE(ture_spain, pre_spain)

mae4 = MAE(ture_italy, pre_italy)
rmse4 = RMSE(ture_italy, pre_italy)
mape4 = MAPE(ture_italy, pre_italy)

print("英国 rmes mae mape ", rmse1, mae1, mape1)
print("法国指标：rmse,mae,mape", rmse2, mae2, mape2)
print("意大利指标：rmse,mae,mape", rmse3, mae3, mape3)
print("西班牙指标：rmse,mae,mape", rmse4, mae4, mape4)

p1 = all_predict_value1[:,0:101]
t1 = all_y_true1[:,0:101]

p2 = all_predict_value1[:,101:202]
t2 = all_y_true1[:,101:202]

p3 = all_predict_value1[:,202:303]
t3= all_y_true1[:,202:303]

p4 = all_predict_value1[:,303:404]
t4 = all_y_true1[:,303:404]

p5 = all_predict_value1[:,404:505]
t5 = all_y_true1[:,404:505]


p6 = all_predict_value1[:,505:606]
t6 = all_y_true1[:,505:606]

p7 = all_predict_value1[:,606:707]
t7 = all_y_true1[:,606:707]

p8 = all_predict_value1[:,707:808]
t8 = all_y_true1[:,707:808]

p9 = all_predict_value1[:,808:909]
t9 = all_y_true1[:,808:909]

p10 = all_predict_value1[:,909:1010]
t10 = all_y_true1[:,909:1010]

p11 = all_predict_value1[:,1010:1111]
t11 = all_y_true1[:,1010:1111]

p12 = all_predict_value1[:,1111:1212]
t12 = all_y_true1[:,1111:1212]

p13 = all_predict_value1[:,1212:1313]
t13 = all_y_true1[:,1212:1313]

p14 = all_predict_value1[:,1313:1414]
t14 = all_y_true1[:,1313:1414]

p15 = all_predict_value1[:,1414:1515]
t15 = all_y_true1[:,1414:1515]

p16 = all_predict_value1[:,1515:1616]
t16 = all_y_true1[:,1515:1616]


mae1 = MAE(ture_uk,pre_uk)
rmse1 = RMSE(ture_uk,pre_uk)
mape1= MAPE(ture_uk,pre_uk)

mae2 = MAE(ture_franch,pre_franch)
rmse2 = RMSE(ture_franch,pre_franch)
mape2= MAPE(ture_franch,pre_franch)

mae3 = MAE(ture_spain,pre_spain)
rmse3 = RMSE(ture_spain,pre_spain)
mape3 = MAPE(ture_spain,pre_spain)

mae4 = MAE(ture_italy,pre_italy)
rmse4 = RMSE(ture_italy,pre_italy)
mape4 = MAPE(ture_italy,pre_italy)

mae111 = MAE(t1,p1)
rmse111 = RMSE(t1,p1)
mape111 = MAPE(t1,p1)

mae222 = MAE(t2,p2)
rmse222 = RMSE(t2,p2)
mape222 = MAPE(t2,p2)

mae333 = MAE(t3,p3)
rmse333 = RMSE(t3,p3)
mape333 = MAPE(t3,p3)

mae444 = MAE(t4,p4)
rmse444 = RMSE(t4,p4)
mape444 = MAPE(t4,p4)

mae5 = MAE(t5,p5)
rmse5 = RMSE(t5,p5)
mape5 = MAPE(t5,p5)

mae6 = MAE(t6,p6)
rmse6 = RMSE(t6,p6)
mape6 = MAPE(t6,p6)

mae7 = MAE(t7,p7)
rmse7 = RMSE(t7,p7)
mape7 = MAPE(t7,p7)
mae8 = MAE(t8,p8)
rmse8 = RMSE(t8,p8)
mape8 = MAPE(t8,p8)
mae9 = MAE(t9,p9)
rmse9 = RMSE(t9,p9)
mape9 = MAPE(t9,p9)
mae10 = MAE(t10,p10)
rmse10 = RMSE(t10,p10)
mape10 = MAPE(t10,p10)
mae11 = MAE(t11,p11)
rmse11 = RMSE(t11,p11)
mape11 = MAPE(t11,p11)
mae12 = MAE(t12,p12)
rmse12 = RMSE(t12,p12)
mape12 = MAPE(t12,p12)
mae13 = MAE(t13,p13)
rmse13 = RMSE(t13,p13)
mape13 = MAPE(t13,p13)
mae14 = MAE(t14,p14)
rmse14 = RMSE(t14,p14)
mape14 = MAPE(t14,p14)
mae15 = MAE(t15,p15)
rmse15 = RMSE(t15,p15)
mape15 = MAPE(t15,p15)
mae16 = MAE(t16,p16)
rmse16 = RMSE(t16,p16)
mape16 = MAPE(t16,p16)
print("丹麦指标：rmse,mae,mape",rmse111,mae111,mape111)
print("法国指标：rmse,mae,mape",rmse222,mae222,mape222)
print("芬兰指标：rmse,mae,mape",rmse333,mae333,mape333)
print("荷兰指标：rmse,mae,mape", rmse444, mae444, mape444)
print("挪威指标：rmse,mae,mape", rmse5, mae5, mape5)
print("瑞典指标：rmse,mae,mape", rmse6, mae6, mape6)
print("西班牙指标：rmse,mae,mape", rmse7, mae7, mape7)
print("比利时指标：rmse,mae,mape", rmse8, mae8, mape8)
print("英国指标：rmse,mae,mape", rmse9, mae9, mape9)
print("瑞士指标：rmse,mae,mape", rmse10, mae10, mape10)
print("意大利指标：rmse,mae,mape", rmse11, mae11, mape11)
print("奥地利指标：rmse,mae,mape", rmse12, mae12, mape12)
print("葡萄牙指标：rmse,mae,mape", rmse13, mae13, mape13)
print("匈牙利指标：rmse,mae,mape", rmse14, mae14, mape14)
print("斯洛伐克指标：rmse,mae,mape", rmse15, mae15, mape15)
print("捷克指标：rmse,mae,mape", rmse16, mae16, mape16)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)

pre1 = np.around(pre,  # numpy数组或列表
        decimals=5  # 保留几位小数
    )
ture1 = np.around(ture,decimals=5)
# np.savetxt('F:/csv文件大全/最终版/LSTM-TURE.txt', ture1, fmt=['%.06f,'] * ture1.shape[1], newline='\r\n')
# np.savetxt('F:/csv文件大全/最终版/LSTM-PRE.txt', pre1, fmt=['%.06f,'] * pre1.shape[1], newline='\r\n')
np.savetxt('F:/csv文件大全/最终版/lstm的测试/LSTM-TURE1.txt', ture1, fmt=['%.06f,'] * ture1.shape[1], newline='\r\n')
np.savetxt('F:/csv文件大全/最终版/lstm的测试/LSTM-PRE1.txt', pre1, fmt=['%.06f,'] * pre1.shape[1], newline='\r\n')
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
