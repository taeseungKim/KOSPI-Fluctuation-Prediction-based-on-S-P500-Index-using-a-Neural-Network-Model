from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from math import log
from mpl_toolkits.mplot3d import axes3d
from sklearn.model_selection import cross_val_score
from collections import OrderedDict
import tensorflow as tf

kospi = pd.read_csv('/home/tskim/바탕화면/kospi(00.1.1_16.2.12).csv') #미국 주가 제외 x
sp = pd.read_csv('/home/tskim/바탕화면/s&p(00.1.1_16.2.12).csv')
sp = np.array(sp.dropna())
kospi = np.array(kospi.dropna())
kospi_indexs_all = kospi[:,1:7]
kospi_close = kospi[:,4:5] #close price
sp_all = sp[:,1:7]
sp_close = sp[:,4:5]

kospi_input = OrderedDict() #input dictionary로
for i in range(len(kospi)):
    kospi_input["".join(kospi[:, :1][i])] = kospi[:, 1:7][i]
sp_input = OrderedDict()
for i in range(len(sp)):
    sp_input["".join(sp[:,:1][i])] = sp[:,1:7][i]

del sp_input['1999-12-31']
sl = []
for i in range(0,len(sp_close)-1):
    sl.append(sp_close[i+1] - sp_close[i])
sl = np.array(sl).reshape(-1,)
sl_z = stats.zscore(sl).reshape(-1,1)

i=0
for k,v in sp_input.items():#하루 전 종가등락 dic에 병합
    sp_input[k] = np.append(sp_input[k],sl_z[i])
    i=i+1

rl = []
rl_past = []
for i in range(0, len(kospi_close) - 1): # label
    if kospi_close[i] < kospi_close[i + 1]:
        rl.append(0)
    else:
        rl.append(1)
for i in range(0, len(kospi_close) - 1): # label
    rl_past.append(kospi_close[i + 1] - kospi_close[i])

del kospi_input['2000-01-04']
kospi_input.popitem()#맨뒤의 값 삭제

rl_past = np.delete(rl_past,3977)
rl_past_z = stats.zscore(rl_past).reshape(-1,1)

a=0
for k,v in kospi_input.items():#하루 전 종가등락 dic에 병합
   kospi_input[k] = np.append(kospi_input[k],rl_past_z[a])
   a=a+1
##################### 한국 날짜 미국 날짜 같은 날만 남기고 제거
input = OrderedDict()
for k,v in kospi_input.items():
    arr = []
    if k in sp_input:
        arr.append(kospi_input[k])
        arr.append(sp_input[k])
        input[k] = arr
#####################making target
arr = []
for k,v in input.items():
    arr.append(v[0])

a=[] #answer
for i in range(len(arr)-1):
    if arr[i+1][3] > arr[i][3]:
        a.append(1) #종가 상승시 1
    else:
        a.append(0) #종가 하락,유지시 0
input.popitem()#remove last value
#####################
ko = []
sp = []
for k,v in input.items():
    ko.append(v[0])
    sp.append(v[1])
k_open = []
k_high = []
k_low = []
k_close = []
k_adjClose = []
k_volume = []
k_ud = []
for i in ko:
    k_open.append(i[0])
    k_high.append(i[1])
    k_low.append(i[2])
    k_close.append(i[3])
    k_adjClose.append(i[4])
    k_volume.append(i[5])
    k_ud.append(i[6])
k_open_z = stats.zscore(k_open)
k_high_z = stats.zscore(k_high)
k_low_z = stats.zscore(k_low)
k_close_z = stats.zscore(k_close)
k_adjClose_z = stats.zscore(k_adjClose)
k_volume_z = stats.zscore(k_volume)
k_ud_z = k_ud

s_open = []
s_high = []
s_low = []
s_close = []
s_adjClose = []
s_volume = []
s_ud = []
for i in sp:
    s_open.append(i[0])
    s_high.append(i[1])
    s_low.append(i[2])
    s_close.append(i[3])
    s_adjClose.append(i[4])
    s_volume.append(i[5])
    s_ud.append(i[6])
s_open_z = stats.zscore(s_open)
s_high_z = stats.zscore(s_high)
s_low_z = stats.zscore(s_low)
s_close_z = stats.zscore(s_close)
s_adjClose_z = stats.zscore(s_adjClose)
s_volume_z = stats.zscore(s_volume)
s_ud_z = s_ud

input_n = OrderedDict()
i = 0
for k,v in input.items():
    all = []
    k_a = []
    s_a = []
    k_a.append(k_open_z[i])
    k_a.append(k_high_z[i])
    k_a.append(k_low_z[i])
    k_a.append(k_close_z[i])
    k_a.append(k_adjClose_z[i])
    k_a.append(k_volume_z[i])
    k_a.append(k_ud_z[i])
    s_a.append(s_open_z[i])
    s_a.append(s_high_z[i])
    s_a.append(s_low_z[i])
    s_a.append(s_close_z[i])
    s_a.append(s_adjClose_z[i])
    s_a.append(s_volume_z[i])
    s_a.append(s_ud_z[i])
    all.append(k_a)
    all.append(s_a)
    input_n[k] = all
    i=i+1
#####################